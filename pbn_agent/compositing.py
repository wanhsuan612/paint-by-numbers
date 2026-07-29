from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class CompositeResult:
    image_path: Path
    cutout_path: Path
    report_path: Path


@dataclass(frozen=True)
class AlignmentResult:
    image_path: Path
    report_path: Path


def _estimate_border_key(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    band = max(2, round(min(height, width) * 0.01))
    border = np.concatenate(
        [
            rgb[:band].reshape(-1, 3),
            rgb[-band:].reshape(-1, 3),
            rgb[:, :band].reshape(-1, 3),
            rgb[:, -band:].reshape(-1, 3),
        ]
    )
    return np.median(border, axis=0).astype(np.float32)


def align_stage_a_to_source(
    subject_path: Path,
    source_path: Path,
    output_path: Path,
) -> AlignmentResult:
    subject = np.asarray(Image.open(subject_path).convert("RGB"))
    height, width = subject.shape[:2]
    key = _estimate_border_key(subject)
    red, green, blue = key.tolist()
    if red < 170 or blue < 170 or green > 130:
        raise ValueError("Stage A alignment requires a flat magenta chroma background")

    source = Image.open(source_path).convert("RGB")
    source_canvas = np.asarray(
        ImageOps.fit(source, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    )
    subject_mask = (
        np.linalg.norm(subject.astype(np.float32) - key[None, None, :], axis=2) > 70
    ).astype(np.uint8) * 255
    sift = cv2.SIFT_create(nfeatures=8000)
    subject_points, subject_descriptors = sift.detectAndCompute(
        cv2.cvtColor(subject, cv2.COLOR_RGB2GRAY), subject_mask
    )
    source_points, source_descriptors = sift.detectAndCompute(
        cv2.cvtColor(source_canvas, cv2.COLOR_RGB2GRAY), None
    )
    if subject_descriptors is None or source_descriptors is None:
        raise ValueError("Could not detect enough features to align Stage A")
    candidates = cv2.BFMatcher().knnMatch(subject_descriptors, source_descriptors, k=2)
    matches = [first for first, second in candidates if first.distance < 0.65 * second.distance]
    if len(matches) < 6:
        raise ValueError(f"Could not align Stage A reliably; only {len(matches)} feature matches")

    subject_xy = np.float32([subject_points[match.queryIdx].pt for match in matches])
    source_xy = np.float32([source_points[match.trainIdx].pt for match in matches])
    transform, inliers = cv2.estimateAffinePartial2D(
        subject_xy,
        source_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=5,
        maxIters=10000,
        confidence=0.999,
    )
    inlier_count = 0 if inliers is None else int(inliers.sum())
    if transform is None or inlier_count < 4:
        raise ValueError(f"Could not align Stage A reliably; only {inlier_count} inliers")

    scale = float(np.hypot(transform[0, 0], transform[0, 1]))
    rotation_degrees = float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))
    if not 0.4 <= scale <= 1.4 or abs(rotation_degrees) > 15:
        raise ValueError(
            f"Stage A alignment transform is implausible: scale={scale:.3f}, rotation={rotation_degrees:.3f}"
        )

    aligned = cv2.warpAffine(
        subject,
        transform,
        (width, height),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(int(round(value)) for value in key),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(aligned).save(output_path)
    report_path = output_path.with_suffix(".alignment.json")
    report_path.write_text(
        json.dumps(
            {
                "subject_path": str(subject_path),
                "source_path": str(source_path),
                "output_path": str(output_path),
                "feature_match_count": len(matches),
                "inlier_count": inlier_count,
                "scale": round(scale, 6),
                "rotation_degrees": round(rotation_degrees, 6),
                "transform": [[round(float(value), 8) for value in row] for row in transform],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return AlignmentResult(output_path, report_path)


def create_chroma_cutout(subject_path: Path, cutout_path: Path) -> dict[str, object]:
    rgb = np.asarray(Image.open(subject_path).convert("RGB"))
    key = _estimate_border_key(rgb)
    red, green, blue = key.tolist()
    if red < 170 or blue < 170 or green > 130:
        raise ValueError(
            "Stage A must use a flat magenta chroma background before deterministic C compositing"
        )

    distance = np.linalg.norm(rgb.astype(np.float32) - key[None, None, :], axis=2)
    # A hard mask intentionally drops antialiased pixels blended with the key
    # color. Keeping those pixels would preserve a magenta halo in Stage C.
    opaque_distance = 95.0
    rgb16 = rgb.astype(np.int16)
    magenta_strength = np.minimum(rgb16[..., 0], rgb16[..., 2]) - rgb16[..., 1]
    foreground = (distance >= opaque_distance) & (magenta_strength < 70)
    alpha = np.where(foreground, 255, 0).astype(np.uint8)

    coverage = float(np.count_nonzero(alpha >= 128) / alpha.size)
    if not 0.05 <= coverage <= 0.70:
        raise ValueError(f"Unexpected Stage A foreground coverage: {coverage:.3f}")

    rgba = np.dstack([rgb, alpha])
    cutout_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(cutout_path)
    return {
        "estimated_key_rgb": [round(float(value)) for value in key],
        "foreground_coverage": round(coverage, 6),
        "transparent_pixel_count": int(np.count_nonzero(alpha == 0)),
        "partial_pixel_count": int(np.count_nonzero((alpha > 0) & (alpha < 255))),
        "opaque_pixel_count": int(np.count_nonzero(alpha == 255)),
    }


def composite_stage_c(subject_path: Path, background_path: Path, output_path: Path) -> CompositeResult:
    cutout_path = output_path.with_name(f"{output_path.stem}-subject-cutout.png")
    report_path = output_path.with_suffix(".composite.json")
    report = create_chroma_cutout(subject_path, cutout_path)

    foreground = Image.open(cutout_path).convert("RGBA")
    background = Image.open(background_path).convert("RGBA")
    if foreground.size != background.size:
        raise ValueError(
            f"Stage A and B canvas sizes must match exactly: {foreground.size} != {background.size}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    composite = Image.alpha_composite(background, foreground).convert("RGB")
    composite.save(output_path)

    foreground_array = np.asarray(foreground)
    composite_array = np.asarray(composite)
    opaque = foreground_array[..., 3] == 255
    if np.any(opaque):
        delta = np.abs(
            composite_array[opaque].astype(np.int16) - foreground_array[..., :3][opaque].astype(np.int16)
        )
        maximum_delta = int(delta.max())
    else:
        maximum_delta = 0
    report.update(
        {
            "subject_path": str(subject_path),
            "background_path": str(background_path),
            "output_path": str(output_path),
            "opaque_subject_max_rgb_delta": maximum_delta,
            "pixel_identity_passed": maximum_delta == 0,
        }
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return CompositeResult(output_path, cutout_path, report_path)
