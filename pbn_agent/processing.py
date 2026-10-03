from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .models import PaletteEntry, RegionReport, WorkflowSpec


@dataclass(frozen=True)
class ProcessingResult:
    simplified_image: Path
    completed_preview: Path
    blank_line_art: Path
    numbered_line_art: Path
    palette_image: Path
    palette_json: Path
    region_report_json: Path


def _center_crop_to_aspect(array: np.ndarray, aspect_ratio: float) -> np.ndarray:
    height, width = array.shape[:2]
    current = width / height
    if abs(current - aspect_ratio) < 1e-6:
        return array
    if current > aspect_ratio:
        target_width = round(height * aspect_ratio)
        left = (width - target_width) // 2
        return array[:, left : left + target_width]
    target_height = round(width / aspect_ratio)
    top = (height - target_height) // 2
    return array[top : top + target_height, :]


def _resize_working(array: np.ndarray, width: int, interpolation: int) -> np.ndarray:
    height = round(array.shape[0] * width / array.shape[1])
    return cv2.resize(array, (width, height), interpolation=interpolation)


def _auto_protect_highlights(rgb: np.ndarray) -> np.ndarray:
    """Protect tiny bright islands touching dark details, such as eye catchlights."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bright_threshold = max(225, int(np.percentile(gray, 97.5)))
    bright = (gray >= bright_threshold).astype(np.uint8)
    dark = (gray <= 135).astype(np.uint8)
    dark_nearby = cv2.dilate(dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    candidates = (bright & dark_nearby).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    protected = np.zeros_like(candidates)
    max_area = max(12, round(rgb.shape[0] * rgb.shape[1] * 0.00008))
    inner_kernel = np.ones((5, 5), dtype=np.uint8)
    outer_kernel = np.ones((11, 11), dtype=np.uint8)
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if not 1 <= area <= max_area:
            continue
        x, y, width, height = (int(value) for value in stats[component_id, :4])
        left, top = max(0, x - 6), max(0, y - 6)
        right, bottom = min(rgb.shape[1], x + width + 6), min(rgb.shape[0], y + height + 6)
        component = (components[top:bottom, left:right] == component_id).astype(np.uint8)
        # Ring just outside the anti-aliased edge of the highlight.
        ring = cv2.dilate(component, outer_kernel).astype(bool) & ~cv2.dilate(component, inner_kernel).astype(bool)
        surroundings = gray[top:bottom, left:right][ring]
        # A catchlight sits in a dark pupil or mid-tone iris on every side; bright fur edges
        # always have light fur or background on one side.
        if (surroundings < 205).mean() >= 0.8 and (surroundings <= 135).mean() >= 0.3:
            protected[top:bottom, left:right][component.astype(bool)] = 1
    return protected.astype(bool)


def _prepare_mask(
    mask_path: Path | None,
    source_shape: tuple[int, int],
    aspect: float,
    working_shape: tuple[int, int],
) -> np.ndarray:
    if mask_path is None:
        return np.zeros(working_shape, dtype=bool)
    mask = np.asarray(Image.open(mask_path).convert("L"))
    if mask.shape != source_shape:
        mask = cv2.resize(mask, (source_shape[1], source_shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = _center_crop_to_aspect(mask, aspect)
    mask = cv2.resize(mask, (working_shape[1], working_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask >= 128


def _weighted_samples(
    rgb: np.ndarray,
    protected: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    flat = rgb.reshape(-1, 3).astype(np.float32)
    protected_flat = protected.reshape(-1)
    ordinary_indices = np.flatnonzero(~protected_flat)
    protected_indices = np.flatnonzero(protected_flat)
    protected_budget = min(len(protected_indices), max_samples // 4)
    ordinary_budget = min(len(ordinary_indices), max_samples - protected_budget)
    if ordinary_budget:
        ordinary_indices = rng.choice(ordinary_indices, ordinary_budget, replace=False)
    if protected_budget and len(protected_indices) > protected_budget:
        protected_indices = rng.choice(protected_indices, protected_budget, replace=False)
    indices = np.concatenate([ordinary_indices, protected_indices])
    samples = flat[indices]
    weights = np.ones(len(indices), dtype=np.float32)
    if protected_budget:
        weights[-protected_budget:] = 80.0
    return samples, weights


def _weighted_kmeans(
    samples: np.ndarray,
    weights: np.ndarray,
    color_count: int,
    forced_colors: list[tuple[int, int, int]],
    seed: int = 7,
    iterations: int = 16,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    forced = np.asarray(forced_colors[:color_count], dtype=np.float32).reshape(-1, 3)
    centers: list[np.ndarray] = [color for color in forced]
    if not centers:
        centers.append(samples[int(rng.integers(0, len(samples)))])
    while len(centers) < color_count:
        center_array = np.asarray(centers, dtype=np.float32)
        distance = ((samples[:, None, :] - center_array[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        probability = distance * weights
        total = float(probability.sum())
        if total <= 0:
            centers.append(samples[int(rng.integers(0, len(samples)))])
        else:
            centers.append(samples[int(rng.choice(len(samples), p=probability / total))])
    center_array = np.asarray(centers, dtype=np.float32)
    locked_count = len(forced)
    for _ in range(iterations):
        distances = ((samples[:, None, :] - center_array[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        updated = center_array.copy()
        for index in range(locked_count, color_count):
            members = labels == index
            if members.any():
                updated[index] = np.average(samples[members], axis=0, weights=weights[members])
        if np.max(np.abs(updated - center_array)) < 0.35:
            center_array = updated
            break
        center_array = updated
    return np.clip(np.rint(center_array), 0, 255).astype(np.uint8)


def _assign_palette(rgb: np.ndarray, palette: np.ndarray, chunk_size: int = 100_000) -> np.ndarray:
    flat = rgb.reshape(-1, 3).astype(np.float32)
    labels = np.empty(len(flat), dtype=np.uint8)
    palette_float = palette.astype(np.float32)
    for start in range(0, len(flat), chunk_size):
        chunk = flat[start : start + chunk_size]
        distances = ((chunk[:, None, :] - palette_float[None, :, :]) ** 2).sum(axis=2)
        labels[start : start + len(chunk)] = distances.argmin(axis=1).astype(np.uint8)
    return labels.reshape(rgb.shape[:2])


def _nearest_neighbor_label(
    current: int,
    neighbors: np.ndarray,
    palette: np.ndarray,
) -> int | None:
    unique, counts = np.unique(neighbors[neighbors != current], return_counts=True)
    if not len(unique):
        return None
    current_color = palette[current].astype(np.float32)
    distances = ((palette[unique].astype(np.float32) - current_color) ** 2).sum(axis=1)
    # Prefer color similarity, using boundary frequency as a stable tie breaker.
    score = distances - np.log1p(counts) * 15.0
    return int(unique[int(score.argmin())])


def _merge_small_components(
    labels: np.ndarray,
    palette: np.ndarray,
    minimum_area: int,
    protected: np.ndarray,
    minimum_radius: float = 0.0,
    passes: int = 3,
) -> np.ndarray:
    merged = labels.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(passes):
        changed = 0
        for color_index in range(len(palette)):
            count, components, stats, _ = cv2.connectedComponentsWithStats(
                (merged == color_index).astype(np.uint8), 8
            )
            for component_id in range(1, count):
                area = int(stats[component_id, cv2.CC_STAT_AREA])
                x = int(stats[component_id, cv2.CC_STAT_LEFT])
                y = int(stats[component_id, cv2.CC_STAT_TOP])
                width = int(stats[component_id, cv2.CC_STAT_WIDTH])
                height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
                component_roi = components[y : y + height, x : x + width] == component_id
                if area >= minimum_area:
                    if minimum_radius <= 0:
                        continue
                    padded = np.pad(component_roi.astype(np.uint8), 1)
                    if cv2.distanceTransform(padded, cv2.DIST_L2, 5).max() >= minimum_radius:
                        continue
                if np.any(protected[y : y + height, x : x + width] & component_roi):
                    continue
                left = max(0, x - 1)
                top = max(0, y - 1)
                right = min(merged.shape[1], x + width + 1)
                bottom = min(merged.shape[0], y + height + 1)
                expanded_component = components[top:bottom, left:right] == component_id
                ring = cv2.dilate(expanded_component.astype(np.uint8), kernel).astype(bool) & ~expanded_component
                neighbor = _nearest_neighbor_label(
                    color_index,
                    merged[top:bottom, left:right][ring],
                    palette,
                )
                if neighbor is not None:
                    region = merged[y : y + height, x : x + width]
                    region[component_roi] = neighbor
                    changed += area
        if changed == 0:
            break
    return merged


def _smooth_labels(
    labels: np.ndarray,
    protected: np.ndarray,
    sigma: float,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Replace stair-stepped label edges with smooth curves; protected pixels keep their label."""
    width, height = output_size or (labels.shape[1], labels.shape[0])
    best_score = np.full((height, width), -1.0, dtype=np.float32)
    smoothed = np.zeros((height, width), dtype=labels.dtype)
    for label in np.unique(labels):
        score = cv2.GaussianBlur((labels == label).astype(np.float32), (0, 0), sigma)
        if output_size is not None:
            score = cv2.resize(score, output_size, interpolation=cv2.INTER_LINEAR)
        better = score > best_score
        best_score[better] = score[better]
        smoothed[better] = label
    if output_size is None:
        smoothed[protected] = labels[protected]
    else:
        keep = cv2.resize(protected.astype(np.uint8), output_size, interpolation=cv2.INTER_NEAREST).astype(bool)
        nearest = cv2.resize(labels, output_size, interpolation=cv2.INTER_NEAREST)
        smoothed[keep] = nearest[keep]
    return smoothed


def _boundary_image(labels: np.ndarray, line_width: int = 2) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary = cv2.dilate(boundary.astype(np.uint8), np.ones((line_width, line_width), np.uint8)).astype(bool)
    line_art = np.full(labels.shape, 255, dtype=np.uint8)
    line_art[boundary] = 0
    return line_art


def _draw_labels(
    blank_final: np.ndarray,
    labels: np.ndarray,
    palette: np.ndarray,
    scale_x: float,
    scale_y: float,
    desired_height_px: int,
    protected: np.ndarray,
) -> tuple[np.ndarray, RegionReport]:
    numbered = cv2.cvtColor(blank_final, cv2.COLOR_GRAY2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_height = cv2.getTextSize("36", font, 1.0, 1)[0][1]
    font_scale = max(0.45, desired_height_px / max(base_height, 1))
    thickness = max(1, round(desired_height_px / 22))
    region_count = internal = vertical = callout = protected_count = 0

    for color_index in range(len(palette)):
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            (labels == color_index).astype(np.uint8), 8
        )
        text = str(color_index + 1)
        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        digit_size, _ = cv2.getTextSize("8", font, font_scale, thickness)
        for component_id in range(1, count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < 4:
                continue
            region_count += 1
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            roi = (components[y : y + height, x : x + width] == component_id).astype(np.uint8)
            if np.any(protected[y : y + height, x : x + width] & roi.astype(bool)):
                protected_count += 1
            distance = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
            _, max_distance, _, max_point = cv2.minMaxLoc(distance)
            center_x = round((x + max_point[0]) * scale_x)
            center_y = round((y + max_point[1]) * scale_y)
            full_width = width * scale_x
            full_height = height * scale_y
            radius = max_distance * min(scale_x, scale_y)

            if (
                full_width >= text_size[0] + 8
                and full_height >= text_size[1] + baseline + 8
                and radius >= text_size[1] * 0.55
            ):
                origin = (center_x - text_size[0] // 2, center_y + text_size[1] // 2)
                cv2.putText(numbered, text, origin, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
                internal += 1
            elif (
                len(text) == 2
                and full_width >= digit_size[0] + 8
                and full_height >= digit_size[1] * 2 + 14
            ):
                first_y = center_y - digit_size[1] // 2
                for offset, digit in enumerate(text):
                    digit_width = cv2.getTextSize(digit, font, font_scale, thickness)[0][0]
                    cv2.putText(
                        numbered,
                        digit,
                        (center_x - digit_width // 2, first_y + offset * (digit_size[1] + 4)),
                        font,
                        font_scale,
                        (0, 0, 0),
                        thickness,
                        cv2.LINE_AA,
                    )
                vertical += 1
            else:
                target_x = min(numbered.shape[1] - text_size[0] - 4, center_x + max(12, round(full_width / 2)))
                target_y = min(numbered.shape[0] - baseline - 4, max(text_size[1] + 4, center_y))
                cv2.line(numbered, (center_x, center_y), (target_x, target_y), (80, 80, 80), 1, cv2.LINE_AA)
                cv2.putText(
                    numbered,
                    text,
                    (target_x, target_y),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA,
                )
                callout += 1

    return numbered, RegionReport(
        region_count=region_count,
        internal_label_count=internal,
        vertical_label_count=vertical,
        callout_label_count=callout,
        protected_region_count=protected_count,
    )


def _palette_art(palette: np.ndarray, counts: np.ndarray) -> np.ndarray:
    columns = 6
    rows = int(np.ceil(len(palette) / columns))
    cell_width, cell_height = 210, 110
    canvas = np.full((rows * cell_height, columns * cell_width, 3), 255, dtype=np.uint8)
    for index, color in enumerate(palette):
        row, column = divmod(index, columns)
        x, y = column * cell_width, row * cell_height
        display_color = tuple(int(value) for value in color[::-1])
        cv2.rectangle(canvas, (x + 8, y + 8), (x + 82, y + 82), display_color, -1)
        cv2.rectangle(canvas, (x + 8, y + 8), (x + 82, y + 82), (40, 40, 40), 1)
        cv2.putText(canvas, str(index + 1), (x + 92, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        cv2.putText(
            canvas,
            f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}",
            (x + 92, y + 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (50, 50, 50),
            1,
        )
        cv2.putText(
            canvas,
            f"{int(counts[index])} px",
            (x + 92, y + 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (90, 90, 90),
            1,
        )
    return canvas


def create_paint_by_numbers(
    image_path: Path,
    output_directory: Path,
    spec: WorkflowSpec,
    protected_mask_path: Path | None = None,
    seed: int = 7,
) -> ProcessingResult:
    output_directory.mkdir(parents=True, exist_ok=True)
    source = np.asarray(Image.open(image_path).convert("RGB"))
    cropped = _center_crop_to_aspect(source, spec.canvas.aspect_ratio)
    working = _resize_working(cropped, spec.canvas.working_width_px, cv2.INTER_AREA)
    manual_protected = _prepare_mask(
        protected_mask_path,
        source.shape[:2],
        spec.canvas.aspect_ratio,
        working.shape[:2],
    )
    auto_protected = _auto_protect_highlights(working)
    protected = manual_protected | auto_protected

    rng = np.random.default_rng(seed)
    samples, weights = _weighted_samples(working, protected, 60_000, rng)
    palette = _weighted_kmeans(
        samples,
        weights,
        spec.target_colors,
        spec.forced_palette_colors,
        seed=seed,
    )
    labels = _assign_palette(working, palette)
    if spec.forced_palette_colors:
        labels[auto_protected] = 0

    final_scale = spec.canvas.output_width_px / working.shape[1]
    number_height_final = round(spec.minimum_number_height_mm / 25.4 * spec.canvas.dpi)
    number_height_working = max(4, round(number_height_final / final_scale))
    minimum_area = max(24, round(number_height_working * number_height_working * 1.8))
    minimum_radius = spec.minimum_region_width_mm / 2 / 25.4 * spec.canvas.dpi / final_scale
    labels = _smooth_labels(labels, protected, sigma=1.5)
    labels = _merge_small_components(labels, palette, minimum_area, protected, minimum_radius)

    counts = np.bincount(labels.reshape(-1), minlength=len(palette))
    output_size = (spec.canvas.output_width_px, spec.canvas.output_height_px)
    labels_final = _smooth_labels(labels, protected, sigma=1.0, output_size=output_size)
    preview_final = palette[labels_final]
    line_width = max(2, round(0.25 / 25.4 * spec.canvas.dpi))
    blank_final = _boundary_image(labels_final, line_width)
    numbered, report = _draw_labels(
        blank_final,
        labels,
        palette,
        output_size[0] / labels.shape[1],
        output_size[1] / labels.shape[0],
        number_height_final,
        protected,
    )

    simplified_path = output_directory / "simplified-E.png"
    completed_path = output_directory / "completed-preview.png"
    blank_path = output_directory / "blank-line-art.png"
    numbered_path = output_directory / "numbered-line-art.png"
    palette_path = output_directory / "palette.png"
    palette_json_path = output_directory / "palette.json"
    report_path = output_directory / "region-report.json"

    Image.open(image_path).convert("RGB").save(simplified_path)
    Image.fromarray(preview_final).save(completed_path, dpi=(spec.canvas.dpi, spec.canvas.dpi))
    Image.fromarray(blank_final).save(blank_path, dpi=(spec.canvas.dpi, spec.canvas.dpi))
    Image.fromarray(cv2.cvtColor(numbered, cv2.COLOR_BGR2RGB)).save(
        numbered_path, dpi=(spec.canvas.dpi, spec.canvas.dpi)
    )
    cv2.imwrite(str(palette_path), _palette_art(palette, counts))

    entries = [
        PaletteEntry(
            number=index + 1,
            rgb=tuple(int(value) for value in color),
            hex=f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}",
            pixel_count=int(counts[index]),
        ).model_dump(mode="json")
        for index, color in enumerate(palette)
    ]
    palette_json_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return ProcessingResult(
        simplified_image=simplified_path,
        completed_preview=completed_path,
        blank_line_art=blank_path,
        numbered_line_art=numbered_path,
        palette_image=palette_path,
        palette_json=palette_json_path,
        region_report_json=report_path,
    )

