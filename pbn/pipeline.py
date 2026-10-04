"""Classic paint-by-numbers pipeline: smooth -> quantize -> merge small regions -> outline and number."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage


@dataclass
class Params:
    work_long_side: int = 1200  # pixels on the long side used for processing
    n_colors: int = 24
    smooth_spatial: int = 12  # mean-shift spatial radius
    smooth_color: int = 24  # mean-shift color radius
    min_region_frac: float = 0.0004  # smallest region, as a fraction of image area
    smooth_borders: int = 7  # majority-filter window for ragged color borders; 0 disables
    protect_delta_e: float = 35.0  # small regions this different from their surroundings are kept...
    protect_min_frac: float = 0.00003  # ...unless smaller than this fraction of image area
    seed: int = 0


@dataclass
class Result:
    image: np.ndarray  # resized source, RGB uint8
    smoothed: np.ndarray  # RGB uint8
    palette: np.ndarray  # (k, 3) RGB uint8
    labels: np.ndarray  # (h, w) palette index per pixel
    regions: np.ndarray  # (h, w) region id per pixel
    stats: dict = field(default_factory=dict)


def resize_long_side(rgb: np.ndarray, long_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1:
        return rgb.copy()
    return cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def smooth(rgb: np.ndarray, p: Params) -> np.ndarray:
    """Edge-preserving flattening: mean shift removes texture, bilateral cleans what is left."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.pyrMeanShiftFiltering(bgr, p.smooth_spatial, p.smooth_color, maxLevel=2)
    bgr = cv2.bilateralFilter(bgr, 9, 40, 9)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def quantize(rgb: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray]:
    """K-means in Lab so color distances match perception. Returns (palette RGB, labels)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(p.seed)
    sample = lab[rng.choice(len(lab), size=min(len(lab), 60_000), replace=False)]
    cv2.setRNGSeed(p.seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, _, centers = cv2.kmeans(sample, p.n_colors, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    dists = (centers**2).sum(axis=1)[None, :] - 2 * lab @ centers.T
    labels = dists.argmin(axis=1).reshape(rgb.shape[:2])
    palette = cv2.cvtColor(centers.astype(np.uint8)[None], cv2.COLOR_LAB2RGB)[0]
    return palette, labels.astype(np.int32)


def palette_lab(palette: np.ndarray) -> np.ndarray:
    """Palette in true CIELAB (L 0-100), so differences read as Delta E."""
    return cv2.cvtColor(palette[None].astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[0]


def majority_filter(labels: np.ndarray, n_colors: int, size: int) -> np.ndarray:
    """Replace each pixel with the most common color in its window, straightening ragged borders."""
    votes = np.stack(
        [cv2.boxFilter((labels == c).astype(np.float32), -1, (size, size)) for c in range(n_colors)]
    )
    return votes.argmax(axis=0).astype(np.int32)


def label_regions(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Split each palette color into 4-connected regions with unique ids starting at 1."""
    regions = np.zeros(labels.shape, dtype=np.int32)
    next_id = 0
    for color in np.unique(labels):
        comp, n = ndimage.label(labels == color)
        mask = comp > 0
        regions[mask] = comp[mask] + next_id
        next_id += n
    return regions, next_id


def merge_small_regions(
    labels: np.ndarray,
    palette: np.ndarray,
    min_area: int,
    protect_area: int = 0,
    protect_delta_e: float = float("inf"),
) -> np.ndarray:
    """Repaint every region smaller than min_area with the neighbor color it shares the longest border with.

    Regions of at least protect_area pixels that differ from that neighbor by protect_delta_e or more
    (an eye catchlight, a pink nose) are kept: they are small but carry the likeness.
    """
    labels = labels.copy()
    lab_palette = palette_lab(palette)
    protected: set[tuple[int, int]] = set()
    cross = ndimage.generate_binary_structure(2, 1)
    for _ in range(10):
        regions, _ = label_regions(labels)
        areas = np.bincount(regions.ravel())
        small = np.flatnonzero(areas < min_area)
        small = small[small > 0]
        first_pixel = ndimage.minimum_position(np.arange(regions.size).reshape(regions.shape), regions, small)
        small = np.array([rid for rid, pos in zip(small, first_pixel) if pos not in protected], dtype=int)
        if len(small) == 0:
            break
        slices = ndimage.find_objects(regions)
        # Smallest first, so specks are absorbed before their neighbors are judged.
        for rid in small[np.argsort(areas[small])]:
            sl = slices[rid - 1]
            y0, y1 = max(sl[0].start - 1, 0), min(sl[0].stop + 1, labels.shape[0])
            x0, x1 = max(sl[1].start - 1, 0), min(sl[1].stop + 1, labels.shape[1])
            mask = regions[y0:y1, x0:x1] == rid
            ring = ndimage.binary_dilation(mask, cross) & ~mask
            own = labels[y0:y1, x0:x1][mask][0]
            neighbors = labels[y0:y1, x0:x1][ring]
            neighbors = neighbors[neighbors != own]
            if len(neighbors) == 0:
                continue
            counts = np.bincount(neighbors, minlength=len(palette))
            best = np.flatnonzero(counts == counts.max())
            if len(best) > 1:
                d = ((lab_palette[best] - lab_palette[own]) ** 2).sum(axis=1)
                best = best[[d.argmin()]]
            contrast = float(np.linalg.norm(lab_palette[best[0]] - lab_palette[own]))
            if areas[rid] >= protect_area and contrast >= protect_delta_e:
                ys, xs = np.nonzero(mask)
                protected.add((int(ys[0] + y0), int(xs[0] + x0)))
                continue
            labels[y0:y1, x0:x1][mask] = best[0]
    return labels


def compact_palette(labels: np.ndarray, palette: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop palette entries no region uses any more, and order the rest light to dark."""
    used = np.unique(labels)
    lightness = cv2.cvtColor(palette[used][None], cv2.COLOR_RGB2LAB)[0][:, 0]
    used = used[np.argsort(-lightness.astype(int))]
    remap = np.full(len(palette), -1, dtype=np.int32)
    remap[used] = np.arange(len(used))
    return palette[used], remap[labels]


def run(rgb: np.ndarray, p: Params | None = None) -> Result:
    p = p or Params()
    image = resize_long_side(rgb, p.work_long_side)
    smoothed = smooth(image, p)
    palette, labels = quantize(smoothed, p)
    _, raw_count = label_regions(labels)
    min_area = max(4, round(p.min_region_frac * labels.size))
    if p.smooth_borders:
        labels = majority_filter(labels, len(palette), p.smooth_borders)
    protect_area = max(4, round(p.protect_min_frac * labels.size))
    labels = merge_small_regions(labels, palette, min_area, protect_area, p.protect_delta_e)
    palette, labels = compact_palette(labels, palette)
    regions, count = label_regions(labels)
    stats = {
        "size": [int(image.shape[1]), int(image.shape[0])],
        "colors": int(len(palette)),
        "regions_before_merge": int(raw_count),
        "regions": int(count),
        "min_region_px": int(min_area),
    }
    return Result(image, smoothed, palette, labels, regions, stats)
