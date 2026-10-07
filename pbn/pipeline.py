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
    protect_min_frac: float = 0.00002  # ...unless smaller than this fraction of image area
    min_color_gap: float = 10.0  # palette colors closer than this Delta E are merged, so gradients band less
    rescue_colors: int = 4  # palette slots that may be handed to colors K-means missed
    rescue_delta_e: float = 9.0  # a pixel this far from every palette color is badly matched...
    rescue_min_frac: float = 0.0002  # ...and a connected patch of them this large earns its own color
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


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 to true CIELAB (L 0-100), so differences read as Delta E."""
    return cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(lab[None].astype(np.float32), cv2.COLOR_LAB2RGB)[0]
    return np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)


def assign(lab: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest center for each Lab pixel, and the Delta E to it."""
    d2 = (lab**2).sum(axis=1)[:, None] - 2 * lab @ centers.T + (centers**2).sum(axis=1)[None, :]
    labels = d2.argmin(axis=1)
    return labels.astype(np.int32), np.sqrt(np.maximum(d2[np.arange(len(lab)), labels], 0))


def merge_closest_pair(centers: np.ndarray, counts: np.ndarray, frozen: np.ndarray) -> tuple[np.ndarray, ...]:
    """Replace the two most similar unfrozen colors with their pixel-weighted mean."""
    gaps = np.linalg.norm(centers[:, None] - centers[None], axis=2)
    np.fill_diagonal(gaps, np.inf)
    gaps[frozen, :] = gaps[:, frozen] = np.inf
    i, j = np.unravel_index(gaps.argmin(), gaps.shape)
    total = counts[i] + counts[j]
    centers[i] = (centers[i] * counts[i] + centers[j] * counts[j]) / max(total, 1)
    counts[i] = total
    keep = np.arange(len(centers)) != j
    return centers[keep], counts[keep], frozen[keep], float(gaps[i, j])


def quantize(rgb: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray]:
    """K-means in Lab, then space the colors out and rescue colors it missed. Returns (palette RGB, labels).

    K-means spends its colors where the pixels are. A large smooth gradient (a white chest, a plate)
    soaks up a ramp of near-identical colors, which paints as many thin contour bands; merging colors
    closer than min_color_gap keeps the shading as fewer, clearly different steps.

    The same bias means a small but distinctive area (a pink nose, green eyes) can end up with no
    color close to it. Each rescue finds the largest connected patch that every palette color matches
    badly and gives it its own color, merging the two most similar colors if the palette is full.
    """
    h, w = rgb.shape[:2]
    lab = to_lab(rgb).reshape(-1, 3)
    rng = np.random.default_rng(p.seed)
    sample = lab[rng.choice(len(lab), size=min(len(lab), 60_000), replace=False)]
    cv2.setRNGSeed(p.seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, _, centers = cv2.kmeans(sample, p.n_colors, None, criteria, 4, cv2.KMEANS_PP_CENTERS)

    labels, _ = assign(lab, centers)
    counts = np.bincount(labels, minlength=len(centers)).astype(np.float32)
    frozen = np.zeros(len(centers), dtype=bool)
    while len(centers) > 2:
        merged, merged_counts, merged_frozen, gap = merge_closest_pair(centers.copy(), counts.copy(), frozen)
        if gap >= p.min_color_gap:
            break
        centers, counts, frozen = merged, merged_counts, merged_frozen

    rescue_area = p.rescue_min_frac * h * w
    # A rescued color must be new, not a mid-tone the spacing step just merged away.
    distinct = max(p.rescue_delta_e, p.min_color_gap)
    for _ in range(p.rescue_colors):
        labels, err = assign(lab, centers)
        patches, _ = ndimage.label((err > p.rescue_delta_e).reshape(h, w))
        areas = np.bincount(patches.ravel())
        areas[0] = 0
        color = None
        for candidate in np.argsort(-areas):
            if areas[candidate] < rescue_area:
                break
            mean = lab[patches.ravel() == candidate].mean(axis=0)
            # Skip a patch whose average is already in the palette; only its spread is off.
            if np.linalg.norm(centers - mean, axis=1).min() >= distinct:
                color = mean
                break
        if color is None:
            break
        counts = np.bincount(labels, minlength=len(centers)).astype(np.float32)
        if len(centers) >= p.n_colors:
            # Never merge away a color we just rescued.
            centers, counts, frozen, _ = merge_closest_pair(centers, counts, frozen)
        centers = np.vstack([centers, color[None]]).astype(np.float32)
        frozen = np.append(frozen, True)

    labels, _ = assign(lab, centers)
    return lab_to_rgb(centers), labels.reshape(h, w)


def palette_lab(palette: np.ndarray) -> np.ndarray:
    return to_lab(palette[None])[0]


def majority_filter(labels: np.ndarray, n_colors: int, size: int) -> np.ndarray:
    """Replace each pixel with the most common color in its window, straightening ragged borders."""
    votes = np.stack(
        [cv2.boxFilter((labels == c).astype(np.float32), -1, (size, size)) for c in range(n_colors)]
    )
    return votes.argmax(axis=0).astype(np.int32)


def restore_details(
    before: np.ndarray, after: np.ndarray, palette: np.ndarray, min_area: int, delta_e: float
) -> np.ndarray:
    """Undo the majority filter where it erased a small patch of sharply different color, like a catchlight."""
    lab_palette = palette_lab(palette)
    erased = np.linalg.norm(lab_palette[before] - lab_palette[after], axis=-1) >= delta_e
    patches, _ = ndimage.label(erased)
    areas = np.bincount(patches.ravel())
    keep = (areas >= min_area)[patches] & erased
    restored = after.copy()
    restored[keep] = before[keep]
    return restored


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
    protect_area = max(4, round(p.protect_min_frac * labels.size))
    if p.smooth_borders:
        filtered = majority_filter(labels, len(palette), p.smooth_borders)
        labels = restore_details(labels, filtered, palette, protect_area, p.protect_delta_e)
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
