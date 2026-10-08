"""Turn a pipeline Result into printable images."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from pbn.pipeline import Result

LINE = (90, 90, 90)
TEXT = (60, 60, 60)


def preview(r: Result) -> np.ndarray:
    return r.palette[r.labels]


def boundaries(regions: np.ndarray) -> np.ndarray:
    """One-pixel boundary mask wherever a pixel differs from its right or lower neighbor."""
    edge = np.zeros(regions.shape, dtype=bool)
    edge[:, :-1] |= regions[:, :-1] != regions[:, 1:]
    edge[:-1, :] |= regions[:-1, :] != regions[1:, :]
    return edge


def label_positions(r: Result) -> list[tuple[int, int, int, float]]:
    """For each region: (x, y, palette index, inner radius) at the point farthest from its border."""
    out = []
    slices = ndimage.find_objects(r.regions)
    for rid, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        mask = np.pad(r.regions[sl] == rid, 1)
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        y, x = np.unravel_index(dist.argmax(), dist.shape)
        color = int(r.labels[sl][mask[1:-1, 1:-1]][0])
        out.append((x - 1 + sl[1].start, y - 1 + sl[0].start, color, float(dist.max())))
    return out


def numbered(r: Result, with_color: bool = False) -> np.ndarray:
    base = preview(r).copy() if with_color else np.full((*r.labels.shape, 3), 255, np.uint8)
    base[boundaries(r.regions)] = LINE
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x, y, color, radius in label_positions(r):
        text = str(color + 1)
        # Fit the number inside the region's inscribed circle, within a readable range.
        scale = float(np.clip(radius / (9 * len(text)), 0.25, 0.6))
        (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
        cv2.putText(base, text, (int(x - tw / 2), int(y + th / 2)), font, scale, TEXT, 1, cv2.LINE_AA)
    return base


def palette_sheet(r: Result, swatch: int = 60, cols: int = 6) -> np.ndarray:
    rows = -(-len(r.palette) // cols)
    sheet = np.full((rows * (swatch + 24) + 8, cols * (swatch + 12) + 8, 3), 255, np.uint8)
    for i, rgb in enumerate(r.palette):
        y, x = 8 + (i // cols) * (swatch + 24), 8 + (i % cols) * (swatch + 12)
        sheet[y : y + swatch, x : x + swatch] = rgb
        cv2.rectangle(sheet, (x, y), (x + swatch - 1, y + swatch - 1), LINE, 1)
        cv2.putText(sheet, str(i + 1), (x + 2, y + swatch + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1, cv2.LINE_AA)
    return sheet
