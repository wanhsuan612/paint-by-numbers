from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .models import ComplexityMetrics


def analyze_complexity(path: Path) -> ComplexityMetrics:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    height, width = rgb.shape[:2]
    preview = cv2.resize(rgb, (min(width, 768), round(height * min(width, 768) / width)))
    gray = cv2.cvtColor(preview, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 70, 150)
    edge_density = float(np.count_nonzero(edges) / edges.size)

    quantized = Image.fromarray(preview).quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    labels = np.asarray(quantized)
    coarse_color_count = int(np.unique(labels).size)
    tiny_pixels = 0
    total_pixels = labels.size
    for value in np.unique(labels):
        count, _, stats, _ = cv2.connectedComponentsWithStats((labels == value).astype(np.uint8), 8)
        if count <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        tiny_pixels += int(areas[areas < 12].sum())
    return ComplexityMetrics(
        width=width,
        height=height,
        coarse_color_count=coarse_color_count,
        edge_density=round(edge_density, 6),
        tiny_component_ratio=round(tiny_pixels / total_pixels, 6),
    )

