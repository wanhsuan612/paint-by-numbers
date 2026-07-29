from pathlib import Path

import numpy as np
from PIL import Image

from pbn_agent.models import CanvasSpec, WorkflowSpec
from pbn_agent.processing import create_paint_by_numbers


def test_rare_white_highlight_is_preserved(tmp_path: Path) -> None:
    image = np.full((100, 125, 3), (142, 94, 58), dtype=np.uint8)
    image[35:65, 45:80] = (76, 48, 30)
    image[44:58, 54:70] = (30, 24, 20)
    image[48:51, 59:62] = (252, 250, 244)
    source = tmp_path / "eye.png"
    Image.fromarray(image).save(source)
    spec = WorkflowSpec(
        target_colors=4,
        canvas=CanvasSpec(width_cm=12.5, height_cm=10, dpi=80, working_width_px=250),
        minimum_number_height_mm=1.5,
        forced_palette_colors=[(248, 245, 235), (28, 23, 18)],
    )
    result = create_paint_by_numbers(source, tmp_path / "final", spec)
    preview = np.asarray(Image.open(result.completed_preview).convert("RGB"))
    light_pixels = np.all(preview >= np.array([240, 238, 228]), axis=2)
    assert int(light_pixels.sum()) > 0
    assert result.blank_line_art.is_file()
    assert result.numbered_line_art.is_file()
    assert len(__import__("json").loads(result.palette_json.read_text())) == 4

