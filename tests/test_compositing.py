from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pbn_agent.compositing import composite_stage_c


def test_deterministic_composite_preserves_opaque_subject_pixels(tmp_path: Path) -> None:
    subject = np.full((80, 100, 3), (255, 0, 255), dtype=np.uint8)
    subject[20:60, 25:75] = (180, 110, 45)
    subject[30:50, 40:60] = (250, 248, 240)
    background = np.full((80, 100, 3), (30, 60, 90), dtype=np.uint8)
    subject_path = tmp_path / "subject.png"
    background_path = tmp_path / "background.png"
    output_path = tmp_path / "C.png"
    Image.fromarray(subject).save(subject_path)
    Image.fromarray(background).save(background_path)

    result = composite_stage_c(subject_path, background_path, output_path)
    composite = np.asarray(Image.open(result.image_path).convert("RGB"))

    assert tuple(composite[40, 50]) == (250, 248, 240)
    assert tuple(composite[10, 10]) == (30, 60, 90)
    assert '"pixel_identity_passed": true' in result.report_path.read_text(encoding="utf-8")


def test_deterministic_composite_rejects_non_chroma_stage_a(tmp_path: Path) -> None:
    subject_path = tmp_path / "subject.png"
    background_path = tmp_path / "background.png"
    Image.new("RGB", (100, 80), (248, 245, 235)).save(subject_path)
    Image.new("RGB", (100, 80), (30, 60, 90)).save(background_path)

    with pytest.raises(ValueError, match="magenta chroma"):
        composite_stage_c(subject_path, background_path, tmp_path / "C.png")
