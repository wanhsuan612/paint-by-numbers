from pathlib import Path

import pytest
from PIL import Image

from pbn_agent.models import StageStatus, WorkflowSpec
from pbn_agent.workflow import _invalidate_dependents, initialize_run, load_state, plan_run, save_state, stage_request


def test_run_initialization_and_offline_plan(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (120, 90), (180, 140, 100)).save(source)
    run = initialize_run(source, tmp_path / "output", WorkflowSpec(), "test-run")
    plan_path = plan_run(run, settings=None, offline=True)  # type: ignore[arg-type]
    assert plan_path.is_file()
    assert load_state(run).plan_path == "plan.json"
    references, prompt, output, attempt = stage_request(run, "A")
    assert references == [run / "source.png"]
    assert "primary subject or subjects identified from the source image" in prompt
    assert "three cats" not in prompt
    assert output.name == "attempt-01.png"
    assert attempt == 1


def test_cannot_generate_c_before_a_and_b(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (120, 90), (180, 140, 100)).save(source)
    run = initialize_run(source, tmp_path / "output", WorkflowSpec(), "test-run")
    plan_run(run, settings=None, offline=True)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="Stage A"):
        stage_request(run, "C")


def test_upstream_revision_invalidates_only_dependent_stages(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (120, 90), (180, 140, 100)).save(source)
    run = initialize_run(source, tmp_path / "output", WorkflowSpec(), "test-run")
    state = load_state(run)
    state.stages["A"].status = StageStatus.COMPLETED
    state.stages["B"].status = StageStatus.COMPLETED
    state.stages["B"].current_artifact = "stages/B/attempt-01.png"
    state.stages["C"].status = StageStatus.AWAITING_APPROVAL
    state.stages["C"].current_artifact = "stages/C/attempt-01.png"
    save_state(run, state)

    _invalidate_dependents(state, "A")

    assert state.stages["B"].status == StageStatus.COMPLETED
    assert state.stages["B"].current_artifact is not None
    assert state.stages["C"].status == StageStatus.PENDING
    assert state.stages["C"].current_artifact is None


def test_a_revision_uses_current_output_as_edit_target(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (120, 90), (180, 140, 100)).save(source)
    run = initialize_run(source, tmp_path / "output", WorkflowSpec(), "test-run")
    plan_run(run, settings=None, offline=True)  # type: ignore[arg-type]
    current = run / "stages" / "A" / "attempt-01.png"
    Image.new("RGB", (120, 90), (255, 0, 255)).save(current)
    state = load_state(run)
    state.stages["A"].attempts = 1
    state.stages["A"].status = StageStatus.COMPLETED
    state.stages["A"].current_artifact = "stages/A/attempt-01.png"
    save_state(run, state)

    references, prompt, _, _ = stage_request(run, "A", "change only the background")

    assert references == [current, run / "source.png"]
    assert "Image 1 is the current stage output and the edit target" in prompt
