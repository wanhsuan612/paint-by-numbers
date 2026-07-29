from __future__ import annotations

import asyncio
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from .agent import create_workflow_plan
from .compositing import align_stage_a_to_source, composite_stage_c
from .config import Settings
from .image_client import ImageGenerationClient
from .metrics import analyze_complexity
from .models import RunState, StageName, StageStatus, WorkflowPlan, WorkflowSpec
from .processing import ProcessingResult, create_paint_by_numbers
from .prompts import build_default_plan


STAGE_ORDER: tuple[StageName, ...] = ("A", "B", "C", "D", "E")
DEPENDENTS: dict[StageName, tuple[StageName, ...]] = {
    "A": ("C", "D", "E"),
    "B": ("C", "D", "E"),
    "C": ("D", "E"),
    "D": ("E",),
    "E": (),
}


def _invalidate_dependents(state: RunState, stage: StageName) -> None:
    for dependent in DEPENDENTS[stage]:
        downstream = state.stages[dependent]
        downstream.status = StageStatus.PENDING
        downstream.current_artifact = None
        downstream.metrics_path = None
    state.final_outputs = {}


def _write_json(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")  # type: ignore[union-attr]
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def initialize_run(
    source_path: Path,
    output_root: Path,
    spec: WorkflowSpec | None = None,
    run_id: str | None = None,
) -> Path:
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    spec = spec or WorkflowSpec()
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_id = run_id or f"{source_path.stem}-{timestamp}"
    run_directory = (output_root / run_id).resolve()
    if run_directory.exists():
        raise FileExistsError(f"Run already exists: {run_directory}")
    run_directory.mkdir(parents=True)
    for stage in STAGE_ORDER:
        (run_directory / "stages" / stage).mkdir(parents=True)
    source_copy = run_directory / f"source{source_path.suffix.lower()}"
    shutil.copy2(source_path, source_copy)
    _write_json(run_directory / "spec.json", spec)
    state = RunState(
        run_id=run_id,
        source_path=str(source_copy),
    )
    _write_json(run_directory / "state.json", state)
    return run_directory


def load_state(run_directory: Path) -> RunState:
    return RunState.model_validate_json((run_directory / "state.json").read_text(encoding="utf-8"))


def load_spec(run_directory: Path) -> WorkflowSpec:
    return WorkflowSpec.model_validate_json((run_directory / "spec.json").read_text(encoding="utf-8"))


def save_state(run_directory: Path, state: RunState) -> None:
    _write_json(run_directory / "state.json", state)


def save_plan(run_directory: Path, plan: WorkflowPlan) -> Path:
    path = run_directory / "plan.json"
    _write_json(path, plan)
    state = load_state(run_directory)
    state.plan_path = "plan.json"
    save_state(run_directory, state)
    return path


def load_plan(run_directory: Path) -> WorkflowPlan:
    state = load_state(run_directory)
    if state.plan_path is None:
        raise RuntimeError("Run has no plan. Run the plan command first.")
    return WorkflowPlan.model_validate_json((run_directory / state.plan_path).read_text(encoding="utf-8"))


def plan_run(run_directory: Path, settings: Settings, offline: bool = False) -> Path:
    spec = load_spec(run_directory)
    if offline:
        plan = build_default_plan(spec)
    else:
        state = load_state(run_directory)
        plan = asyncio.run(create_workflow_plan(Path(state.source_path), spec, settings))
    return save_plan(run_directory, plan)


def _artifact(run_directory: Path, state: RunState, stage: StageName) -> Path:
    relative = state.stages[stage].current_artifact
    if relative is None:
        raise RuntimeError(f"Stage {stage} has no artifact")
    return run_directory / relative


def _references(
    run_directory: Path,
    state: RunState,
    stage: StageName,
    revision: bool = False,
) -> list[Path]:
    source = Path(state.source_path)
    if stage in ("A", "B"):
        if revision and state.stages[stage].current_artifact:
            return [_artifact(run_directory, state, stage), source]
        return [source]
    if stage == "C":
        return [source, _artifact(run_directory, state, "A"), _artifact(run_directory, state, "B")]
    if stage == "D":
        return [_artifact(run_directory, state, "C"), source]
    return [_artifact(run_directory, state, "D"), source]


def _check_stage_ready(state: RunState, stage: StageName) -> None:
    if stage == "C":
        for dependency in ("A", "B"):
            if state.stages[dependency].status not in (StageStatus.COMPLETED, StageStatus.APPROVED):
                raise RuntimeError(f"Stage {dependency} must be completed before C")
    elif stage == "D" and state.stages["C"].status != StageStatus.APPROVED:
        raise RuntimeError("Stage C must be approved before D")
    elif stage == "E" and state.stages["D"].status not in (StageStatus.COMPLETED, StageStatus.APPROVED):
        raise RuntimeError("Stage D must be completed before E")


def stage_request(
    run_directory: Path,
    stage: StageName,
    feedback: str | None = None,
) -> tuple[list[Path], str, Path, int]:
    state = load_state(run_directory)
    spec = load_spec(run_directory)
    _check_stage_ready(state, stage)
    progress = state.stages[stage]
    maximum_attempts = 1 + spec.max_revisions_per_stage
    if progress.attempts >= maximum_attempts:
        raise RuntimeError(
            f"Stage {stage} reached its limit of {maximum_attempts} attempts; manual direction is required"
        )
    if progress.attempts > 0 and not feedback:
        raise ValueError("A revision attempt requires explicit feedback")
    plan = load_plan(run_directory)
    prompt = plan.stage(stage).prompt
    if feedback:
        input_roles = ""
        if stage in ("A", "B") and progress.current_artifact:
            input_roles = (
                "\n\nRevision input roles: Image 1 is the current stage output and the edit target. "
                "Image 2 is the original source used only to verify identity, color, geometry, and composition."
            )
        prompt += (
            input_roles
            + "\n\nRevision request:\n"
            + feedback.strip()
            + "\nApply only this correction. Reassert every invariant and do not redesign unrelated content."
        )
    attempt = progress.attempts + 1
    output_path = run_directory / "stages" / stage / f"attempt-{attempt:02d}.png"
    return _references(run_directory, state, stage, revision=bool(feedback)), prompt, output_path, attempt


def generate_stage(
    run_directory: Path,
    stage: StageName,
    settings: Settings,
    feedback: str | None = None,
) -> Path:
    references, prompt, output_path, attempt = stage_request(run_directory, stage, feedback)
    if stage == "C":
        composite_stage_c(references[1], references[2], output_path)
    else:
        client = ImageGenerationClient(settings)
        client.edit(references, prompt, output_path)
        if stage == "A":
            generated_path = output_path
            output_path = output_path.with_name(f"{output_path.stem}-aligned.png")
            align_stage_a_to_source(generated_path, Path(load_state(run_directory).source_path), output_path)
    metrics = analyze_complexity(output_path)
    metrics_path = output_path.with_suffix(".metrics.json")
    _write_json(metrics_path, metrics)

    state = load_state(run_directory)
    progress = state.stages[stage]
    progress.attempts = attempt
    progress.current_artifact = str(output_path.relative_to(run_directory))
    progress.metrics_path = str(metrics_path.relative_to(run_directory))
    if feedback:
        progress.feedback_history.append(feedback.strip())
    progress.status = (
        StageStatus.AWAITING_APPROVAL if stage in load_spec(run_directory).approval_stages else StageStatus.COMPLETED
    )
    _invalidate_dependents(state, stage)
    save_state(run_directory, state)
    shutil.copy2(output_path, run_directory / f"{stage}.png")
    return output_path


def align_current_stage_a(run_directory: Path) -> Path:
    state = load_state(run_directory)
    current = _artifact(run_directory, state, "A")
    if current.stem.endswith("-aligned"):
        return current
    output_path = current.with_name(f"{current.stem}-aligned.png")
    align_stage_a_to_source(current, Path(state.source_path), output_path)
    metrics_path = output_path.with_suffix(".metrics.json")
    _write_json(metrics_path, analyze_complexity(output_path))
    progress = state.stages["A"]
    progress.current_artifact = str(output_path.relative_to(run_directory))
    progress.metrics_path = str(metrics_path.relative_to(run_directory))
    _invalidate_dependents(state, "A")
    save_state(run_directory, state)
    shutil.copy2(output_path, run_directory / "A.png")
    return output_path


def rebuild_current_stage_c(run_directory: Path) -> Path:
    state = load_state(run_directory)
    output_path = _artifact(run_directory, state, "C")
    subject_path = _artifact(run_directory, state, "A")
    background_path = _artifact(run_directory, state, "B")
    composite_stage_c(subject_path, background_path, output_path)
    metrics_path = output_path.with_suffix(".metrics.json")
    _write_json(metrics_path, analyze_complexity(output_path))
    state.stages["C"].metrics_path = str(metrics_path.relative_to(run_directory))
    save_state(run_directory, state)
    shutil.copy2(output_path, run_directory / "C.png")
    return output_path


def approve_stage(run_directory: Path, stage: StageName) -> None:
    spec = load_spec(run_directory)
    if stage not in spec.approval_stages:
        raise ValueError(f"Stage {stage} is not an approval stage")
    state = load_state(run_directory)
    progress = state.stages[stage]
    if progress.status != StageStatus.AWAITING_APPROVAL or not progress.current_artifact:
        raise RuntimeError(f"Stage {stage} is not awaiting approval")
    progress.status = StageStatus.APPROVED
    save_state(run_directory, state)


def finalize_run(
    run_directory: Path,
    protected_mask_path: Path | None = None,
) -> ProcessingResult:
    state = load_state(run_directory)
    if state.stages["E"].status != StageStatus.APPROVED:
        raise RuntimeError("Stage E must be approved before finalization")
    result = create_paint_by_numbers(
        _artifact(run_directory, state, "E"),
        run_directory / "final",
        load_spec(run_directory),
        protected_mask_path=protected_mask_path,
    )
    state.final_outputs = {
        "simplified_E": str(result.simplified_image.relative_to(run_directory)),
        "completed_preview": str(result.completed_preview.relative_to(run_directory)),
        "blank_line_art": str(result.blank_line_art.relative_to(run_directory)),
        "numbered_line_art": str(result.numbered_line_art.relative_to(run_directory)),
        "palette_image": str(result.palette_image.relative_to(run_directory)),
        "palette_json": str(result.palette_json.relative_to(run_directory)),
        "region_report": str(result.region_report_json.relative_to(run_directory)),
    }
    save_state(run_directory, state)
    return result
