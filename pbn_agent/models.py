from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


StageName = Literal["A", "B", "C", "D", "E"]


class StageStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    FAILED = "failed"


class CanvasSpec(BaseModel):
    width_cm: float = 50.0
    height_cm: float = 40.0
    dpi: int = 300
    working_width_px: int = 2000

    @property
    def aspect_ratio(self) -> float:
        return self.width_cm / self.height_cm

    @property
    def output_width_px(self) -> int:
        return round(self.width_cm / 2.54 * self.dpi)

    @property
    def output_height_px(self) -> int:
        return round(self.height_cm / 2.54 * self.dpi)


class WorkflowSpec(BaseModel):
    subject: list[str] = Field(default_factory=list)
    background: list[str] = Field(default_factory=list)
    must_preserve: list[str] = Field(
        default_factory=lambda: [
            "the number and identity of every primary subject",
            "each subject's silhouette, pose or orientation, action, and relative placement",
            "dominant subject colors and recognition-critical features",
        ]
    )
    allowed_changes: list[str] = Field(
        default_factory=lambda: [
            "simplify fine surface texture and repeated patterns",
            "merge small nonessential color variations into broad shapes",
        ]
    )
    target_colors: int = 36
    difficulty: str = "adult beginner with no drawing experience"
    style: str = "simple oil painting with broad clean strokes and no fragmented color patches"
    max_revisions_per_stage: int = 2
    approval_stages: list[StageName] = Field(default_factory=lambda: ["C", "E"])
    minimum_number_height_mm: float = 3.0
    minimum_region_width_mm: float = 1.5
    canvas: CanvasSpec = Field(default_factory=CanvasSpec)
    forced_palette_colors: list[tuple[int, int, int]] = Field(
        default_factory=lambda: [(248, 245, 235), (28, 23, 18)]
    )

    @model_validator(mode="after")
    def validate_constraints(self) -> "WorkflowSpec":
        if not 2 <= self.target_colors <= 64:
            raise ValueError("target_colors must be between 2 and 64")
        if self.max_revisions_per_stage < 0:
            raise ValueError("max_revisions_per_stage cannot be negative")
        return self


class StagePrompt(BaseModel):
    stage: StageName
    purpose: str
    prompt: str
    required_inputs: list[str]
    pass_conditions: list[str]


class ProtectedFeature(BaseModel):
    name: str
    preservation_rule: str
    failure_if: str


class WorkflowPlan(BaseModel):
    style_summary: str
    protected_features: list[ProtectedFeature]
    stages: list[StagePrompt]

    @model_validator(mode="after")
    def validate_stage_sequence(self) -> "WorkflowPlan":
        if [stage.stage for stage in self.stages] != ["A", "B", "C", "D", "E"]:
            raise ValueError("stages must contain A-E exactly once and in order")
        return self

    def stage(self, name: StageName) -> StagePrompt:
        for stage in self.stages:
            if stage.stage == name:
                return stage
        raise KeyError(f"Stage {name} is missing from the plan")


class ComplexityMetrics(BaseModel):
    width: int
    height: int
    coarse_color_count: int
    edge_density: float
    tiny_component_ratio: float


class StageProgress(BaseModel):
    status: StageStatus = StageStatus.PENDING
    attempts: int = 0
    current_artifact: str | None = None
    feedback_history: list[str] = Field(default_factory=list)
    metrics_path: str | None = None


class RunState(BaseModel):
    run_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    source_path: str
    spec_path: str = "spec.json"
    plan_path: str | None = None
    stages: dict[StageName, StageProgress] = Field(
        default_factory=lambda: {name: StageProgress() for name in ("A", "B", "C", "D", "E")}
    )
    final_outputs: dict[str, str] = Field(default_factory=dict)

    @property
    def directory(self) -> Path:
        return Path(self.source_path).parent


class PaletteEntry(BaseModel):
    number: int
    rgb: tuple[int, int, int]
    hex: str
    pixel_count: int


class RegionReport(BaseModel):
    region_count: int
    internal_label_count: int
    vertical_label_count: int
    callout_label_count: int
    protected_region_count: int
