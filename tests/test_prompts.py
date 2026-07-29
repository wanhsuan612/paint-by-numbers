import re

import pytest

from pbn_agent.models import StagePrompt, WorkflowPlan
from pbn_agent.models import WorkflowSpec
from pbn_agent.prompts import build_default_plan


def test_default_plan_contains_exactly_a_to_e() -> None:
    plan = build_default_plan(WorkflowSpec())
    assert [stage.stage for stage in plan.stages] == ["A", "B", "C", "D", "E"]
    assert "36" in plan.stage("E").prompt
    assert "not a\nfinished illustration or oil painting" in plan.stage("A").prompt.lower()
    assert "three cats" not in plan.model_dump_json().lower()
    assert "yarn" not in plan.model_dump_json().lower()


def test_plan_uses_run_specific_subjects_without_assuming_animals() -> None:
    spec = WorkflowSpec(
        subject=["one red bicycle"],
        background=["a brick wall", "a paved lane"],
        must_preserve=[
            "exactly one bicycle",
            "the original frame geometry and wheel count",
            "the red frame color and white front reflector",
        ],
        allowed_changes=["simplify spoke detail and brick texture"],
    )

    plan = build_default_plan(spec)
    serialized = plan.model_dump_json().lower()

    assert "one red bicycle" in plan.stage("A").prompt.lower()
    assert "frame geometry and wheel count" in plan.stage("A").prompt.lower()
    assert "brick wall" in plan.stage("B").prompt.lower()
    assert "spoke detail and brick texture" in plan.stage("A").prompt.lower()
    assert re.search(r"\bcats?\b", serialized) is None
    assert re.search(r"\byarn\b", serialized) is None
    assert re.search(r"\beyes?\b", serialized) is None


def test_plan_rejects_duplicate_or_out_of_order_stages() -> None:
    stage = StagePrompt(stage="A", purpose="x", prompt="x", required_inputs=[], pass_conditions=[])
    with pytest.raises(ValueError, match="exactly once"):
        WorkflowPlan(style_summary="x", protected_features=[], stages=[stage] * 5)
