from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path

from agents import Agent, Runner

from .config import Settings
from .models import WorkflowPlan, WorkflowSpec
from .prompts import build_default_plan


ORCHESTRATOR_INSTRUCTIONS = """
You are the single Paint-by-Numbers Orchestrator. Produce a fixed five-stage A-E image workflow plan.
You do not reorder, skip, or add stages. You protect semantic identity before aesthetic simplification.

The stage contract is immutable:
- A: source image -> flat, segmentation-friendly run-specific subjects on solid #FF00FF.
- B: source image -> flat, segmentation-friendly background with those subjects removed.
- C: deterministic code removes A's chroma background and pixel-composites A over B, then requests approval.
- D: approved C + source -> simplified oil-paint rendering.
- E: D + source -> cleaned full-color image only, then the application requests human approval.
- Only after E approval, deterministic code performs 36-color quantization, line art, numbering, and palette output.

Never ask an image stage to perform another stage's work. In particular, Stage E must not use exactly 36 colors
and must not create line art, numbers, a palette, or a proof sheet. It may only make the full-color image easier
for deterministic 36-color conversion later. Human approval is handled by the application, not by image output.

Stages A and B are structural simplification assets, not finished paintings. Do not apply oil-paint styling in
A or B. They must use large flat closed color regions with smooth boundaries and no visible brushwork, fine surface
texture, repeated narrow marks, small pattern fragments, gradients, speckles, or micro-texture. Preserve only the
small details identified as recognition-critical for the current run; otherwise avoid regions or narrow marks
smaller than about 1% of the image width. Oil-paint character begins only in Stage D and must still preserve the
large-region structure.

Stage A's background must be perfectly uniform solid #FF00FF magenta, with no shadow, gradient, texture, floor
plane, or magenta inside the subjects. Stage C is code-owned pixel compositing; the Stage C prompt documents
validation criteria but must not request a generative redraw.

The final product is for an adult beginner who cannot draw. Every run-specific subject must remain recognizable,
color-faithful, and compositionally faithful. Preserve anatomy only when the subject is living, and preserve small
features only when the source inspection or workflow specification identifies them as recognition-critical. Stage
prompts must explicitly distinguish what changes from what remains invariant. Stage C and stage E require human
approval outside the agent.

The workflow specification is run data, not a permanent ontology. Never assume any subject category, anatomy,
material, accessory, environment, or visual feature unless it is present in the current source image or named in
the current workflow specification. Generate source-specific protected features and pass conditions for this run
only.

Return structured output only. Include exactly one prompt for each of A, B, C, D, and E.
""".strip()


def build_orchestrator(settings: Settings) -> Agent:
    return Agent(
        name="Paint-by-Numbers Orchestrator",
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        model=settings.text_model,
        output_type=WorkflowPlan,
    )


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


async def create_workflow_plan(
    source_path: Path,
    spec: WorkflowSpec,
    settings: Settings,
) -> WorkflowPlan:
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    agent = build_orchestrator(settings)
    fallback = build_default_plan(spec)
    user_text = f"""
Inspect the source image and refine the supplied baseline workflow without weakening any invariant.

Workflow specification:
{json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2)}

Baseline prompts and protected-feature policy:
{json.dumps(fallback.model_dump(mode="json"), ensure_ascii=False, indent=2)}

Make every prompt specific to what is visibly present in this source image. Keep exactly five stages A-E.
""".strip()
    result = await Runner.run(
        agent,
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text},
                    {
                        "type": "input_image",
                        "image_url": _image_data_url(source_path),
                        "detail": "original",
                    },
                ],
            }
        ],
    )
    plan = result.final_output
    if not isinstance(plan, WorkflowPlan):
        raise TypeError("Agent did not return a WorkflowPlan")
    if [stage.stage for stage in plan.stages] != ["A", "B", "C", "D", "E"]:
        raise ValueError("Agent plan must contain stages A-E exactly once and in order")

    # Dependencies and post-approval processing are code-owned, not model-owned.
    # Reassert these boundaries in the prompts so a visually specific plan cannot
    # silently move quantization or approval work into an image generation stage.
    for stage in plan.stages:
        baseline_stage = fallback.stage(stage.stage)
        stage.required_inputs = baseline_stage.required_inputs
        if stage.stage in ("A", "B"):
            stage.prompt += (
                "\n\nWorkflow boundary: this is a flat colored segmentation planning asset, not a finished "
                "painting. Use large smooth closed regions with nearly uniform fill. Do not show oil brushwork, "
                "fine surface texture, repeated narrow marks, small pattern fragments, gradients, speckles, "
                "or micro-texture. Except for protected semantic details, avoid regions or narrow "
                "marks smaller than about 1% of the image width. Oil-paint character begins only in Stage D."
            )
            if stage.stage == "A":
                stage.prompt += (
                    "Use a perfectly uniform solid #FF00FF chroma-key background with no shadows, gradients, "
                    "texture, floor plane, or magenta anywhere inside the subjects."
                )
        elif stage.stage == "C":
            stage.prompt += (
                "\n\nWorkflow boundary: output only the composite image. "
                "The application requests human approval after this image is generated."
            )
        elif stage.stage == "E":
            stage.prompt += (
                "\n\nWorkflow boundary: output one cleaned full-color image only. "
                "Do not quantize to exactly 36 colors and do not create line art, numbers, labels, "
                "a palette, or a proof sheet. Deterministic code performs those operations only "
                "after the application records human approval of this image."
            )
    return plan
