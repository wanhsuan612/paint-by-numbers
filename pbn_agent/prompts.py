from __future__ import annotations

from .models import ProtectedFeature, StagePrompt, WorkflowPlan, WorkflowSpec


def _describe(items: list[str], fallback: str) -> str:
    cleaned = [item.strip() for item in items if item.strip()]
    return "; ".join(cleaned) if cleaned else fallback


def build_default_plan(spec: WorkflowSpec) -> WorkflowPlan:
    subjects = _describe(
        spec.subject,
        "the primary subject or subjects identified from the source image",
    )
    background = _describe(
        spec.background,
        "the visible environment behind and around the primary subjects",
    )
    preservation_rules = _describe(
        spec.must_preserve,
        "subject count, identity, silhouette, pose or orientation, action, placement, dominant colors, and recognition-critical features",
    )
    allowed_changes = _describe(
        spec.allowed_changes,
        "simplify fine texture, repeated patterns, and small nonessential color variations",
    )
    flat_style = (
        "a colored segmentation planning image for a beginner paint-by-numbers kit: large, continuous, closed, "
        "nearly uniform color regions with smooth boundaries and clear silhouettes; no visible brush texture, "
        "gradients, speckles, or fragmented patches"
    )
    run_context = f"""
Run-specific subject scope: {subjects}.
Run-specific background scope: {background}.
Run-specific preservation rules: {preservation_rules}.
Allowed simplifications: {allowed_changes}.
""".strip()

    stages = [
        StagePrompt(
            stage="A",
            purpose="Isolate the run-specific subjects and simplify them into a flat paint-by-numbers subject layer.",
            required_inputs=["source image"],
            pass_conditions=[
                "Every subject named in the run specification is present exactly once, with no added subject.",
                "Subject count, identity, silhouette, pose or orientation, action, relative arrangement, and dominant colors match the source and run-specific preservation rules.",
                "Nonessential texture and repeated detail are replaced by large, smooth, closed, nearly uniform color regions.",
                "The background is one uniform solid #FF00FF field, clearly separated from every subject.",
            ],
            prompt=f"""
GOAL
Create one flat subject layer for a beginner paint-by-numbers workflow. This is a structural color plan, not a
finished illustration or oil painting.

CONTENT
Keep only these run-specific subjects: {subjects}. Preserve their count, identity, silhouette, pose or orientation,
action, relative arrangement, dominant colors, and only the recognition-critical features named here:
{preservation_rules}.

SIMPLIFICATION
Apply these allowed changes: {allowed_changes}. Replace nonessential texture and repeated detail with a small
number of large, smooth, closed, nearly uniform color regions. Keep a small detail only when the run-specific
preservation rules make it essential for recognition. Do not add decorative detail, visible brushwork, gradients,
speckles, thin repeated marks, or fragmented color patches.

BACKGROUND
Replace everything outside the subjects with one perfectly uniform solid #FF00FF background. Do not place
#FF00FF inside a subject. Do not add scenery, cast shadows, text, borders, line art, numbers, or a palette.

{run_context}
""".strip(),
        ),
        StagePrompt(
            stage="B",
            purpose="Remove the run-specific subjects and simplify the remaining background.",
            required_inputs=["source image"],
            pass_conditions=[
                "Every subject named in the run specification and its subject-specific remnants are absent.",
                "Background layout, perspective, major structures, dominant colors, and broad lighting remain recognizable.",
                "Fine texture and repeated patterns are simplified into broad closed regions without inventing new background objects.",
            ],
            prompt=f"""
GOAL
Create one empty-background color plan in this style: {flat_style}.

CHANGE ONLY
Remove these run-specific subjects and their subject-specific remnants: {subjects}. Fill the revealed areas
plausibly from surrounding evidence. Simplify only what the run specification allows: {allowed_changes}.

PRESERVE
Keep the layout, perspective, major structures, dominant colors, and broad lighting of this background:
{background}. Do not invent new objects or restore removed subjects. Output one background image only.

{run_context}
""".strip(),
        ),
        StagePrompt(
            stage="C",
            purpose="Combine the simplified subjects and background while preserving the source composition.",
            required_inputs=["source image", "stage A", "stage B"],
            pass_conditions=[
                "All run-specific subjects and background structures are present without additions or omissions.",
                "Subject placement, scale, orientation, action, overlap, and floor or surface contact match the source.",
                "Run-specific preservation rules remain satisfied and broad lighting is coherent.",
            ],
            prompt=f"""
IMAGE ROLES
Image 1 is the source composition and geometry reference. Image 2 is the simplified subject layer. Image 3 is the
simplified background layer.

GOAL
Combine Image 2 and Image 3 at the placement, scale, orientation, overlaps, and surface contacts shown in Image 1.
Preserve the flat, large-region complexity of both simplified inputs. Harmonize only broad contact shadows, broad
lighting, edge treatment, and color temperature. Do not redraw identities, restore source texture, or add objects.

PRESERVE
{preservation_rules}.

{run_context}
""".strip(),
        ),
        StagePrompt(
            stage="D",
            purpose="Apply a restrained simple-oil finish without changing the approved composition or region structure.",
            required_inputs=["stage C", "source image"],
            pass_conditions=[
                "The approved composition and all run-specific preservation rules remain unchanged.",
                "Oil character is broad and restrained, without fine bristle texture or new fragmented regions.",
                "Dominant subject and background colors remain faithful to the source.",
            ],
            prompt=f"""
IMAGE ROLES
Image 1 is the approved composite and immutable composition. Image 2 is used only to verify source identity and
dominant colors.

CHANGE ONLY
Give Image 1 a restrained simple-oil finish using broad, smooth marks contained inside its existing large closed
regions. Do not introduce new small boundaries, fine texture, fragmented dabs, gradients, speckles, or decorative
detail.

PRESERVE
Keep composition, geometry, subject count, identity, pose or orientation, action, placement, dominant colors,
recognition-critical details, background structures, and broad lighting unchanged. In particular: {preservation_rules}.

Target final style: {spec.style}.
""".strip(),
        ),
        StagePrompt(
            stage="E",
            purpose="Clean the oil rendering into a full-color master for later deterministic color reduction.",
            required_inputs=["stage D", "source image"],
            pass_conditions=[
                "Fragmented marks and tiny nonessential patches are merged into broad appropriate regions.",
                "Every run-specific preservation rule and recognition-critical small feature remains intact.",
                "The output remains full color and contains no line art, numbers, labels, or palette.",
            ],
            prompt=f"""
IMAGE ROLES
Image 1 is the painting to clean. Image 2 is used only to verify source identity and dominant colors.

CHANGE ONLY
Merge remaining isolated speckles, fragmented marks, micro-shadows, and tiny nonessential color patches into the
nearest semantically and chromatically appropriate broad region. Smooth jagged boundaries while keeping crisp,
closed silhouettes and a restrained simple-oil appearance suitable for {spec.difficulty}.

PRESERVE
Do not merge or recolor any small feature named by the run-specific preservation rules. Preserve composition,
subject identity, geometry, dominant colors, background structures, and broad lighting. In particular:
{preservation_rules}.

Output one cleaned full-color image only. Do not reduce it to exactly {spec.target_colors} colors and do not create
line art, outlines, numbers, labels, swatches, a palette, or a proof sheet. Those deterministic operations happen
only after human approval.

{run_context}
""".strip(),
        ),
    ]

    protected = [
        ProtectedFeature(
            name="run-specific preserved features",
            preservation_rule=f"Preserve every item in the run specification: {preservation_rules}.",
            failure_if="Any named subject, identity cue, action, placement, dominant color, or recognition-critical feature is lost, added, moved, or materially changed.",
        ),
        ProtectedFeature(
            name="small recognition-critical color regions",
            preservation_rule="Keep a small region when the run-specific preservation rules identify it as semantically important, even if its image-wide pixel share is low.",
            failure_if="A protected small region disappears or is merged only because it occupies few pixels in the full image.",
        ),
    ]
    return WorkflowPlan(style_summary=spec.style, protected_features=protected, stages=stages)
