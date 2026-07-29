from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FONT = ImageFont.load_default(size=18)
SMALL_FONT = ImageFont.load_default(size=14)


def rounded_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], text: str, color: str) -> None:
    draw.rounded_rectangle(xy, radius=14, fill=color, outline="#243447", width=2)
    left, top, right, bottom = xy
    box = draw.multiline_textbbox((0, 0), text, font=FONT, align="center")
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.multiline_text(((left + right - width) / 2, (top + bottom - height) / 2), text, font=FONT, fill="#14202b", align="center")


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line([start, end], fill="#516273", width=4)
    draw.polygon([(end[0], end[1]), (end[0] - 10, end[1] - 7), (end[0] - 10, end[1] + 7)], fill="#516273")


def interactions() -> None:
    image = Image.new("RGB", (1500, 620), "#f6f3ea")
    draw = ImageDraw.Draw(image)
    draw.text((48, 30), "Paint-by-Numbers MVP - single-agent interactions", font=FONT, fill="#14202b")
    rounded_box(draw, (50, 190, 280, 350), "User\nCLI approvals", "#f8d7a4")
    rounded_box(draw, (420, 150, 730, 390), "Single Orchestrator\nplan + prompts +\nrevision decisions", "#b8d8eb")
    rounded_box(draw, (870, 80, 1180, 240), "GPT Image\nA-E image edits", "#cde7cc")
    rounded_box(draw, (870, 350, 1180, 540), "Deterministic tools\n36 colors\nprotected details\nline art + labels", "#ded0ef")
    rounded_box(draw, (1260, 210, 1450, 390), "Run artifacts\nstate + images\nmetrics + final", "#f4c7c3")
    arrow(draw, (280, 270), (420, 270))
    arrow(draw, (730, 220), (870, 160))
    arrow(draw, (730, 330), (870, 430))
    arrow(draw, (1180, 160), (1260, 270))
    arrow(draw, (1180, 430), (1260, 340))
    image.save(DOCS / "agent-interactions.png")


def sequence() -> None:
    image = Image.new("RGB", (1600, 800), "#f6f3ea")
    draw = ImageDraw.Draw(image)
    draw.text((48, 28), "A-E generation and approval sequence", font=FONT, fill="#14202b")
    labels = [
        ("Source", "#eee4ce"),
        ("A\nSubjects", "#cde7cc"),
        ("B\nBackground", "#cde7cc"),
        ("C\nComposite", "#b8d8eb"),
        ("Approve C", "#f8d7a4"),
        ("D\nOil style", "#cde7cc"),
        ("E\nCleanup", "#b8d8eb"),
        ("Approve E", "#f8d7a4"),
        ("36 colors\n+ outputs", "#ded0ef"),
    ]
    x_positions = [40, 220, 400, 580, 760, 940, 1120, 1300, 1450]
    for index, ((label, color), x) in enumerate(zip(labels, x_positions, strict=True)):
        width = 140 if index < 8 else 120
        rounded_box(draw, (x, 270, x + width, 430), label, color)
        if index < len(labels) - 1:
            next_x = x_positions[index + 1]
            arrow(draw, (x + width, 350), (next_x, 350))
    draw.text((580, 500), "C and E are hard gates; later stages cannot run before approval.", font=SMALL_FONT, fill="#516273")
    draw.text((220, 550), "Each stage: initial attempt + at most two feedback-driven revisions.", font=SMALL_FONT, fill="#516273")
    image.save(DOCS / "agent-sequence.png")


if __name__ == "__main__":
    DOCS.mkdir(parents=True, exist_ok=True)
    interactions()
    sequence()
