# Paint by Numbers

Turn a photo into a beginner-friendly paint-by-numbers kit: a reduced color palette, numbered line art,
and a colored preview.

The pipeline is classic image processing — no generative AI. It is deterministic (same input and settings
give the same kit) and processes a phone photo in about two seconds.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync
uv run python -m pbn samples/cat.jpg
```

Results are written to `output/<image name>/`:

| File | Contents |
|---|---|
| `1-source.png` | Input, resized to the working size |
| `2-smoothed.png` | After texture flattening |
| `3-preview.png` | Finished painting, filled with palette colors |
| `4-numbered.png` | Printable template: outlines and numbers on white |
| `5-numbered-color.png` | Template over the preview, for checking region placement |
| `6-palette.png` | Numbered color swatches |
| `stats.json` | Image size, color count, region count, minimum region size |

### Options

| Flag | Default | Effect |
|---|---|---|
| `--colors` | `24` | Maximum palette size. The final palette may be smaller (see color spacing below). |
| `--min-region` | `0.0004` | Smallest paintable region, as a fraction of image area. Raise it for fewer, larger regions. |
| `--long-side` | `1200` | Working resolution in pixels on the long side. |
| `--out` | `output` | Output directory. |

Finer settings live on `Params` in [pbn/pipeline.py](pbn/pipeline.py).

## How it works

All color math happens in CIELAB, so distances (Delta E) match how different two colors look.

1. **Smooth.** Mean-shift and bilateral filtering flatten texture (fur, crumbs, wood grain) while
   keeping edges sharp.
2. **Pick colors.** K-means chooses up to `n_colors` colors.
3. **Space colors apart.** Palette colors closer than `min_color_gap` (10 Delta E) are merged. A large
   gradient, like a white chest, otherwise soaks up a ramp of near-identical colors and paints as many
   thin contour bands; spacing keeps the shading as a few clearly different steps.
4. **Rescue missed colors.** K-means favors colors that cover many pixels, so a small but distinctive
   area — a pink nose, olive-green eyes — can end up with no close color. Up to four times, the
   largest patch that no palette color matches well gets its own color.
5. **Straighten borders.** A majority filter replaces ragged, pixel-noisy borders with smooth ones, then
   restores small high-contrast details it erased, such as eye catchlights.
6. **Merge small regions.** Regions below the minimum size are repainted with the neighbor they share
   the longest border with. Small regions that differ sharply from their surroundings are kept, because
   they often carry the likeness.
7. **Render.** Outlines are drawn between regions, and each number sits at the point farthest from its
   region's border.

## Samples

| Image | Why it's there |
|---|---|
| `samples/cat.jpg` | One subject on a soft background; tests small features (eyes, catchlights, pink nose) and smooth fur gradients. |
| `samples/breakfast.jpg` | Several distinct objects and colors; tests small-region merging (egg, bagel speckles) and saturated accents. |

Put photos you don't want committed, such as photos of people, in `samples/private/`; it is git-ignored.

## Development

```bash
uv run pytest
```

Tests in [tests/test_pipeline.py](tests/test_pipeline.py) cover color rescue, color spacing, and
catchlight restoration on small synthetic images.

## Status and roadmap

This is a working prototype. Next up:

- **Printable kit:** an A4 PDF with a colored reference, a numbered template with light lines, a
  clean-borders page, and a palette page with hex codes, at full print resolution.
- **Detail level:** a beginner / standard / detailed setting that trades likeness against how easy the
  kit is to paint.
- **Number sizing:** guarantee every number is readable at print size.
