"""Usage: uv run python -m pbn samples/cat.jpg [--colors 24] [--min-region 0.0004] [--out output]"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from pbn import render
from pbn.pipeline import Params, run


def main() -> None:
    ap = argparse.ArgumentParser(prog="pbn")
    ap.add_argument("image", type=Path)
    ap.add_argument("--colors", type=int, default=Params.n_colors)
    ap.add_argument("--min-region", type=float, default=Params.min_region_frac)
    ap.add_argument("--long-side", type=int, default=Params.work_long_side)
    ap.add_argument("--out", type=Path, default=Path("output"))
    args = ap.parse_args()

    rgb = np.asarray(ImageOps.exif_transpose(Image.open(args.image)).convert("RGB"))
    params = Params(work_long_side=args.long_side, n_colors=args.colors, min_region_frac=args.min_region)
    r = run(rgb, params)

    out = args.out / args.image.stem
    out.mkdir(parents=True, exist_ok=True)
    images = {
        "1-source": r.image,
        "2-smoothed": r.smoothed,
        "3-preview": render.preview(r),
        "4-numbered": render.numbered(r),
        "5-numbered-color": render.numbered(r, with_color=True),
        "6-palette": render.palette_sheet(r),
    }
    for name, img in images.items():
        Image.fromarray(img).save(out / f"{name}.png")
    (out / "stats.json").write_text(json.dumps(r.stats, indent=2))
    print(out, json.dumps(r.stats))


if __name__ == "__main__":
    main()
