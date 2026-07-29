from __future__ import annotations

import base64
from contextlib import ExitStack
from pathlib import Path

from openai import OpenAI

from .config import Settings


class ImageGenerationClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.has_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        self.settings = settings
        self.client = OpenAI()

    def edit(
        self,
        reference_paths: list[Path],
        prompt: str,
        output_path: Path,
    ) -> Path:
        if not reference_paths:
            raise ValueError("At least one reference image is required")
        missing = [path for path in reference_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing reference images: {missing}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            images = [stack.enter_context(path.open("rb")) for path in reference_paths]
            result = self.client.images.edit(
                model=self.settings.image_model,
                image=images,
                prompt=prompt,
                size=self.settings.image_size,
                quality=self.settings.image_quality,
            )
        if not result.data or not result.data[0].b64_json:
            raise RuntimeError("Image API returned no image data")
        output_path.write_bytes(base64.b64decode(result.data[0].b64_json))
        return output_path

