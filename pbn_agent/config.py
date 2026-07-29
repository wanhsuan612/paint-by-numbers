from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    text_model: str = "gpt-5.6"
    image_model: str = "gpt-image-2"
    image_size: str = "1536x1024"
    image_quality: str = "high"

    @property
    def has_api_key(self) -> bool:
        value = os.getenv("OPENAI_API_KEY", "")
        return len(value.strip()) > 20


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env.local", override=False)
    return Settings(
        text_model=os.getenv("OPENAI_TEXT_MODEL", "gpt-5.6"),
        image_model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2"),
        image_size=os.getenv("OPENAI_IMAGE_SIZE", "1536x1024"),
        image_quality=os.getenv("OPENAI_IMAGE_QUALITY", "high"),
    )

