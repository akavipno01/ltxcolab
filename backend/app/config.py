from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "LTX-Video API"
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("LTX_VIDEO_DATA_DIR", BASE_DIR / "data")).resolve()
OUTPUTS_DIR = DATA_DIR / "outputs"
TEMP_DIR = DATA_DIR / "temp"
MODELS_DIR = DATA_DIR / "models"
MODEL_DIR = MODELS_DIR / "ltx-video"
DB_PATH = DATA_DIR / "ltx-video.sqlite3"

# Default to the distilled FP8 model requested
MODEL_ID = os.environ.get("LTX_VIDEO_MODEL", "a-r-r-o-w/LTXV-2B-0.9.8-Distilled-FP8")

def ensure_directories() -> None:
    for directory in (DATA_DIR, OUTPUTS_DIR, TEMP_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
