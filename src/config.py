from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(relative: str | Path) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def ensure_dir(relative: str | Path) -> Path:
    path = resolve_path(relative)
    path.mkdir(parents=True, exist_ok=True)
    return path
