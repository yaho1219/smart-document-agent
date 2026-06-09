"""설정 로딩 유틸리티.

`config/settings.yaml`을 읽어 dict로 제공하고, 프로젝트 루트 기준
상대경로를 절대경로로 해석하는 헬퍼를 둔다.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """settings.yaml을 한 번만 읽어 캐싱한다."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(relative: str | Path) -> Path:
    """프로젝트 루트 기준 상대경로를 절대경로로 변환한다."""
    p = Path(relative)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def ensure_dir(relative: str | Path) -> Path:
    """디렉터리를 보장(생성)하고 절대경로를 반환한다."""
    path = resolve_path(relative)
    path.mkdir(parents=True, exist_ok=True)
    return path
