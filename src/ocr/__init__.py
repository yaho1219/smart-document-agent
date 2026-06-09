"""OCR 엔진 팩토리.

Mac ARM에서는 PaddleOCR이 멈출 수 있어 기본값은 EasyOCR이다.
config/settings.yaml 의 ocr.engine 으로 변경 가능: easyocr | paddle
"""
from __future__ import annotations

from src.config import load_config


def _engine_name() -> str:
    return load_config().get("ocr", {}).get("engine", "easyocr")


def run_ocr(image, *, verbose: bool = False):
    if _engine_name() == "paddle":
        from src.ocr.paddle_ocr_engine import run_ocr as _run
    else:
        from src.ocr.easy_ocr_engine import run_ocr as _run
    return _run(image, verbose=verbose)


def warmup_ocr(*, verbose: bool = True) -> None:
    if _engine_name() == "paddle":
        from src.ocr.paddle_ocr_engine import warmup_ocr as _warmup
    else:
        from src.ocr.easy_ocr_engine import warmup_ocr as _warmup
    _warmup(verbose=verbose)
