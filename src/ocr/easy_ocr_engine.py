from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.schemas import OCRResult, OCRWord

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr

        _reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
    return _reader


def warmup_ocr(*, verbose: bool = True) -> None:
    if verbose:
        print("[OCR] EasyOCR 모델 로딩 중... (최초 1회, 약 30~60초)", flush=True)
    _get_reader()
    if verbose:
        print("[OCR] 모델 로딩 완료", flush=True)


def _to_path(image: np.ndarray | str | Path) -> str:
    if isinstance(image, (str, Path)):
        return str(image)
    import tempfile
    import os

    fd, path = tempfile.mkstemp(suffix=".jpg", prefix="ocr_")
    os.close(fd)
    cv2.imencode(".jpg", image)[1].tofile(path)
    return path


def run_ocr(
    image: np.ndarray | str | Path,
    *,
    verbose: bool = False,
) -> OCRResult:
    reader = _get_reader()
    path = _to_path(image)
    tmp = path if isinstance(image, np.ndarray) else None

    try:
        if verbose:
            print("[OCR] 텍스트 추출 중...", flush=True)
        raw = reader.readtext(
            path,
            paragraph=False,
            canvas_size=2240,
            mag_ratio=1.5,
            text_threshold=0.6,
            low_text=0.35,
        )
    finally:
        if tmp:
            import os
            os.remove(tmp)

    words: list[OCRWord] = []
    for bbox, text, conf in raw:
        text = (text or "").strip()
        if not text:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        words.append(
            OCRWord(
                text=text,
                confidence=float(conf),
                bbox=[int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
            )
        )

    words.sort(key=lambda w: (w.bbox[1], w.bbox[0]) if w.bbox else (0, 0))
    full_text = "\n".join(w.text for w in words)

    if verbose:
        print(f"[OCR] 완료 — {len(words)}개 텍스트 추출", flush=True)

    return OCRResult(words=words, full_text=full_text)
