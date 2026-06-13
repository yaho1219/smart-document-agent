from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config
from src.schemas import OCRResult, OCRWord

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_engine = None


def _get_engine(*, verbose: bool = False):
    global _engine
    if _engine is None:
        if verbose:
            print(
                "[OCR] PaddleOCR 모델 로딩 중... "
                "(최초 1회, 약 10~30초)",
                flush=True,
            )
        from paddleocr import PaddleOCR

        cfg = load_config().get("ocr", {})
        _engine = PaddleOCR(
            lang=cfg.get("lang", "korean"),
            use_angle_cls=cfg.get("use_angle_cls", False),
            use_gpu=cfg.get("use_gpu", False),
            show_log=False,
            cpu_threads=cfg.get("cpu_threads", 1),
            enable_mkldnn=False,
        )
        if verbose:
            print("[OCR] 모델 로딩 완료", flush=True)
    return _engine


def warmup_ocr(*, verbose: bool = True) -> None:
    _get_engine(verbose=verbose)


def _to_xyxy(box: list[list[float]]) -> list[int]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _parse_raw(raw) -> OCRResult:
    words: list[OCRWord] = []
    pages = raw if raw else []
    for page in pages:
        if not page:
            continue
        for line in page:
            box, (text, conf) = line[0], line[1]
            text = (text or "").strip()
            if not text:
                continue
            words.append(
                OCRWord(text=text, confidence=float(conf), bbox=_to_xyxy(box))
            )

    words.sort(key=lambda w: (w.bbox[1], w.bbox[0]) if w.bbox else (0, 0))
    full_text = "\n".join(w.text for w in words)
    return OCRResult(words=words, full_text=full_text)


def _to_ocr_path(image: np.ndarray | str | Path) -> tuple[str, str | None]:
    if isinstance(image, (str, Path)):
        return str(image), None

    fd, path = tempfile.mkstemp(suffix=".jpg", prefix="ocr_")
    os.close(fd)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("OCR용 임시 이미지 인코딩 실패")
    buf.tofile(path)
    return path, path


def _call_ocr(engine, image_path: str, *, use_cls: bool, timeout: int):
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(engine.ocr, image_path, cls=use_cls)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout:
            raise TimeoutError(
                f"OCR이 {timeout}초 안에 끝나지 않았습니다. "
                "다른 streamlit/python 프로세스가 실행 중이면 종료 후 재시도하세요."
            )


def run_ocr(
    image: np.ndarray | str | Path,
    *,
    verbose: bool = False,
) -> OCRResult:
    cfg = load_config().get("ocr", {})
    timeout = int(cfg.get("timeout_seconds", 120))
    use_cls = bool(cfg.get("use_angle_cls", False))

    engine = _get_engine(verbose=verbose)
    image_path, tmp_path = _to_ocr_path(image)

    try:
        if verbose:
            print(f"[OCR] 텍스트 추출 중... (최대 {timeout}초)", flush=True)

        raw = _call_ocr(engine, image_path, use_cls=use_cls, timeout=timeout)
        result = _parse_raw(raw)

        if not result.words and use_cls:
            if verbose:
                print("[OCR] 재시도 (각도 분류 없음)...", flush=True)
            raw = _call_ocr(engine, image_path, use_cls=False, timeout=timeout)
            result = _parse_raw(raw)

        if verbose:
            print(f"[OCR] 완료 — {len(result.words)}개 텍스트 추출", flush=True)
        return result
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
