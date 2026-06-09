"""PaddleOCR 래퍼.

한국어 인식 성능이 우수한 PaddleOCR을 로컬에서 구동한다.
Mac ARM 환경에서 스레드 데드락/멈춤을 피하기 위해 단일 스레드·
임시 파일 경로 입력·타임아웃을 적용한다.

담당: CV Engineer
"""
from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config
from src.schemas import OCRResult, OCRWord

# Mac CPU에서 OpenMP/MKL 스레드 충돌로 멈추는 경우 방지
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_engine = None


def _get_engine(*, verbose: bool = False):
    """PaddleOCR 인스턴스를 한 번만 생성해 재사용한다."""
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
            use_angle_cls=cfg.get("use_angle_cls", False),  # Mac에서 hang 방지
            use_gpu=cfg.get("use_gpu", False),
            show_log=False,
            cpu_threads=cfg.get("cpu_threads", 1),
            enable_mkldnn=False,
        )
        if verbose:
            print("[OCR] 모델 로딩 완료", flush=True)
    return _engine


def warmup_ocr(*, verbose: bool = True) -> None:
    """앱 시작 시 OCR 엔진을 미리 로드한다."""
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
    """PaddleOCR 입력용 경로를 반환한다. ndarray면 임시 파일로 저장."""
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
    """타임아웃을 걸어 OCR을 실행한다."""
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
    """이미지에서 텍스트를 추출한다."""
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
