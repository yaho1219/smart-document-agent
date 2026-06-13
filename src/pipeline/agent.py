"""DocumentAgent: end-to-end 오케스트레이션.

이미지 입력 → OpenCV 전처리 → PaddleOCR → LayoutLM 분류 →
Ollama 구조화 → SQLite 저장으로 이어지는 전체 워크플로우를 담당한다.

담당: PM & AI Lead
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from src.classification.layoutlm_classifier import classify_document
from src.extraction.ollama_extractor import (
    extract_business_card,
    extract_receipt,
    is_ollama_available,
)
from src.ocr import run_ocr
from src.preprocessing.image_preprocess import PreprocessResult, preprocess_image
from src.schemas import DocumentResult, DocumentType
from src.storage.db import save_result

# 진행 상황 콜백: (단계명, 0~1 진행률) -> None
ProgressCb = Callable[[str, float], None]


def _noop(_stage: str, _pct: float) -> None:  # 기본 콜백
    pass


class DocumentAgent:
    """단일 이미지를 처리하는 파이프라인 에이전트."""

    def __init__(self, *, persist: bool = True):
        self.persist = persist

    def process_image(
        self,
        image_path: str | Path | np.ndarray,
        *,
        progress: ProgressCb = _noop,
        source_name: str | None = None,
        verbose: bool = False,
    ) -> tuple[DocumentResult, PreprocessResult]:
        """이미지 1장을 처리해 결과와 전처리 산출물을 반환한다."""
        source = source_name or (
            str(image_path) if isinstance(image_path, (str, Path)) else "memory_image"
        )
        warnings: list[str] = []

        # 1) 전처리
        progress("이미지 전처리", 0.08)
        pre = preprocess_image(image_path)
        progress("이미지 전처리 완료", 0.18)

        # 2) OCR — 전처리본과 원본 둘 다 인식 후 신뢰도 높은 쪽 채택
        #    (사진 품질에 따라 유리한 입력이 달라 인식률을 크게 좌우)
        progress("OCR 모델 준비", 0.22)
        ocr_proc = run_ocr(pre.processed, verbose=verbose)
        progress("OCR 원본 재인식", 0.32)
        ocr_orig = run_ocr(pre.original, verbose=verbose)

        def _score(o):
            return sum(w.confidence for w in o.words)

        ocr = ocr_proc if _score(ocr_proc) >= _score(ocr_orig) else ocr_orig

        progress("OCR 추출 완료", 0.42)
        if not ocr.words:
            warnings.append("OCR에서 텍스트를 추출하지 못했습니다.")

        # 3) 분류
        progress("문서 분류", 0.50)
        classification = classify_document(ocr)
        progress("문서 분류 완료", 0.58)
        if classification.doc_type == DocumentType.UNKNOWN:
            warnings.append("문서 유형을 판별하지 못했습니다(미상).")

        # 4) 구조화 (Ollama LLM — 수십 초 걸릴 수 있음)
        progress("정보 구조화 (Ollama)", 0.62)
        if not is_ollama_available():
            warnings.append("Ollama 서버 미연결: 정규식 fallback으로 추출합니다.")
        elif verbose:
            print("[LLM] Ollama로 필드 추출 중... (30초~2분 소요 가능)", flush=True)

        receipt = None
        business_card = None
        if classification.doc_type == DocumentType.RECEIPT:
            receipt = extract_receipt(ocr.full_text)
        elif classification.doc_type == DocumentType.BUSINESS_CARD:
            business_card = extract_business_card(ocr.full_text)
        progress("정보 구조화 완료", 0.85)

        result = DocumentResult(
            source_file=source,
            ocr=ocr,
            classification=classification,
            receipt=receipt,
            business_card=business_card,
            warnings=warnings,
        )

        # 5) 저장
        progress("저장", 0.9)
        if self.persist and classification.doc_type != DocumentType.UNKNOWN:
            try:
                save_result(result)
            except Exception as exc:  # 저장 실패가 전체를 막지 않도록
                result.warnings.append(f"DB 저장 실패: {exc}")

        progress("완료", 1.0)
        return result, pre
