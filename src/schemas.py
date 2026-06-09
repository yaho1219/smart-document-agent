"""파이프라인 전역에서 사용하는 데이터 스키마.

OCR 결과 → 분류 → 구조화 → 저장까지 일관된 타입으로 흐르게 한다.
Pydantic v2 기반으로 LLM 출력 검증에 사용한다.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    RECEIPT = "receipt"
    BUSINESS_CARD = "business_card"
    UNKNOWN = "unknown"


# --- OCR ---------------------------------------------------------------


class OCRWord(BaseModel):
    """OCR 한 토막의 텍스트와 위치(bbox), 신뢰도."""

    text: str
    confidence: float = 0.0
    # bbox: [x_min, y_min, x_max, y_max]
    bbox: list[int] = Field(default_factory=list)


class OCRResult(BaseModel):
    words: list[OCRWord] = Field(default_factory=list)
    full_text: str = ""

    @property
    def mean_confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.confidence for w in self.words) / len(self.words)


# --- 구조화된 결과 -----------------------------------------------------


class ReceiptItem(BaseModel):
    name: str = ""
    quantity: Optional[float] = None
    price: Optional[float] = None


class ReceiptData(BaseModel):
    merchant: str = ""
    date: str = ""
    items: list[ReceiptItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    payment_method: str = ""


class BusinessCardData(BaseModel):
    name: str = ""
    company: str = ""
    title: str = ""
    phone: str = ""
    email: str = ""
    address: str = ""
    website: str = ""


# --- 파이프라인 최종 산출물 -------------------------------------------


class ClassificationResult(BaseModel):
    doc_type: DocumentType = DocumentType.UNKNOWN
    confidence: float = 0.0
    source: str = "fallback"  # "layoutlm" | "fallback"


class DocumentResult(BaseModel):
    """단일 이미지 처리의 end-to-end 결과."""

    source_file: str
    processed_at: datetime = Field(default_factory=datetime.now)
    ocr: OCRResult
    classification: ClassificationResult
    receipt: Optional[ReceiptData] = None
    business_card: Optional[BusinessCardData] = None
    warnings: list[str] = Field(default_factory=list)

    def structured_dict(self) -> dict:
        """저장/표시용 평탄화된 dict."""
        if self.classification.doc_type == DocumentType.RECEIPT and self.receipt:
            return self.receipt.model_dump()
        if (
            self.classification.doc_type == DocumentType.BUSINESS_CARD
            and self.business_card
        ):
            return self.business_card.model_dump()
        return {}
