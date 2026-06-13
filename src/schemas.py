from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    RECEIPT = "receipt"
    BUSINESS_CARD = "business_card"
    UNKNOWN = "unknown"


class OCRWord(BaseModel):
    text: str
    confidence: float = 0.0
    bbox: list[int] = Field(default_factory=list)


class OCRResult(BaseModel):
    words: list[OCRWord] = Field(default_factory=list)
    full_text: str = ""

    @property
    def mean_confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.confidence for w in self.words) / len(self.words)


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


class ClassificationResult(BaseModel):
    doc_type: DocumentType = DocumentType.UNKNOWN
    confidence: float = 0.0
    source: str = "fallback"


class DocumentResult(BaseModel):
    source_file: str
    processed_at: datetime = Field(default_factory=datetime.now)
    ocr: OCRResult
    classification: ClassificationResult
    receipt: Optional[ReceiptData] = None
    business_card: Optional[BusinessCardData] = None
    warnings: list[str] = Field(default_factory=list)

    def structured_dict(self) -> dict:
        if self.classification.doc_type == DocumentType.RECEIPT and self.receipt:
            return self.receipt.model_dump()
        if (
            self.classification.doc_type == DocumentType.BUSINESS_CARD
            and self.business_card
        ):
            return self.business_card.model_dump()
        return {}
