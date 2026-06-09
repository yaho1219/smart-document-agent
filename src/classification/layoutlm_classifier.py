"""LayoutLM 기반 문서 분류 (영수증 / 명함 / 미상).

하이브리드 1단계. 학습된 LayoutLM 가중치가 있으면 bbox+text를 함께
사용해 분류하고, 없거나 신뢰도가 낮으면 키워드 규칙으로 fallback 한다.

LayoutLM은 OCR 박스 좌표를 0~1000 정규화해 입력받는 점이 일반
텍스트 분류기와 다르다.

담당: PM & AI Lead (모델 선정/통합), NLP Engineer (규칙)
"""
from __future__ import annotations

from pathlib import Path

from src.config import load_config, resolve_path
from src.schemas import ClassificationResult, DocumentType, OCRResult

# --- 키워드 fallback 사전 -------------------------------------------

_RECEIPT_KEYWORDS = [
    "합계", "소계", "부가세", "공급가액", "결제", "카드", "현금영수증",
    "영수증", "거래", "금액", "단가", "수량", "포인트", "승인", "원",
    "total", "subtotal", "tax", "cash", "card", "vat", "receipt",
]
_CARD_KEYWORDS = [
    "대표", "이사", "팀장", "과장", "부장", "매니저", "사원", "대리",
    "주식회사", "(주)", "tel", "fax", "mobile", "e-mail", "email",
    "@", "ceo", "manager", "director", "co., ltd", "www.", "http",
]

_model = None
_tokenizer = None


def _keyword_score(text: str, keywords: list[str]) -> int:
    low = text.lower()
    return sum(1 for kw in keywords if kw.lower() in low)


def _fallback_classify(ocr: OCRResult) -> ClassificationResult:
    """키워드 출현 빈도로 문서 유형을 추정한다."""
    text = ocr.full_text
    if not text.strip():
        return ClassificationResult(
            doc_type=DocumentType.UNKNOWN, confidence=0.0, source="fallback"
        )

    receipt_score = _keyword_score(text, _RECEIPT_KEYWORDS)
    card_score = _keyword_score(text, _CARD_KEYWORDS)
    total = receipt_score + card_score

    if total == 0:
        return ClassificationResult(
            doc_type=DocumentType.UNKNOWN, confidence=0.0, source="fallback"
        )

    if receipt_score >= card_score:
        doc_type = DocumentType.RECEIPT
        confidence = receipt_score / total
    else:
        doc_type = DocumentType.BUSINESS_CARD
        confidence = card_score / total

    return ClassificationResult(
        doc_type=doc_type, confidence=round(confidence, 3), source="fallback"
    )


def _load_layoutlm():
    """학습된 LayoutLM 가중치를 1회 로드한다. 없으면 (None, None)."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    cfg = load_config().get("classification", {})
    weights_dir = resolve_path(
        load_config().get("paths", {}).get(
            "layoutlm_weights", "models/layoutlm_classifier"
        )
    )
    if not Path(weights_dir).exists():
        return None, None

    try:
        import torch  # noqa: F401
        from transformers import (
            LayoutLMForSequenceClassification,
            LayoutLMTokenizer,
        )

        _tokenizer = LayoutLMTokenizer.from_pretrained(str(weights_dir))
        _model = LayoutLMForSequenceClassification.from_pretrained(str(weights_dir))
        _model.eval()
        return _model, _tokenizer
    except Exception:
        # 로딩 실패 시 안전하게 fallback로 회귀
        return None, None


def _normalize_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    """OCR 픽셀 bbox를 LayoutLM의 0~1000 좌표계로 정규화한다."""
    if not bbox or width == 0 or height == 0:
        return [0, 0, 0, 0]
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(1000, int(1000 * x0 / width))),
        max(0, min(1000, int(1000 * y0 / height))),
        max(0, min(1000, int(1000 * x1 / width))),
        max(0, min(1000, int(1000 * y1 / height))),
    ]


def _layoutlm_classify(ocr: OCRResult) -> ClassificationResult | None:
    """LayoutLM 추론. 모델이 없거나 오류 시 None 반환."""
    model, tokenizer = _load_layoutlm()
    if model is None or not ocr.words:
        return None

    try:
        import torch

        cfg = load_config().get("classification", {})
        labels: list[str] = cfg.get(
            "labels", ["receipt", "business_card", "unknown"]
        )

        # 전체 이미지 좌표계 추정(최대 좌표값 사용)
        max_x = max((w.bbox[2] for w in ocr.words if w.bbox), default=1)
        max_y = max((w.bbox[3] for w in ocr.words if w.bbox), default=1)

        words = [w.text for w in ocr.words]
        boxes = [_normalize_bbox(w.bbox, max_x, max_y) for w in ocr.words]

        encoding = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )

        # 토큰별 bbox 정렬 (word_ids 기반)
        word_ids = encoding.word_ids(0)
        token_boxes = []
        for wid in word_ids:
            if wid is None:
                token_boxes.append([0, 0, 0, 0])
            else:
                token_boxes.append(boxes[wid])
        encoding["bbox"] = torch.tensor([token_boxes])

        with torch.no_grad():
            logits = model(**encoding).logits
            probs = torch.softmax(logits, dim=-1)[0]
            idx = int(torch.argmax(probs))
            confidence = float(probs[idx])

        label = labels[idx] if idx < len(labels) else "unknown"
        return ClassificationResult(
            doc_type=DocumentType(label),
            confidence=round(confidence, 3),
            source="layoutlm",
        )
    except Exception:
        return None


def classify_document(ocr: OCRResult) -> ClassificationResult:
    """문서 유형을 분류한다.

    1) LayoutLM 추론 시도
    2) 모델이 없거나 신뢰도가 임계값 미만이면 키워드 fallback 사용
    """
    cfg = load_config().get("classification", {})
    threshold = cfg.get("confidence_threshold", 0.55)

    layout_result = _layoutlm_classify(ocr)
    if layout_result and layout_result.confidence >= threshold:
        return layout_result

    return _fallback_classify(ocr)
