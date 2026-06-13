from __future__ import annotations

RECEIPT_SCHEMA = """{
  "merchant": "상호명 (문자열)",
  "date": "거래일시 (YYYY-MM-DD 또는 원문)",
  "items": [{"name": "품목명", "quantity": 수량(숫자), "price": 금액(숫자)}],
  "subtotal": 공급가액(숫자 또는 null),
  "tax": 부가세(숫자 또는 null),
  "total": 합계금액(숫자 또는 null),
  "payment_method": "결제수단 (카드/현금 등)"
}"""

BUSINESS_CARD_SCHEMA = """{
  "name": "이름",
  "company": "회사명",
  "title": "직책",
  "phone": "전화번호",
  "email": "이메일",
  "address": "주소",
  "website": "웹사이트"
}"""

_BASE_INSTRUCTION = (
    "당신은 한국어 비즈니스 문서를 위한 정보 추출 엔진입니다. "
    "OCR 텍스트를 읽고 주어진 스키마에 맞는 유효한 JSON 객체만 반환하세요. "
    "설명, 마크다운, 코드 블록은 절대 추가하지 마세요.\n"

    "엄격한 규칙:\n"

    "1. 모든 값은 OCR 텍스트에 있는 그대로(VERBATIM) 복사하세요. "
    "절대로 값을 만들어내거나, 번역하거나, 의역하지 마세요. "
    "OCR 텍스트에 값이 존재하지 않으면 빈 문자열(\"\") 또는 null을 사용하세요.\n"

    "2. payment_method는 OCR 텍스트에 적힌 카드명 또는 결제수단명을 "
    "그대로 사용해야 합니다. "
    "(예: 'NH카드', 'IBK비씨카드', '신한카드', '현금') "
    "일반적인 카드명이나 다른 카드명으로 대체하지 마세요.\n"

    "3. date는 YYYY-MM-DD 형식으로 정규화해야 합니다.\n"

    "4. total은 최종 결제 금액입니다. "
    "보통 '합계', '총액', '결제금액', '받을금액' 등의 항목에 해당하며, "
    "일반적으로 가장 큰 금액입니다.\n"

    "5. 숫자는 통화 기호(₩, 원 등)나 천 단위 구분 쉼표 없이 "
    "숫자만 포함해야 합니다.\n"

    "6. OCR 텍스트에는 인식 오류(오타)가 포함될 수 있습니다. "
    "이 경우에도 추측해서 수정하지 말고 OCR에 인식된 그대로 복사하세요."
)


def build_prompt(doc_type: str, ocr_text: str) -> str:
    schema = BUSINESS_CARD_SCHEMA if doc_type == "business_card" else RECEIPT_SCHEMA
    kind = "business card" if doc_type == "business_card" else "receipt"
    return (
        f"{_BASE_INSTRUCTION}\n\n"
        f"Document type: {kind}\n"
        f"JSON schema:\n{schema}\n\n"
        f"OCR text:\n\"\"\"\n{ocr_text}\n\"\"\"\n\n"
        f"JSON:"
    )
