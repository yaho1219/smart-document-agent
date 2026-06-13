"""로컬 LLM(Ollama) 정보 구조화용 프롬프트.

문서 유형별로 추출할 필드를 JSON 스키마로 명시해 LLM이 일관된
형식으로 응답하도록 유도한다.

담당: NLP Engineer
"""
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
    "You are an information extraction engine for Korean business documents. "
    "Read the OCR text and return ONLY a valid JSON object that matches the "
    "given schema. Do not add explanations, markdown, or code fences.\n"
    "STRICT RULES:\n"
    "1. Copy every value VERBATIM from the OCR text. NEVER invent, translate, "
    "or paraphrase. If a value is not present in the OCR text, use an empty "
    "string or null.\n"
    "2. payment_method must be the exact card/payment name as written in the "
    "OCR text (e.g. 'NH카드', 'IBK비씨카드', '신한카드', '현금'). Do not "
    "substitute it with a generic or different card name.\n"
    "3. date must be normalized to YYYY-MM-DD format.\n"
    "4. total is the final charged amount, usually labeled 합계, 총액, "
    "결제금액, or 받을금액. It is normally the largest amount.\n"
    "5. Numbers must be plain digits without currency symbols or thousands "
    "separators.\n"
    "6. OCR text may contain typos from misrecognition; still copy values "
    "as written rather than guessing replacements."
)


def build_prompt(doc_type: str, ocr_text: str) -> str:
    """문서 유형에 맞는 추출 프롬프트를 생성한다."""
    schema = BUSINESS_CARD_SCHEMA if doc_type == "business_card" else RECEIPT_SCHEMA
    kind = "business card" if doc_type == "business_card" else "receipt"
    return (
        f"{_BASE_INSTRUCTION}\n\n"
        f"Document type: {kind}\n"
        f"JSON schema:\n{schema}\n\n"
        f"OCR text:\n\"\"\"\n{ocr_text}\n\"\"\"\n\n"
        f"JSON:"
    )
