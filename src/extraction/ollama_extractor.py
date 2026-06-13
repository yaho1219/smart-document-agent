"""Ollama 로컬 LLM을 이용한 JSON 정보 구조화.

하이브리드 2단계. OCR 원문과 문서 유형을 받아 로컬 Llama-3로
구조화된 JSON을 생성하고 Pydantic으로 검증한다. LLM이 없거나 응답이
불완전하면 정규식 기반 보조 추출로 핵심 필드를 채운다.

담당: NLP Engineer
"""
from __future__ import annotations

import json
import re

import requests

from src.config import load_config
from src.extraction.prompts import build_prompt
from src.schemas import (
    BusinessCardData,
    DocumentType,
    ReceiptData,
    ReceiptItem,
)

# --- 정규식 (fallback) ----------------------------------------------

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?:\+?\d{2,3}[-\s]?)?0?\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}")
_URL_RE = re.compile(r"(?:https?://)?(?:www\.)[\w./-]+", re.IGNORECASE)
# 1,234 / 12000 / 1,234.50 등 금액
_AMOUNT_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
# OCR 원문에서 카드/결제수단 후보 추출 (예: NH카드, IBK비씨카드, 신한체크카드)
_CARD_RE = re.compile(r"[A-Za-z가-힣]{1,12}(?:체크|신용|채움)?카드")
_DATE_RE = re.compile(r"(\d{4})[./\-년\s]{1,2}(\d{1,2})[./\-월\s]{1,2}(\d{1,2})")

# 국내 주요 카드사 키워드 (후보 우선순위 판단용)
_CARD_ISSUERS = (
    "NH", "농협", "KB", "국민", "신한", "삼성", "현대", "롯데",
    "하나", "우리", "IBK", "기업", "비씨", "BC", "씨티", "카카오", "토스",
)


def _squash(text: str) -> str:
    """공백 제거 소문자화 — 느슨한 포함 비교용."""
    return re.sub(r"\s+", "", text).lower()


def _fix_payment_method(ocr_text: str, llm_value: str) -> str:
    """LLM이 만든 결제수단이 OCR 원문에 실제로 있는지 검증한다.

    원문에 없는 값(환각)이면 원문에서 찾은 카드명으로 교체한다.
    """
    squashed_ocr = _squash(ocr_text)
    llm_clean = (llm_value or "").strip()

    # LLM 값이 원문에 그대로 존재하면 신뢰
    if llm_clean and _squash(llm_clean) in squashed_ocr:
        return llm_clean

    # 원문에서 카드명 후보 수집 → 카드사 키워드 포함 후보 우선
    candidates = _CARD_RE.findall(ocr_text)
    for cand in candidates:
        if any(issuer.lower() in cand.lower() for issuer in _CARD_ISSUERS):
            return cand
    if candidates:
        return candidates[0]

    # 현금 결제 단서
    if "현금" in ocr_text:
        return "현금"

    # 후보가 전혀 없으면 일반 표현(카드/현금)만 인정, 그 외 환각은 제거
    if llm_clean in ("카드", "현금", "신용카드", "체크카드"):
        return llm_clean
    return ""


def _normalize_date(value: str, ocr_text: str = "") -> str:
    """날짜를 YYYY-MM-DD로 정규화한다. 실패 시 OCR 원문에서 재탐색."""
    for source in (value or "", ocr_text):
        m = _DATE_RE.search(source)
        if m:
            y, mo, d = m.groups()
            try:
                if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
                    return f"{y}-{int(mo):02d}-{int(d):02d}"
            except ValueError:
                continue
    return (value or "").strip()


def _call_ollama(prompt: str) -> str | None:
    """Ollama /api/generate 호출. 실패 시 None."""
    cfg = load_config().get("extraction", {})
    host = cfg.get("ollama_host", "http://localhost:11434")
    payload = {
        "model": cfg.get("model", "llama3"),
        "prompt": prompt,
        "stream": False,
        "format": "json",  # Ollama가 JSON 출력을 강제하도록
        "options": {"temperature": cfg.get("temperature", 0.0)},
    }
    try:
        resp = requests.post(
            f"{host}/api/generate",
            json=payload,
            timeout=cfg.get("timeout_seconds", 120),
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception:
        return None


def _extract_json(text: str) -> dict | None:
    """LLM 응답 문자열에서 첫 번째 JSON 객체를 파싱한다."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 코드펜스/잡문이 섞인 경우 중괄호 블록만 추출
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _to_float(value) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^\d.]", "", str(value))
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


# --- 정규식 보조 추출 -----------------------------------------------


def _regex_business_card(text: str) -> BusinessCardData:
    email = _EMAIL_RE.search(text)
    phone = _PHONE_RE.search(text)
    url = _URL_RE.search(text)
    return BusinessCardData(
        email=email.group(0) if email else "",
        phone=phone.group(0).strip() if phone else "",
        website=url.group(0) if url else "",
    )


# 금액이 아닌 숫자가 섞이는 라인 (카드번호/승인번호/단말기 등)
_NOT_AMOUNT_KEYWORDS = (
    "카드번호", "승인", "단말", "전표", "사업자", "가맹", "포인트",
    "tel", "전화", "번호", "일시", "fax", "바코드",
)
_TOTAL_KEYWORDS = ("합계", "총액", "결제금액", "받을금액", "총 금액", "판매금액")

# 금액 타당 범위: 10원 ~ 1억 미만
_MIN_AMOUNT, _MAX_AMOUNT = 10, 100_000_000


def _plausible_amount(value: float | None) -> bool:
    return value is not None and _MIN_AMOUNT <= value < _MAX_AMOUNT


def _find_total(text: str) -> float | None:
    """OCR 원문에서 합계 금액을 추정한다.

    1) 합계/총액 키워드 라인의 금액 우선
    2) 없으면 콤마 표기 금액 중 최대값 (카드번호 등 숫자열 배제)
    """
    lines = text.splitlines()
    keyword_amounts: list[float] = []
    comma_amounts: list[float] = []

    for i, line in enumerate(lines):
        low = line.lower()
        if any(kw in low for kw in _NOT_AMOUNT_KEYWORDS):
            continue

        # OCR이 라벨/값을 다른 줄로 분리하는 경우가 많아 다음 줄까지 본다
        has_total_kw = any(kw in line for kw in _TOTAL_KEYWORDS) or (
            i > 0 and any(kw in lines[i - 1] for kw in _TOTAL_KEYWORDS)
        )

        for raw in _AMOUNT_RE.findall(line):
            val = _to_float(raw)
            if not _plausible_amount(val):
                continue
            if has_total_kw:
                keyword_amounts.append(val)
            if "," in raw:  # 콤마 표기는 금액일 가능성이 높음
                comma_amounts.append(val)

    if keyword_amounts:
        return max(keyword_amounts)
    if comma_amounts:
        return max(comma_amounts)
    return None


def _regex_receipt(text: str) -> ReceiptData:
    return ReceiptData(total=_find_total(text))


# --- 공개 API --------------------------------------------------------


def extract_receipt(ocr_text: str) -> ReceiptData:
    """영수증 OCR 텍스트를 ReceiptData로 구조화한다."""
    cfg = load_config().get("extraction", {})
    retries = max(0, cfg.get("max_retries", 1)) + 1
    prompt = build_prompt("receipt", ocr_text)

    for _ in range(retries):
        raw = _call_ollama(prompt)
        data = _extract_json(raw) if raw else None
        if data:
            items = []
            for it in data.get("items", []) or []:
                if isinstance(it, dict):
                    items.append(
                        ReceiptItem(
                            name=str(it.get("name", "")),
                            quantity=_to_float(it.get("quantity")),
                            price=_to_float(it.get("price")),
                        )
                    )
            try:
                receipt = ReceiptData(
                    merchant=str(data.get("merchant", "")),
                    date=_normalize_date(str(data.get("date", "")), ocr_text),
                    items=items,
                    subtotal=_to_float(data.get("subtotal")),
                    tax=_to_float(data.get("tax")),
                    total=_to_float(data.get("total")),
                    payment_method=_fix_payment_method(
                        ocr_text, str(data.get("payment_method", ""))
                    ),
                )
                # 합계가 비정상(누락/터무니없는 값)이면 원문 기반으로 교정
                if not _plausible_amount(receipt.total):
                    receipt.total = _find_total(ocr_text)
                # 부가세/공급가액도 타당성 검사 (LLM이 자릿수 환각하는 경우)
                if not _plausible_amount(receipt.tax):
                    receipt.tax = None
                if not _plausible_amount(receipt.subtotal):
                    receipt.subtotal = None
                return receipt
            except Exception:
                continue

    # LLM 실패 → 정규식 fallback
    fallback = _regex_receipt(ocr_text)
    fallback.date = _normalize_date("", ocr_text)
    fallback.payment_method = _fix_payment_method(ocr_text, "")
    return fallback


def extract_business_card(ocr_text: str) -> BusinessCardData:
    """명함 OCR 텍스트를 BusinessCardData로 구조화한다."""
    cfg = load_config().get("extraction", {})
    retries = max(0, cfg.get("max_retries", 1)) + 1
    prompt = build_prompt("business_card", ocr_text)

    for _ in range(retries):
        raw = _call_ollama(prompt)
        data = _extract_json(raw) if raw else None
        if data:
            try:
                card = BusinessCardData(
                    name=str(data.get("name", "")),
                    company=str(data.get("company", "")),
                    title=str(data.get("title", "")),
                    phone=str(data.get("phone", "")),
                    email=str(data.get("email", "")),
                    address=str(data.get("address", "")),
                    website=str(data.get("website", "")),
                )
                # 핵심 식별정보가 비면 정규식으로 보강
                rx = _regex_business_card(ocr_text)
                card.email = card.email or rx.email
                card.phone = card.phone or rx.phone
                card.website = card.website or rx.website
                return card
            except Exception:
                continue

    return _regex_business_card(ocr_text)


def is_ollama_available() -> bool:
    """Ollama 서버 가용성 확인 (UI 안내용)."""
    cfg = load_config().get("extraction", {})
    host = cfg.get("ollama_host", "http://localhost:11434")
    try:
        resp = requests.get(f"{host}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False
