from __future__ import annotations

import json
import re

import requests

from src.config import load_config
from src.extraction.prompts import build_prompt
from src.schemas import (
    BusinessCardData,
    ReceiptData,
    ReceiptItem,
)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?:\+?\d{2,3}[-\s]?)?0?\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}")
_URL_RE = re.compile(r"(?:https?://)?(?:www\.)[\w./-]+", re.IGNORECASE)
_AMOUNT_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_CARD_RE = re.compile(r"[A-Za-z가-힣]{1,12}(?:체크|신용|채움)?카드")
_DATE_RE = re.compile(r"(\d{4})[./\-년\s]{1,2}(\d{1,2})[./\-월\s]{1,2}(\d{1,2})")

_CARD_ISSUERS = (
    "NH", "농협", "KB", "국민", "신한", "삼성", "현대", "롯데",
    "하나", "우리", "IBK", "기업", "비씨", "BC", "씨티", "카카오", "토스",
)


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _fix_payment_method(ocr_text: str, llm_value: str) -> str:
    squashed_ocr = _squash(ocr_text)
    llm_clean = (llm_value or "").strip()

    if llm_clean and _squash(llm_clean) in squashed_ocr:
        return llm_clean

    candidates = _CARD_RE.findall(ocr_text)
    for cand in candidates:
        if any(issuer.lower() in cand.lower() for issuer in _CARD_ISSUERS):
            return cand
    if candidates:
        return candidates[0]

    if "현금" in ocr_text:
        return "현금"

    if llm_clean in ("카드", "현금", "신용카드", "체크카드"):
        return llm_clean
    return ""


def _normalize_date(value: str, ocr_text: str = "") -> str:
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
    cfg = load_config().get("extraction", {})
    host = cfg.get("ollama_host", "http://localhost:11434")
    payload = {
        "model": cfg.get("model", "llama3"),
        "prompt": prompt,
        "stream": False,
        "format": "json",
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
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
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


def _regex_business_card(text: str) -> BusinessCardData:
    email = _EMAIL_RE.search(text)
    phone = _PHONE_RE.search(text)
    url = _URL_RE.search(text)
    return BusinessCardData(
        email=email.group(0) if email else "",
        phone=phone.group(0).strip() if phone else "",
        website=url.group(0) if url else "",
    )


_NOT_AMOUNT_KEYWORDS = (
    "카드번호", "승인", "단말", "전표", "사업자", "가맹", "포인트",
    "tel", "전화", "번호", "일시", "fax", "바코드",
)
_TOTAL_KEYWORDS = ("합계", "총액", "결제금액", "받을금액", "총 금액", "판매금액")

_MIN_AMOUNT, _MAX_AMOUNT = 10, 100_000_000


def _plausible_amount(value: float | None) -> bool:
    return value is not None and _MIN_AMOUNT <= value < _MAX_AMOUNT


def _find_total(text: str) -> float | None:
    lines = text.splitlines()
    keyword_amounts: list[float] = []
    comma_amounts: list[float] = []

    for i, line in enumerate(lines):
        low = line.lower()
        if any(kw in low for kw in _NOT_AMOUNT_KEYWORDS):
            continue

        has_total_kw = any(kw in line for kw in _TOTAL_KEYWORDS) or (
            i > 0 and any(kw in lines[i - 1] for kw in _TOTAL_KEYWORDS)
        )

        for raw in _AMOUNT_RE.findall(line):
            val = _to_float(raw)
            if not _plausible_amount(val):
                continue
            if has_total_kw:
                keyword_amounts.append(val)
            if "," in raw:
                comma_amounts.append(val)

    if keyword_amounts:
        return max(keyword_amounts)
    if comma_amounts:
        return max(comma_amounts)
    return None


def _regex_receipt(text: str) -> ReceiptData:
    return ReceiptData(total=_find_total(text))


def extract_receipt(ocr_text: str) -> ReceiptData:
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
                if not _plausible_amount(receipt.total):
                    receipt.total = _find_total(ocr_text)
                if not _plausible_amount(receipt.tax):
                    receipt.tax = None
                if not _plausible_amount(receipt.subtotal):
                    receipt.subtotal = None
                return receipt
            except Exception:
                continue

    fallback = _regex_receipt(ocr_text)
    fallback.date = _normalize_date("", ocr_text)
    fallback.payment_method = _fix_payment_method(ocr_text, "")
    return fallback


def extract_business_card(ocr_text: str) -> BusinessCardData:
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
                rx = _regex_business_card(ocr_text)
                card.email = card.email or rx.email
                card.phone = card.phone or rx.phone
                card.website = card.website or rx.website
                return card
            except Exception:
                continue

    return _regex_business_card(ocr_text)


def is_ollama_available() -> bool:
    cfg = load_config().get("extraction", {})
    host = cfg.get("ollama_host", "http://localhost:11434")
    try:
        resp = requests.get(f"{host}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False
