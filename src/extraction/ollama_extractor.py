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


def _regex_receipt(text: str) -> ReceiptData:
    amounts = [_to_float(a) for a in _AMOUNT_RE.findall(text)]
    amounts = [a for a in amounts if a is not None]
    total = max(amounts) if amounts else None
    return ReceiptData(total=total)


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
                return ReceiptData(
                    merchant=str(data.get("merchant", "")),
                    date=str(data.get("date", "")),
                    items=items,
                    subtotal=_to_float(data.get("subtotal")),
                    tax=_to_float(data.get("tax")),
                    total=_to_float(data.get("total")),
                    payment_method=str(data.get("payment_method", "")),
                )
            except Exception:
                continue

    # LLM 실패 → 정규식 fallback
    return _regex_receipt(ocr_text)


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
