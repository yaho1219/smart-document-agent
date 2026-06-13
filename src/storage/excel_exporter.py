"""Excel 내보내기.

처리 결과를 영수증/명함 시트로 분리해 .xlsx로 저장한다.

담당: Dev Engineer
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import ensure_dir, load_config
from src.schemas import DocumentResult, DocumentType


def _receipt_rows(results: list[DocumentResult]) -> list[dict]:
    rows = []
    for r in results:
        if r.classification.doc_type != DocumentType.RECEIPT or not r.receipt:
            continue
        rc = r.receipt
        item_names = "; ".join(i.name for i in rc.items if i.name)
        rows.append(
            {
                "파일": r.source_file,
                "상호명": rc.merchant,
                "거래일": rc.date,
                "품목": item_names,
                "공급가액": rc.subtotal,
                "부가세": rc.tax,
                "합계": rc.total,
                "결제수단": rc.payment_method,
                "신뢰도": r.classification.confidence,
                "처리시각": r.processed_at.isoformat(timespec="seconds"),
            }
        )
    return rows


def _card_rows(results: list[DocumentResult]) -> list[dict]:
    rows = []
    for r in results:
        if r.classification.doc_type != DocumentType.BUSINESS_CARD or not r.business_card:
            continue
        bc = r.business_card
        rows.append(
            {
                "파일": r.source_file,
                "이름": bc.name,
                "회사": bc.company,
                "직책": bc.title,
                "전화": bc.phone,
                "이메일": bc.email,
                "주소": bc.address,
                "웹사이트": bc.website,
                "신뢰도": r.classification.confidence,
                "처리시각": r.processed_at.isoformat(timespec="seconds"),
            }
        )
    return rows


_RECEIPT_COLUMNS = [
    "파일", "상호명", "거래일", "품목", "공급가액", "부가세", "합계",
    "결제수단", "신뢰도", "처리시각",
]
_CARD_COLUMNS = [
    "파일", "이름", "회사", "직책", "전화", "이메일", "주소", "웹사이트",
    "신뢰도", "처리시각",
]


def _load_existing(path: Path, sheet: str, columns: list[str]) -> pd.DataFrame:
    """기존 파일의 시트를 읽는다. 없거나 깨졌으면 빈 DataFrame."""
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except Exception:
        return pd.DataFrame(columns=columns)


def export_to_excel(
    results: list[DocumentResult], filename: str | None = None
) -> Path:
    """결과를 날짜별 xlsx 파일 하나에 누적 저장한다.

    같은 날 여러 번 처리해도 documents_YYYYMMDD.xlsx 한 파일에
    행이 계속 추가된다. (파일+처리시각 기준 중복 제거)
    """
    out_dir = ensure_dir(load_config().get("paths", {}).get("output_dir", "data/output"))
    if filename is None:
        filename = f"documents_{datetime.now():%Y%m%d}.xlsx"
    out_path = out_dir / filename

    new_receipts = pd.DataFrame(_receipt_rows(results), columns=_RECEIPT_COLUMNS)
    new_cards = pd.DataFrame(_card_rows(results), columns=_CARD_COLUMNS)

    receipt_df = pd.concat(
        [_load_existing(out_path, "Receipts", _RECEIPT_COLUMNS), new_receipts],
        ignore_index=True,
    ).drop_duplicates(subset=["파일", "처리시각"], keep="last")

    card_df = pd.concat(
        [_load_existing(out_path, "BusinessCards", _CARD_COLUMNS), new_cards],
        ignore_index=True,
    ).drop_duplicates(subset=["파일", "처리시각"], keep="last")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        receipt_df.to_excel(writer, sheet_name="Receipts", index=False)
        card_df.to_excel(writer, sheet_name="BusinessCards", index=False)

    return out_path
