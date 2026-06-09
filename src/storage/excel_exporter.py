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


def export_to_excel(
    results: list[DocumentResult], filename: str | None = None
) -> Path:
    """결과 리스트를 xlsx로 저장하고 경로를 반환한다."""
    out_dir = ensure_dir(load_config().get("paths", {}).get("output_dir", "data/output"))
    if filename is None:
        filename = f"documents_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    out_path = out_dir / filename

    receipt_df = pd.DataFrame(_receipt_rows(results))
    card_df = pd.DataFrame(_card_rows(results))

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # 빈 DataFrame이어도 시트는 생성해 일관성 유지
        (receipt_df if not receipt_df.empty else pd.DataFrame(
            columns=["파일", "상호명", "거래일", "품목", "공급가액", "부가세", "합계", "결제수단"]
        )).to_excel(writer, sheet_name="Receipts", index=False)

        (card_df if not card_df.empty else pd.DataFrame(
            columns=["파일", "이름", "회사", "직책", "전화", "이메일", "주소", "웹사이트"]
        )).to_excel(writer, sheet_name="BusinessCards", index=False)

    return out_path
