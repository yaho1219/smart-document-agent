def save_result(result: DocumentResult) -> int:
    conn.execute(
        "INSERT INTO documents (source_file, doc_type, confidence, ocr_text, structured, processed_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            result.source_file,
            result.classification.doc_type.value,
            result.classification.confidence,
            result.ocr.full_text,
            json.dumps(result.structured_dict(), ensure_ascii=False),
            result.processed_at.isoformat(),
        ),
    )





    