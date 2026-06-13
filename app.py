from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.extraction.ollama_extractor import is_ollama_available
from src.pipeline.agent import DocumentAgent
from src.schemas import DocumentResult, DocumentType
from src.storage import db
from src.storage.excel_exporter import export_to_excel

st.set_page_config(page_title="스마트 문서 정리 에이전트", page_icon="📄", layout="wide")


def _bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _init_state():
    if "results" not in st.session_state:
        st.session_state.results: list[DocumentResult] = []


def _sidebar():
    cfg = load_config()
    st.sidebar.header("⚙️ 설정")

    ext_cfg = cfg.get("extraction", {})
    model = st.sidebar.text_input("Ollama 모델", value=ext_cfg.get("model", "llama3"))
    threshold = st.sidebar.slider(
        "LayoutLM 분류 신뢰도 임계값",
        0.0, 1.0,
        float(cfg.get("classification", {}).get("confidence_threshold", 0.55)),
        0.05,
    )

    cfg.setdefault("extraction", {})["model"] = model
    cfg.setdefault("classification", {})["confidence_threshold"] = threshold

    st.sidebar.divider()
    ollama_ok = is_ollama_available()
    st.sidebar.markdown(
        f"**Ollama 상태:** {'🟢 연결됨' if ollama_ok else '🔴 미연결 (fallback)'}"
    )
    if not ollama_ok:
        st.sidebar.caption("Ollama 미연결 시 정규식 fallback으로 동작합니다.")

    st.sidebar.divider()
    st.sidebar.caption(
        "온프레미스 파이프라인\n\n"
        "OpenCV → PaddleOCR → LayoutLM → Ollama(Llama-3) → Excel/SQLite"
    )


def _render_result(result: DocumentResult, pre):
    doc_type = result.classification.doc_type
    badge = {
        DocumentType.RECEIPT: "🧾 영수증",
        DocumentType.BUSINESS_CARD: "💼 명함",
        DocumentType.UNKNOWN: "❓ 미상",
    }[doc_type]

    st.markdown(
        f"### {badge}  ·  신뢰도 `{result.classification.confidence}` "
        f"(`{result.classification.source}`)"
    )

    for w in result.warnings:
        st.warning(w)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**전처리 전/후**")
        st.image(_bgr_to_rgb(pre.original), caption="원본", use_column_width=True)
        st.image(_bgr_to_rgb(pre.processed), caption="전처리", use_column_width=True)
        st.caption(f"디버그: {pre.debug}")

    with col2:
        with st.expander("① OCR 추출 결과", expanded=False):
            st.text(result.ocr.full_text or "(추출된 텍스트 없음)")
            st.caption(f"평균 신뢰도: {result.ocr.mean_confidence:.3f}")

        st.markdown("**② 구조화 결과 (JSON)**")
        structured = result.structured_dict()
        if structured:
            st.json(structured)
        else:
            st.info("구조화된 필드가 없습니다.")


def tab_upload():
    st.subheader("이미지 업로드 및 처리")
    files = st.file_uploader(
        "명함/영수증 이미지를 업로드하세요 (다중 선택 가능)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if files and st.button("🚀 파이프라인 실행", type="primary"):
        agent = DocumentAgent(persist=True)
        progress_bar = st.progress(0.0, text="대기 중...")

        for idx, file in enumerate(files):
            import numpy as np

            data = np.frombuffer(file.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                st.error(f"이미지 디코딩 실패: {file.name}")
                continue

            def cb(stage, pct, _name=file.name):
                progress_bar.progress(
                    min(1.0, (idx + pct) / len(files)),
                    text=f"[{_name}] {stage}",
                )

            result, pre = agent.process_image(
                img, progress=cb, source_name=file.name
            )
            st.session_state.results.append(result)

            st.divider()
            st.markdown(f"## 📎 {file.name}")
            _render_result(result, pre)

        progress_bar.progress(1.0, text="모든 처리 완료")


def tab_export():
    st.subheader("내보내기 및 이력")

    results = st.session_state.results
    st.markdown(f"이번 세션 처리 건수: **{len(results)}**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Excel로 내보내기", disabled=not results):
            path = export_to_excel(results)
            st.success(f"저장 완료: {path}")
            with open(path, "rb") as f:
                st.download_button(
                    "⬇️ Excel 다운로드",
                    data=f.read(),
                    file_name=path.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
    with col2:
        if st.button("🗑️ DB 이력 초기화"):
            db.clear_all()
            st.success("DB 이력을 초기화했습니다.")

    st.divider()
    st.markdown("### 🗄️ SQLite 저장 이력")
    rows = db.fetch_all()
    if rows:
        df = pd.DataFrame(rows)[
            ["id", "source_file", "doc_type", "confidence", "processed_at"]
        ]
        st.dataframe(df, use_container_width=True)
    else:
        st.info("저장된 이력이 없습니다.")


def main():
    _init_state()
    st.title("📄 온프레미스 스마트 명함·영수증 정리 에이전트")
    st.caption("로컬 딥러닝 기반 · 외부 API 없음 · 데이터 유출 걱정 없는 보안 솔루션")

    _sidebar()

    tab1, tab2 = st.tabs(["① 업로드 & 처리", "② 내보내기 & 이력"])
    with tab1:
        tab_upload()
    with tab2:
        tab_export()


if __name__ == "__main__":
    main()
