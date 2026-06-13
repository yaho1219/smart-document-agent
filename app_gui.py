#!/usr/bin/env python3
"""온프레미스 스마트 명함/영수증 자동 정리 에이전트 — 데스크톱 GUI (PyQt6).

macOS 26 등 최신 macOS에서 시스템 tkinter가 동작하지 않아 PyQt6를 사용합니다.

사용법:
    python app_gui.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.config import load_config, resolve_path
from src.extraction.ollama_extractor import is_ollama_available
from src.ocr import warmup_ocr
from src.pipeline.agent import DocumentAgent
from src.schemas import DocumentResult, DocumentType
from src.storage import db
from src.storage.excel_exporter import export_to_excel

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

DOC_LABEL = {
    DocumentType.RECEIPT: "영수증",
    DocumentType.BUSINESS_CARD: "명함",
    DocumentType.UNKNOWN: "미상",
}

FIELD_LABELS = {
    "merchant": "상호명", "date": "거래일", "total": "합계",
    "subtotal": "공급가액", "tax": "부가세", "payment_method": "결제수단",
    "name": "이름", "company": "회사", "title": "직책",
    "phone": "전화", "email": "이메일", "address": "주소", "website": "웹사이트",
    "items": "품목",
}


class WarmupThread(QThread):
    done = pyqtSignal(bool, str)

    def run(self) -> None:
        try:
            warmup_ocr(verbose=False)
            self.done.emit(True, "OCR 준비 완료 — 이미지를 추가하세요")
        except Exception as exc:
            self.done.emit(False, f"OCR 로딩 실패: {exc}")


class ProcessThread(QThread):
    progress = pyqtSignal(str, float)
    result_ready = pyqtSignal(object)
    excel_saved = pyqtSignal(str)
    error = pyqtSignal(str, str)
    finished_all = pyqtSignal()

    def __init__(self, files: list[Path]) -> None:
        super().__init__()
        self.files = files

    def run(self) -> None:
        agent = DocumentAgent(persist=True)
        results: list[DocumentResult] = []
        total = len(self.files)

        for i, path in enumerate(self.files):
            def cb(stage: str, pct: float, _i=i, _n=total) -> None:
                self.progress.emit(stage, ((_i) + pct) / _n * 100)

            try:
                result, _ = agent.process_image(
                    path, progress=cb, source_name=path.name, verbose=False
                )
                results.append(result)
                self.result_ready.emit(result)
            except Exception as exc:
                self.error.emit(path.name, str(exc))

        if results:
            try:
                out = export_to_excel(results)
                self.excel_saved.emit(str(out))
            except Exception as exc:
                self.error.emit("Excel", str(exc))

        self.finished_all.emit()


class DocumentAgentApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("스마트 명함·영수증 정리 에이전트")
        self.resize(1000, 700)
        self.setMinimumSize(860, 580)

        self._files: list[Path] = []
        self._results: list[DocumentResult] = []
        self._process_thread: ProcessThread | None = None

        self._build_ui()
        self._update_ollama_status()
        self._warmup = WarmupThread()
        self._warmup.done.connect(self._on_warmup_done)
        self._status.showMessage("OCR 모델 로딩 중... (최초 1회 30~60초)")
        self._warmup.start()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # ── 왼쪽 패널 ──
        left = QWidget()
        left.setFixedWidth(280)
        left_layout = QVBoxLayout(left)

        title = QLabel("처리할 이미지")
        title.setFont(QFont("", 13, QFont.Weight.Bold))
        left_layout.addWidget(title)

        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._file_list.currentRowChanged.connect(self._on_select_file)
        left_layout.addWidget(self._file_list)

        btn_row = QHBoxLayout()
        for text, slot in [
            ("파일 추가", self._add_files),
            ("폴더 추가", self._add_folder),
            ("삭제", self._remove_selected),
        ]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        left_layout.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        left_layout.addWidget(self._progress)

        self._stage_lbl = QLabel("대기 중")
        self._stage_lbl.setStyleSheet("color: #555;")
        left_layout.addWidget(self._stage_lbl)

        self._run_btn = QPushButton("▶  처리 시작")
        self._run_btn.setStyleSheet(
            "QPushButton { background: #2563eb; color: white; padding: 10px; "
            "font-weight: bold; border-radius: 6px; }"
            "QPushButton:disabled { background: #94a3b8; }"
        )
        self._run_btn.clicked.connect(self._start_processing)
        left_layout.addWidget(self._run_btn)

        for text, slot in [
            ("Excel보내기", self._export_excel),
            ("출력 폴더 열기", self._open_output_dir),
        ]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            left_layout.addWidget(b)

        # ── 오른쪽 탭 ──
        self._tabs = QTabWidget()

        self._preview_lbl = QLabel("이미지를 선택하세요")
        self._preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_lbl.setStyleSheet("background: #f1f5f9; border-radius: 8px;")
        self._tabs.addTab(self._preview_lbl, "미리보기")

        self._ocr_text = QTextEdit()
        self._ocr_text.setReadOnly(True)
        self._ocr_text.setFont(QFont("", 12))
        self._tabs.addTab(self._ocr_text, "OCR 텍스트")

        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        self._type_lbl = QLabel("")
        self._type_lbl.setFont(QFont("", 13, QFont.Weight.Bold))
        result_layout.addWidget(self._type_lbl)
        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        self._result_text.setFont(QFont("", 12))
        result_layout.addWidget(self._result_text)
        self._tabs.addTab(result_widget, "추출 결과")

        hist_widget = QWidget()
        hist_layout = QVBoxLayout(hist_widget)
        self._hist_table = QTableWidget(0, 5)
        self._hist_table.setHorizontalHeaderLabels(
            ["ID", "파일", "유형", "신뢰도", "처리시각"]
        )
        self._hist_table.horizontalHeader().setStretchLastSection(True)
        hist_layout.addWidget(self._hist_table)
        refresh_btn = QPushButton("새로고침")
        refresh_btn.clicked.connect(self._refresh_history)
        hist_layout.addWidget(refresh_btn)
        self._tabs.addTab(hist_widget, "처리 이력")

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._refresh_history()

    # ── 초기화 ──

    def _on_warmup_done(self, ok: bool, msg: str) -> None:
        self._stage_lbl.setText(msg)
        self._status.showMessage(msg)

    def _update_ollama_status(self) -> None:
        ok = is_ollama_available()
        suffix = "연결됨" if ok else "미연결 (정규식 fallback)"
        self._status.showMessage(f"Ollama: {suffix}  |  온프레미스 · 데이터 외부 전송 없음")

    # ── 파일 관리 ──

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "이미지 선택", "",
            "이미지 (*.jpg *.jpeg *.png *.bmp *.webp *.tiff);;모든 파일 (*)",
        )
        for p in paths:
            path = Path(p)
            if path not in self._files:
                self._files.append(path)
                self._file_list.addItem(path.name)

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if not folder:
            return
        added = 0
        for p in sorted(Path(folder).iterdir()):
            if p.suffix.lower() in IMAGE_EXTS and p not in self._files:
                self._files.append(p)
                self._file_list.addItem(p.name)
                added += 1
        if added == 0:
            QMessageBox.information(self, "알림", "폴더에 추가할 이미지가 없습니다.")

    def _remove_selected(self) -> None:
        for item in self._file_list.selectedItems():
            idx = self._file_list.row(item)
            self._file_list.takeItem(idx)
            del self._files[idx]

    def _on_select_file(self, row: int) -> None:
        if row < 0 or row >= len(self._files):
            return
        path = self._files[row]
        pix = QPixmap(str(path))
        if not pix.isNull():
            scaled = pix.scaled(
                480, 480,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._preview_lbl.setPixmap(scaled)
        else:
            self._preview_lbl.setText(f"미리보기 불가: {path.name}")

        for r in self._results:
            if r.source_file == path.name:
                self._show_result(r)
                break

    # ── 처리 ──

    def _start_processing(self) -> None:
        if self._process_thread and self._process_thread.isRunning():
            return
        if not self._files:
            QMessageBox.warning(self, "알림", "처리할 이미지를 먼저 추가하세요.")
            return

        self._results.clear()
        self._run_btn.setEnabled(False)
        self._progress.setValue(0)

        self._process_thread = ProcessThread(list(self._files))
        self._process_thread.progress.connect(self._on_progress)
        self._process_thread.result_ready.connect(self._show_result)
        self._process_thread.excel_saved.connect(self._on_excel_saved)
        self._process_thread.error.connect(self._on_error)
        self._process_thread.finished_all.connect(self._on_finished)
        self._process_thread.start()

    def _on_progress(self, stage: str, pct: float) -> None:
        self._progress.setValue(int(pct))
        self._stage_lbl.setText(f"{stage}  ({pct:.0f}%)")

    def _on_excel_saved(self, path: str) -> None:
        self._stage_lbl.setText(f"완료 — Excel 저장: {Path(path).name}")

    def _on_error(self, name: str, msg: str) -> None:
        QMessageBox.critical(self, "오류", f"{name}\n{msg}")

    def _on_finished(self) -> None:
        self._run_btn.setEnabled(True)
        self._progress.setValue(100)
        self._refresh_history()
        if self._results:
            self._tabs.setCurrentIndex(2)

    # ── 결과 표시 ──

    def _show_result(self, result: DocumentResult) -> None:
        if result not in self._results:
            self._results.append(result)

        self._ocr_text.setPlainText(result.ocr.full_text or "(추출된 텍스트 없음)")

        doc = result.classification.doc_type
        self._type_lbl.setText(
            f"{DOC_LABEL[doc]}  ·  신뢰도 {result.classification.confidence:.0%}"
            f"  ({result.classification.source})"
        )

        structured = result.structured_dict()
        if not structured:
            self._result_text.setPlainText("구조화된 필드가 없습니다.")
            return

        lines: list[str] = []
        if result.warnings:
            lines.extend(f"⚠ {w}" for w in result.warnings)
            lines.append("")

        for key, val in structured.items():
            label = FIELD_LABELS.get(key, key)
            if key == "items" and isinstance(val, list):
                lines.append(f"【{label}】")
                for i, item in enumerate(val, 1):
                    if isinstance(item, dict):
                        parts = [
                            f"{FIELD_LABELS.get(k, k)}: {v}"
                            for k, v in item.items() if v
                        ]
                        lines.append(f"  {i}. " + ", ".join(parts))
            elif val not in (None, "", []):
                lines.append(f"{label}: {val}")

        self._result_text.setPlainText("\n".join(lines))

    # ──보내기 / 이력 ──

    def _export_excel(self) -> None:
        if not self._results:
            QMessageBox.information(self, "알림", "보낼 결과가 없습니다. 먼저 이미지를 처리하세요.")
            return
        try:
            path = export_to_excel(self._results)
            QMessageBox.information(self, "완료", f"Excel 저장 완료:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "오류", str(exc))

    def _open_output_dir(self) -> None:
        out = resolve_path(load_config().get("paths", {}).get("output_dir", "data/output"))
        out.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            subprocess.run(["open", str(out)], check=False)
        elif sys.platform == "win32":
            subprocess.run(["explorer", str(out)], check=False)
        else:
            subprocess.run(["xdg-open", str(out)], check=False)

    def _refresh_history(self) -> None:
        rows = db.fetch_all()
        self._hist_table.setRowCount(len(rows))
        type_map = {"receipt": "영수증", "business_card": "명함", "unknown": "미상"}
        for i, row in enumerate(rows):
            conf = row["confidence"]
            self._hist_table.setItem(i, 0, QTableWidgetItem(str(row["id"])))
            self._hist_table.setItem(i, 1, QTableWidgetItem(row["source_file"]))
            self._hist_table.setItem(
                i, 2, QTableWidgetItem(type_map.get(row["doc_type"], row["doc_type"]))
            )
            self._hist_table.setItem(
                i, 3, QTableWidgetItem(f"{conf:.0%}" if conf else "-")
            )
            self._hist_table.setItem(
                i, 4,
                QTableWidgetItem(row["processed_at"][:19].replace("T", " ")),
            )


def main() -> None:
    # urllib3 LibreSSL 경고 억제
    import warnings
    warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = DocumentAgentApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
