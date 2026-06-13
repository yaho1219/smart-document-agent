#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.extraction.ollama_extractor import is_ollama_available
from src.ocr import warmup_ocr
from src.pipeline.agent import DocumentAgent
from src.schemas import DocumentResult, DocumentType
from src.storage import db
from src.storage.excel_exporter import export_to_excel

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def _print_bar(stage: str, pct: float) -> None:
    filled = int(pct * 40)
    bar = "█" * filled + "░" * (40 - filled)
    print(f"\r  [{bar}] {pct * 100:5.1f}%  {stage:<28}", end="", flush=True)
    if pct >= 1.0:
        print()


def _doc_type_label(doc_type: DocumentType) -> str:
    return {
        DocumentType.RECEIPT: "영수증",
        DocumentType.BUSINESS_CARD: "명함",
        DocumentType.UNKNOWN: "미상",
    }[doc_type]


def _print_result(result: DocumentResult) -> None:
    print()
    print("=" * 60)
    print(f"파일      : {result.source_file}")
    print(
        f"문서 유형 : {_doc_type_label(result.classification.doc_type)} "
        f"(신뢰도 {result.classification.confidence}, "
        f"방식: {result.classification.source})"
    )
    print("-" * 60)

    if result.warnings:
        for w in result.warnings:
            print(f"  ⚠ {w}")

    print("\n[OCR 추출 텍스트]")
    print(result.ocr.full_text or "(없음)")

    structured = result.structured_dict()
    if structured:
        print("\n[구조화 결과 (JSON)]")
        print(json.dumps(structured, ensure_ascii=False, indent=2))
    else:
        print("\n[구조화 결과] 없음")

    print("=" * 60)
    print()


def _collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTS else []
    if path.is_dir():
        files = sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        return files
    return []


def process_paths(
    paths: list[Path],
    *,
    export: bool = True,
    verbose: bool = True,
) -> list[DocumentResult]:
    if not paths:
        print("처리할 이미지가 없습니다.")
        return []

    if verbose:
        print("\n[시작] OCR 엔진 사전 로딩...")
        warmup_ocr(verbose=True)
        ollama_ok = is_ollama_available()
        print(f"[상태] Ollama: {'연결됨' if ollama_ok else '미연결 (정규식 fallback)'}\n")

    agent = DocumentAgent(persist=True)
    results: list[DocumentResult] = []

    for i, img_path in enumerate(paths, 1):
        print(f"\n>>> [{i}/{len(paths)}] {img_path.name} 처리 중...")

        def progress(stage: str, pct: float, _i=i, _n=len(paths)) -> None:
            overall = ((_i - 1) + pct) / _n
            _print_bar(f"{img_path.name}: {stage}", overall)

        result, _ = agent.process_image(
            img_path,
            progress=progress,
            source_name=img_path.name,
            verbose=verbose,
        )
        results.append(result)
        _print_result(result)

    if export and results:
        out = export_to_excel(results)
        print(f"[완료] Excel 저장: {out}")

    return results


def show_history() -> None:
    rows = db.fetch_all()
    if not rows:
        print("저장된 이력이 없습니다.")
        return
    print(f"\n{'ID':>4}  {'파일':<24}  {'유형':<14}  {'신뢰도':>6}  처리시각")
    print("-" * 72)
    for r in rows:
        print(
            f"{r['id']:>4}  {r['source_file']:<24}  "
            f"{r['doc_type']:<14}  {r['confidence']:>6.2f}  {r['processed_at']}"
        )
    print()


def interactive_menu() -> None:
    print()
    print("=" * 60)
    print("  온프레미스 스마트 명함·영수증 자동 정리 에이전트")
    print("  OpenCV → PaddleOCR → LayoutLM → Ollama → Excel/DB")
    print("=" * 60)
    print()
    print("  1) 이미지 파일 경로 입력하여 처리")
    print("  2) 폴더 경로 입력하여 일괄 처리")
    print("  3) 저장 이력 조회")
    print("  4) 종료")
    print()

    while True:
        choice = input("메뉴 선택 (1-4): ").strip()
        if choice == "1":
            raw = input("이미지 경로: ").strip().strip("'\"")
            p = Path(raw).expanduser()
            if not p.exists():
                print(f"파일을 찾을 수 없습니다: {p}")
                continue
            process_paths([p])
        elif choice == "2":
            raw = input("폴더 경로: ").strip().strip("'\"")
            p = Path(raw).expanduser()
            imgs = _collect_images(p)
            if not imgs:
                print(f"이미지가 없습니다: {p}")
                continue
            print(f"{len(imgs)}개 이미지 발견")
            process_paths(imgs)
        elif choice == "3":
            show_history()
        elif choice == "4":
            print("종료합니다.")
            break
        else:
            print("1~4 중에서 선택하세요.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="온프레미스 명함/영수증 자동 정리 에이전트",
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="처리할 이미지 파일 경로",
    )
    parser.add_argument(
        "-f", "--folder",
        metavar="DIR",
        help="폴더 내 이미지 일괄 처리",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="DB 저장 이력 조회",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Excel 자동 저장 생략",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="진행 메시지 최소화",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.history:
        show_history()
        return 0

    paths: list[Path] = []
    for raw in args.images:
        p = Path(raw).expanduser()
        paths.extend(_collect_images(p))

    if args.folder:
        folder = Path(args.folder).expanduser()
        paths.extend(_collect_images(folder))

    if paths:
        process_paths(
            paths,
            export=not args.no_export,
            verbose=not args.quiet,
        )
        return 0

    interactive_menu()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
