"""사전학습 모델 가중치 다운로드.

LayoutLM base 모델을 미리 받아 로컬 캐시에 저장한다(오프라인 데모 대비).
PaddleOCR 모델은 최초 실행 시 자동 다운로드되므로 여기서는 LayoutLM만 다룬다.

사용:
    python scripts/download_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 프로젝트 루트를 import 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402


def main() -> None:
    cfg = load_config().get("classification", {})
    model_name = cfg.get("model_name", "microsoft/layoutlm-base-uncased")

    print(f"[download] LayoutLM 토크나이저/모델 캐싱: {model_name}")
    try:
        from transformers import LayoutLMModel, LayoutLMTokenizer

        LayoutLMTokenizer.from_pretrained(model_name)
        LayoutLMModel.from_pretrained(model_name)
        print("[download] 완료. (HuggingFace 캐시에 저장됨)")
    except Exception as exc:
        print(f"[download] 실패: {exc}")
        print("네트워크 또는 transformers 설치 상태를 확인하세요.")
        sys.exit(1)


if __name__ == "__main__":
    main()
