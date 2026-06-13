from __future__ import annotations

import sys
from pathlib import Path

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
