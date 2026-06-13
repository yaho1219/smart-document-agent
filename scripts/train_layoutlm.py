from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import ensure_dir, load_config, resolve_path  # noqa: E402
from src.ocr import run_ocr  # noqa: E402
from src.preprocessing.image_preprocess import preprocess_image  # noqa: E402


def _normalize_bbox(bbox: list[int], w: int, h: int) -> list[int]:
    if not bbox or w == 0 or h == 0:
        return [0, 0, 0, 0]
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(1000, int(1000 * x0 / w))),
        max(0, min(1000, int(1000 * y0 / h))),
        max(0, min(1000, int(1000 * x1 / w))),
        max(0, min(1000, int(1000 * y1 / h))),
    ]


def main() -> None:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        LayoutLMForSequenceClassification,
        LayoutLMTokenizer,
    )

    paths_cfg = load_config().get("paths", {})
    cls_cfg = load_config().get("classification", {})
    labels: list[str] = cls_cfg.get("labels", ["receipt", "business_card", "unknown"])
    label2id = {name: i for i, name in enumerate(labels)}

    samples_dir = resolve_path(paths_cfg.get("samples_dir", "data/samples"))
    csv_path = samples_dir / "labels.csv"
    if not csv_path.exists():
        print(f"[train] 라벨 파일이 없습니다: {csv_path}")
        print("data/samples/labels.csv (filename,label)을 준비하세요.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    model_name = cls_cfg.get("model_name", "microsoft/layoutlm-base-uncased")
    tokenizer = LayoutLMTokenizer.from_pretrained(model_name)

    encoded_samples = []
    for _, row in df.iterrows():
        img_path = samples_dir / str(row["filename"])
        if not img_path.exists():
            print(f"[train] 건너뜀(파일 없음): {img_path}")
            continue
        pre = preprocess_image(img_path)
        ocr = run_ocr(pre.processed)
        if not ocr.words:
            print(f"[train] 건너뜀(OCR 비어있음): {img_path}")
            continue

        max_x = max((w.bbox[2] for w in ocr.words if w.bbox), default=1)
        max_y = max((w.bbox[3] for w in ocr.words if w.bbox), default=1)
        words = [w.text for w in ocr.words]
        boxes = [_normalize_bbox(w.bbox, max_x, max_y) for w in ocr.words]

        enc = tokenizer(
            words, is_split_into_words=True, truncation=True,
            padding="max_length", max_length=512, return_tensors="pt",
        )
        word_ids = enc.word_ids(0)
        token_boxes = [
            boxes[wid] if wid is not None else [0, 0, 0, 0] for wid in word_ids
        ]
        encoded_samples.append(
            {
                "input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "token_type_ids": enc["token_type_ids"][0],
                "bbox": torch.tensor(token_boxes),
                "label": label2id.get(str(row["label"]), label2id["unknown"]),
            }
        )

    if not encoded_samples:
        print("[train] 학습 가능한 샘플이 없습니다. 종료합니다.")
        sys.exit(1)

    class DocDataset(Dataset):
        def __len__(self):
            return len(encoded_samples)

        def __getitem__(self, idx):
            return encoded_samples[idx]

    loader = DataLoader(DocDataset(), batch_size=2, shuffle=True)

    model = LayoutLMForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()

    epochs = 3
    for epoch in range(epochs):
        total = 0.0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                bbox=batch["bbox"],
                labels=batch["label"],
            )
            outputs.loss.backward()
            optimizer.step()
            total += float(outputs.loss)
        print(f"[train] epoch {epoch + 1}/{epochs} loss={total / len(loader):.4f}")

    out_dir = ensure_dir(paths_cfg.get("layoutlm_weights", "models/layoutlm_classifier"))
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[train] 모델 저장 완료: {out_dir}")


if __name__ == "__main__":
    main()
