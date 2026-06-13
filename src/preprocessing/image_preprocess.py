from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    original: np.ndarray
    processed: np.ndarray
    debug: dict = field(default_factory=dict)


def _read_image(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"이미지 디코딩 실패: {path}")
    return img


def _estimate_skew(gray: np.ndarray) -> float:
    inverted = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 50:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > 15 or abs(angle) < 0.5:
        return 0.0
    return float(angle)


def _enhance_color(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    enhanced = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(enhanced, 1.3, blur, -0.3, 0)
    return sharpened


def preprocess_image(
    image: str | Path | np.ndarray,
    *,
    do_deskew: bool = True,
    max_side: int = 1920,
) -> PreprocessResult:
    original = _read_image(image)
    debug: dict = {}

    h, w = original.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    work = (
        cv2.resize(original, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else original.copy()
    )
    debug["resize_scale"] = round(scale, 3)

    angle = 0.0
    if do_deskew:
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        angle = _estimate_skew(gray)
        if angle != 0.0:
            (wh, ww) = work.shape[:2]
            matrix = cv2.getRotationMatrix2D((ww / 2, wh / 2), angle, 1.0)
            work = cv2.warpAffine(
                work, matrix, (ww, wh),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
            )
    debug["deskew_angle"] = round(angle, 2)

    processed = _enhance_color(work)

    return PreprocessResult(original=original, processed=processed, debug=debug)
