"""OpenCV 기반 이미지 전처리.

명함/영수증 사진은 조명, 기울기, 원근 왜곡이 흔하므로 OCR 정확도를
높이기 위한 보정을 수행한다. Streamlit에서 전/후 비교가 가능하도록
원본과 처리본, 디버그 정보를 함께 반환한다.

담당: CV Engineer
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    original: np.ndarray          # BGR 원본
    processed: np.ndarray         # OCR 입력용 (BGR로 통일)
    debug: dict = field(default_factory=dict)


def _read_image(image: str | Path | np.ndarray) -> np.ndarray:
    """경로 또는 ndarray를 BGR 이미지로 로드한다."""
    if isinstance(image, np.ndarray):
        return image
    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {path}")
    # 한글 경로 대응을 위해 np.fromfile 사용
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"이미지 디코딩 실패: {path}")
    return img


def _estimate_skew(gray: np.ndarray) -> float:
    """기울기 각도만 추정한다 (회전은 컬러 이미지에 적용)."""
    inverted = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 50:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    # 15도 초과는 잘못된 추정(세로 문서 오인) 가능성이 높아 보정 생략
    if abs(angle) > 15 or abs(angle) < 0.5:
        return 0.0
    return float(angle)


def _enhance_color(img: np.ndarray) -> np.ndarray:
    """컬러 유지 대비 향상(CLAHE on L) + 가벼운 샤프닝.

    과거 그레이스케일+강한 디노이즈는 한글 획을 뭉개 인식률을
    떨어뜨려, 컬러 기반의 완만한 보정으로 변경.
    """
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
    """이미지를 OCR 친화적으로 보정한다.

    Args:
        image: 파일 경로 또는 BGR ndarray.
        do_deskew: 기울기 보정 수행 여부.
        max_side: 긴 변 기준 리사이즈 상한(과도한 해상도로 인한 지연 방지).
    """
    original = _read_image(image)
    debug: dict = {}

    # 1) 과대 해상도 축소 (작은 글씨 보존을 위해 1920까지 허용)
    h, w = original.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    work = (
        cv2.resize(original, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else original.copy()
    )
    debug["resize_scale"] = round(scale, 3)

    # 2) 기울기 보정 (각도는 그레이로 추정, 회전은 컬러에 적용)
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

    # 3) 컬러 보정 (한글 획 보존)
    processed = _enhance_color(work)

    return PreprocessResult(original=original, processed=processed, debug=debug)
