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


def _deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """텍스트 영역의 기울기를 추정해 수평으로 회전한다."""
    # 어두운 텍스트를 흰색으로 만들어 전경 좌표 추출
    inverted = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 50:  # 전경이 거의 없으면 보정 생략
        return gray, 0.0

    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect 각도 정규화 (-45 ~ 45 범위로)
    if angle < -45:
        angle = 90 + angle
    # 45도 초과 회전은 잘못된 추정(세로 문서 오인) → 보정 생략
    if abs(angle) > 15:
        return gray, 0.0
    if abs(angle) < 0.5:
        return gray, 0.0

    (h, w) = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        gray, matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, float(angle)


def _enhance(gray: np.ndarray) -> np.ndarray:
    """대비 향상(CLAHE) + 디노이즈."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    return denoised


def preprocess_image(
    image: str | Path | np.ndarray,
    *,
    do_deskew: bool = True,
    max_side: int = 1024,
) -> PreprocessResult:
    """이미지를 OCR 친화적으로 보정한다.

    Args:
        image: 파일 경로 또는 BGR ndarray.
        do_deskew: 기울기 보정 수행 여부.
        max_side: 긴 변 기준 리사이즈 상한(과도한 해상도로 인한 지연 방지).
    """
    original = _read_image(image)
    debug: dict = {}

    # 1) 과대 해상도 축소
    h, w = original.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    work = (
        cv2.resize(original, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else original.copy()
    )
    debug["resize_scale"] = round(scale, 3)

    # 2) 그레이스케일
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    # 3) 기울기 보정
    angle = 0.0
    if do_deskew:
        gray, angle = _deskew(gray)
    debug["deskew_angle"] = round(angle, 2)

    # 4) 대비/노이즈 보정
    enhanced = _enhance(gray)

    # PaddleOCR은 3채널 입력을 기대하므로 BGR로 복원
    processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    return PreprocessResult(original=original, processed=processed, debug=debug)
