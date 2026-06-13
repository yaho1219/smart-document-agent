def preprocess_image(image, *, do_deskew=True, max_side=1920):
    original = _read_image(image)
    work = resize_if_needed(original, max_side)

    if do_deskew:                              # ① 기울기 보정
        angle = _estimate_skew(gray)
        work = cv2.warpAffine(work, matrix, ...)

    processed = _enhance_color(work)           # ② CLAHE + 샤프닝
    return PreprocessResult(original, processed)