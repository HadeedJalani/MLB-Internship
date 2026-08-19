# ==========================================================
# MLB Summer Internship - Day 23
# OCR Pipeline
# Image Preprocessing Module
# ==========================================================

import cv2
import numpy as np


def ensure_bgr(image):
    """
    Ensure the image is represented as a 3-channel BGR image.
    """

    if image is None:
        raise ValueError("Input image is empty.")

    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image.copy()


def resize_for_ocr(image, max_dimension=1800):
    """
    Resize very large images while preserving aspect ratio.

    Large images can significantly increase OCR processing time.
    """

    image = ensure_bgr(image)

    height, width = image.shape[:2]
    largest_dimension = max(height, width)

    if largest_dimension <= max_dimension:
        return image

    scale = max_dimension / largest_dimension

    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def convert_to_grayscale(image):
    """
    Convert a BGR image to grayscale.
    """

    image = ensure_bgr(image)

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


def denoise_image(image, strength=5):
    """
    Reduce small amounts of image noise using Gaussian blur.

    The operation is intentionally mild so that character
    boundaries are not destroyed before OCR.
    """

    if image is None:
        raise ValueError("Input image is empty.")

    kernel = max(3, int(strength))

    if kernel % 2 == 0:
        kernel += 1

    return cv2.GaussianBlur(
        image,
        (kernel, kernel),
        0,
    )


def enhance_contrast(image):
    """
    Improve grayscale contrast using CLAHE.

    CLAHE is useful for documents with uneven lighting
    or relatively weak foreground/background separation.
    """

    if image is None:
        raise ValueError("Input image is empty.")

    if len(image.shape) == 3:
        gray = convert_to_grayscale(image)
    else:
        gray = image.copy()

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(gray)


def adaptive_threshold(image):
    """
    Apply adaptive thresholding for document-style images.

    This can be useful when lighting is not uniform across
    the document.
    """

    if image is None:
        raise ValueError("Input image is empty.")

    if len(image.shape) == 3:
        gray = convert_to_grayscale(image)
    else:
        gray = image.copy()

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )


def otsu_threshold(image):
    """
    Apply Otsu's automatic thresholding.
    """

    if image is None:
        raise ValueError("Input image is empty.")

    if len(image.shape) == 3:
        gray = convert_to_grayscale(image)
    else:
        gray = image.copy()

    _, thresholded = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return thresholded


def preprocess_image(
    image,
    method="enhanced",
    resize=True,
):
    """
    Main preprocessing pipeline.

    Available methods:

    - original
    - grayscale
    - denoise
    - enhanced
    - adaptive_threshold
    - otsu_threshold
    """

    if image is None:
        raise ValueError("Input image is empty.")

    processed = ensure_bgr(image)

    if resize:
        processed = resize_for_ocr(processed)

    if method == "original":
        return processed

    if method == "grayscale":
        return convert_to_grayscale(processed)

    if method == "denoise":
        gray = convert_to_grayscale(processed)
        return denoise_image(gray)

    if method == "enhanced":
        return enhance_contrast(processed)

    if method == "adaptive_threshold":
        enhanced = enhance_contrast(processed)
        return adaptive_threshold(enhanced)

    if method == "otsu_threshold":
        enhanced = enhance_contrast(processed)
        return otsu_threshold(enhanced)

    raise ValueError(
        f"Unsupported preprocessing method: {method}"
    )


def prepare_for_ocr(image, method="enhanced"):
    """
    Prepare an image specifically for OCR.

    Returns a format that can be safely passed to EasyOCR.
    """

    processed = preprocess_image(
        image,
        method=method,
        resize=True,
    )

    return processed