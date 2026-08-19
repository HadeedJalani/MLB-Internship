# ==========================================================
# MLB Summer Internship - Day 23
# Dual OCR Pipeline
# EasyOCR + PaddleOCR
# ==========================================================

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


# ==========================================================
# Detection Format
# ==========================================================
#
# Every OCR engine is converted into this format:
#
# {
#     "box": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
#     "text": "Detected text",
#     "confidence": 0.95
# }
#
# This keeps the Streamlit application independent
# of the OCR library being used.
# ==========================================================


def normalize_detection(
    box,
    text,
    confidence,
) -> Dict[str, Any]:
    """
    Convert an OCR detection into the unified format.
    """

    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0

    confidence = max(
        0.0,
        min(1.0, confidence),
    )

    normalized_box = []

    try:

        for point in box:

            if len(point) >= 2:

                normalized_box.append(
                    [
                        int(float(point[0])),
                        int(float(point[1])),
                    ]
                )

    except Exception:

        normalized_box = []

    return {
        "box": normalized_box,
        "text": str(text).strip(),
        "confidence": confidence,
    }


# ==========================================================
# Image Validation
# ==========================================================


def validate_image(image):
    """
    Validate an OpenCV image before OCR processing.
    """

    if image is None:
        raise ValueError(
            "No image was provided."
        )

    if not isinstance(
        image,
        np.ndarray,
    ):
        raise TypeError(
            "Image must be a NumPy array."
        )

    if image.size == 0:
        raise ValueError(
            "The uploaded image is empty."
        )

    return image


# ==========================================================
# Image Preparation
# ==========================================================


def prepare_image(image):
    """
    Prepare an image for OCR.

    Handles grayscale, BGR and BGRA images.
    """

    image = validate_image(image)

    if len(image.shape) == 2:

        return image

    channels = image.shape[2]

    if channels == 4:

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2BGR,
        )

    if channels == 3:

        return image

    raise ValueError(
        "Unsupported image channel configuration."
    )


# ==========================================================
# Preprocessing
# ==========================================================


def preprocess_image(
    image,
    mode="enhanced",
):
    """
    Preprocess image before OCR.

    Modes:
        original
        grayscale
        enhanced
        threshold
        denoise
    """

    image = prepare_image(image)

    if mode == "original":

        return image.copy()

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    if mode == "grayscale":

        return gray

    if mode == "enhanced":

        # Local contrast enhancement
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        enhanced = clahe.apply(
            gray
        )

        # Light denoising
        enhanced = cv2.GaussianBlur(
            enhanced,
            (3, 3),
            0,
        )

        return enhanced

    if mode == "threshold":

        enhanced = cv2.GaussianBlur(
            gray,
            (3, 3),
            0,
        )

        thresholded = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

        return thresholded

    if mode == "denoise":

        return cv2.fastNlMeansDenoising(
            gray,
            None,
            10,
            7,
            21,
        )

    raise ValueError(
        f"Unknown preprocessing mode: {mode}"
    )


# ==========================================================
# EasyOCR
# ==========================================================


def create_easyocr_reader(
    languages=None,
    gpu=False,
):
    """
    Create an EasyOCR reader.

    The application should initialize this once and
    store it in Streamlit session state.
    """

    import easyocr

    if languages is None:
        languages = ["en"]

    return easyocr.Reader(
        languages,
        gpu=gpu,
        verbose=False,
    )


def run_easyocr(
    reader,
    image,
    min_confidence=0.0,
):
    """
    Run EasyOCR and convert results into the
    unified detection format.
    """

    image = prepare_image(image)

    start_time = time.perf_counter()

    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    detections = []

    for result in results:

        if not isinstance(
            result,
            (list, tuple),
        ):
            continue

        if len(result) < 3:
            continue

        box = result[0]
        text = result[1]
        confidence = result[2]

        detection = normalize_detection(
            box,
            text,
            confidence,
        )

        if (
            detection["confidence"]
            >= min_confidence
            and detection["text"]
        ):

            detections.append(
                detection
            )

    return {
        "engine": "EasyOCR",
        "detections": detections,
        "processing_time": elapsed,
        "text": get_text(
            detections
        ),
        "confidence":
            calculate_average_confidence(
                detections
            ),
    }


# ==========================================================
# PaddleOCR
# ==========================================================


def create_paddleocr_reader(
    language="en",
):
    """
    Create a PaddleOCR reader.

    PaddleOCR has changed its constructor/API across
    versions. This function attempts modern initialization
    first and then falls back to a compatible configuration.
    """

    from paddleocr import PaddleOCR

    errors = []

    # ------------------------------------------------------
    # Modern PaddleOCR versions
    # ------------------------------------------------------

    configurations = [
        {
            "lang": language,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": True,
        },
        {
            "lang": language,
        },
    ]

    for configuration in configurations:

        try:

            return PaddleOCR(
                **configuration
            )

        except Exception as error:

            errors.append(
                str(error)
            )

    # ------------------------------------------------------
    # Older versions
    # ------------------------------------------------------

    legacy_configurations = [
        {
            "lang": language,
            "use_angle_cls": True,
        },
        {
            "lang": language,
        },
    ]

    for configuration in legacy_configurations:

        try:

            return PaddleOCR(
                **configuration
            )

        except Exception as error:

            errors.append(
                str(error)
            )

    raise RuntimeError(
        "Unable to initialize PaddleOCR. "
        "Your installed PaddleOCR version may require "
        "different initialization parameters.\n\n"
        + "\n".join(errors[-3:])
    )


def _extract_value(
    item,
    key,
    default=None,
):
    """
    Safely retrieve values from dictionaries or objects.
    """

    if isinstance(
        item,
        dict,
    ):

        return item.get(
            key,
            default,
        )

    return getattr(
        item,
        key,
        default,
    )


def _parse_paddle_detection(
    box,
    text,
    confidence,
):
    """
    Convert a PaddleOCR detection into our unified format.
    """

    return normalize_detection(
        box=box,
        text=text,
        confidence=confidence,
    )


def _parse_paddle_result_object(
    result,
):
    """
    Attempt to parse PaddleOCR's newer result objects.

    New PaddleOCR versions can expose structured results
    through dictionaries / result objects rather than the
    older nested-list format.
    """

    detections = []

    # ------------------------------------------------------
    # Dictionary-like result
    # ------------------------------------------------------

    if isinstance(
        result,
        dict,
    ):

        boxes = (
            result.get("rec_polys")
            or result.get("dt_polys")
            or result.get("text_regions")
            or result.get("boxes")
        )

        texts = (
            result.get("rec_texts")
            or result.get("texts")
        )

        scores = (
            result.get("rec_scores")
            or result.get("scores")
            or result.get("confidences")
        )

        if (
            boxes is not None
            and texts is not None
        ):

            if scores is None:

                scores = [
                    1.0
                    for _ in texts
                ]

            for box, text, score in zip(
                boxes,
                texts,
                scores,
            ):

                detections.append(
                    _parse_paddle_detection(
                        box,
                        text,
                        score,
                    )
                )

            return detections

    # ------------------------------------------------------
    # Object attributes
    # ------------------------------------------------------

    boxes = _extract_value(
        result,
        "rec_polys",
    )

    if boxes is None:

        boxes = _extract_value(
            result,
            "dt_polys",
        )

    texts = _extract_value(
        result,
        "rec_texts",
    )

    scores = _extract_value(
        result,
        "rec_scores",
    )

    if (
        boxes is not None
        and texts is not None
    ):

        if scores is None:

            scores = [
                1.0
                for _ in texts
            ]

        for box, text, score in zip(
            boxes,
            texts,
            scores,
        ):

            detections.append(
                _parse_paddle_detection(
                    box,
                    text,
                    score,
                )
            )

    return detections


def run_paddleocr(
    reader,
    image,
    min_confidence=0.0,
):
    """
    Run PaddleOCR with compatibility handling for
    multiple PaddleOCR API generations.
    """

    image = prepare_image(image)

    start_time = time.perf_counter()

    errors = []

    results = None

    # ------------------------------------------------------
    # New API
    # ------------------------------------------------------

    try:

        results = reader.predict(
            image
        )

    except Exception as error:

        errors.append(
            f"predict(): {error}"
        )

    # ------------------------------------------------------
    # Older API
    # ------------------------------------------------------

    if results is None:

        try:

            results = reader.ocr(
                image
            )

        except Exception as error:

            errors.append(
                f"ocr(): {error}"
            )

    if results is None:

        raise RuntimeError(
            "PaddleOCR failed to process the image.\n"
            + "\n".join(errors)
        )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    detections = []

    # ------------------------------------------------------
    # Parse results
    # ------------------------------------------------------

    if isinstance(
        results,
        (list, tuple),
    ):

        for result in results:

            # New structured result
            parsed = (
                _parse_paddle_result_object(
                    result
                )
            )

            if parsed:

                detections.extend(
                    parsed
                )

                continue

            # --------------------------------------------------
            # Older PaddleOCR format
            #
            # [
            #   [
            #      [box, (text, confidence)],
            #      ...
            #   ]
            # ]
            # --------------------------------------------------

            if isinstance(
                result,
                (list, tuple),
            ):

                for item in result:

                    if not isinstance(
                        item,
                        (list, tuple),
                    ):
                        continue

                    if len(item) < 2:
                        continue

                    box = item[0]
                    recognition = item[1]

                    if isinstance(
                        recognition,
                        (list, tuple),
                    ):

                        if len(
                            recognition
                        ) >= 2:

                            text = (
                                recognition[0]
                            )

                            confidence = (
                                recognition[1]
                            )

                            detections.append(
                                _parse_paddle_detection(
                                    box,
                                    text,
                                    confidence,
                                )
                            )

    # ------------------------------------------------------
    # Final filtering
    # ------------------------------------------------------

    detections = [
        detection
        for detection in detections
        if (
            detection["text"]
            and detection["confidence"]
            >= min_confidence
        )
    ]

    return {
        "engine": "PaddleOCR",
        "detections": detections,
        "processing_time": elapsed,
        "text": get_text(
            detections
        ),
        "confidence":
            calculate_average_confidence(
                detections
            ),
    }


# ==========================================================
# Unified OCR Interface
# ==========================================================


def run_ocr(
    engine,
    image,
    easy_reader=None,
    paddle_reader=None,
    min_confidence=0.0,
):
    """
    Unified OCR interface.

    engine:
        "EasyOCR"
        "PaddleOCR"
        "Compare Both"
    """

    if engine == "EasyOCR":

        if easy_reader is None:

            raise ValueError(
                "EasyOCR reader is not initialized."
            )

        return {
            "easyocr": run_easyocr(
                reader=easy_reader,
                image=image,
                min_confidence=min_confidence,
            )
        }

    if engine == "PaddleOCR":

        if paddle_reader is None:

            raise ValueError(
                "PaddleOCR reader is not initialized."
            )

        return {
            "paddleocr": run_paddleocr(
                reader=paddle_reader,
                image=image,
                min_confidence=min_confidence,
            )
        }

    if engine == "Compare Both":

        if (
            easy_reader is None
            or paddle_reader is None
        ):

            raise ValueError(
                "Both OCR engines must be initialized "
                "for comparison mode."
            )

        easy_result = run_easyocr(
            reader=easy_reader,
            image=image,
            min_confidence=min_confidence,
        )

        paddle_result = run_paddleocr(
            reader=paddle_reader,
            image=image,
            min_confidence=min_confidence,
        )

        return {
            "easyocr": easy_result,
            "paddleocr": paddle_result,
        }

    raise ValueError(
        f"Unsupported OCR engine: {engine}"
    )


# ==========================================================
# Text Utilities
# ==========================================================


def sort_detections(
    detections,
):
    """
    Sort detections top-to-bottom and left-to-right.
    """

    def key(detection):

        box = detection.get(
            "box",
            [],
        )

        if not box:
            return (
                0,
                0,
            )

        try:

            xs = [
                point[0]
                for point in box
            ]

            ys = [
                point[1]
                for point in box
            ]

            return (
                min(ys),
                min(xs),
            )

        except Exception:

            return (
                0,
                0,
            )

    return sorted(
        detections,
        key=key,
    )


def get_text(
    detections,
):
    """
    Convert detections into readable text.
    """

    if not detections:
        return ""

    detections = sort_detections(
        detections
    )

    lines = []

    for detection in detections:

        text = str(
            detection.get(
                "text",
                "",
            )
        ).strip()

        if text:

            lines.append(
                text
            )

    return "\n".join(
        lines
    )


def calculate_average_confidence(
    detections,
):
    """
    Calculate mean OCR confidence.
    """

    if not detections:
        return 0.0

    values = []

    for detection in detections:

        try:

            values.append(
                float(
                    detection.get(
                        "confidence",
                        0.0,
                    )
                )
            )

        except Exception:
            pass

    if not values:
        return 0.0

    return sum(values) / len(
        values
    )


# ==========================================================
# Visualization
# ==========================================================


def draw_detections(
    image,
    detections,
):
    """
    Draw OCR bounding boxes.
    """

    image = prepare_image(
        image
    )

    output = image.copy()

    for detection in detections:

        box = detection.get(
            "box",
            [],
        )

        text = detection.get(
            "text",
            "",
        )

        confidence = float(
            detection.get(
                "confidence",
                0.0,
            )
        )

        if len(box) < 4:
            continue

        try:

            points = np.array(
                box,
                dtype=np.int32,
            )

            cv2.polylines(
                output,
                [points],
                True,
                (0, 200, 100),
                2,
            )

            x = int(
                min(
                    point[0]
                    for point in box
                )
            )

            y = int(
                min(
                    point[1]
                    for point in box
                )
            )

            label = (
                f"{text[:25]} "
                f"{confidence * 100:.0f}%"
            )

            cv2.putText(
                output,
                label,
                (
                    x,
                    max(20, y - 7),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 100),
                1,
                cv2.LINE_AA,
            )

        except Exception:
            continue

    return output


# ==========================================================
# Summary
# ==========================================================


def get_summary(
    result,
):
    """
    Return UI-friendly OCR statistics.
    """

    detections = result.get(
        "detections",
        [],
    )

    text = result.get(
        "text",
        "",
    )

    return {
        "engine": result.get(
            "engine",
            "Unknown",
        ),
        "regions": len(
            detections
        ),
        "confidence":
            calculate_average_confidence(
                detections
            ),
        "words": len(
            text.split()
        ) if text.strip() else 0,
        "characters": len(
            text.replace(
                "\n",
                " ",
            )
        ),
        "processing_time":
            result.get(
                "processing_time",
                0.0,
            ),
    }