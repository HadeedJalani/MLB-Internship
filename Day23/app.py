# ==========================================================
# MLB Summer Internship - Day 23
# Document OCR Studio
#
# Dual OCR Engine Application
# EasyOCR + PaddleOCR + OpenCV + Streamlit
#
# IMPORTANT IMPROVEMENTS
# ----------------------------------------------------------
# 1. OCR models are cached with st.cache_resource.
# 2. OCR inference is cached with st.cache_data.
# 3. EasyOCR results are NOT deleted when PaddleOCR runs.
# 4. PaddleOCR results are NOT deleted when EasyOCR runs.
# 5. Results remain available for comparison.
# 6. Results automatically reset when a new image is uploaded.
# 7. PaddleOCR failures do not affect EasyOCR.
# 8. EasyOCR bounding boxes are refined against original image.
# 9. PaddleOCR oneDNN/MKLDNN compatibility handling included.
# 10. Separate "Run Selected Engine" and "Run Both Engines".
# ==========================================================


# ==========================================================
# PADDLE COMPATIBILITY SETTINGS
#
# These MUST be configured before importing Paddle/PaddleOCR.
# ==========================================================

import os

os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["GLOG_v"] = "0"


# ==========================================================
# IMPORTS
# ==========================================================

import hashlib
import io
import re
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Document OCR Studio",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.8rem;
        border-radius: 18px;
        background:
            linear-gradient(
                135deg,
                #0f172a 0%,
                #1e293b 100%
            );
        border: 1px solid rgba(148,163,184,0.20);
    }

    .hero-title {
        color: #f8fafc;
        font-size: 2.45rem;
        font-weight: 750;
        letter-spacing: -0.8px;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 850px;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    .engine-card {
        padding: 1.1rem 1.25rem;
        border-radius: 13px;
        border: 1px solid rgba(128,128,128,0.22);
        background: rgba(128,128,128,0.045);
        margin-bottom: 1rem;
    }

    .engine-name {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .engine-description {
        font-size: 0.88rem;
        opacity: 0.72;
        line-height: 1.5;
    }

    .status-ok {
        padding: 0.7rem 0.9rem;
        border-radius: 9px;
        background: rgba(34,197,94,0.10);
        border: 1px solid rgba(34,197,94,0.25);
    }

    .status-warning {
        padding: 0.7rem 0.9rem;
        border-radius: 9px;
        background: rgba(245,158,11,0.10);
        border: 1px solid rgba(245,158,11,0.25);
    }

    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(128,128,128,0.18);
        text-align: center;
        opacity: 0.6;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# SESSION STATE
# ==========================================================

if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = {}

if "current_image_hash" not in st.session_state:
    st.session_state.current_image_hash = None

if "paddle_error" not in st.session_state:
    st.session_state.paddle_error = None


# ==========================================================
# BASIC HELPERS
# ==========================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def calculate_image_hash(image_bytes):

    return hashlib.sha256(
        image_bytes
    ).hexdigest()


# ==========================================================
# IMAGE LOADING
# ==========================================================

def load_image(uploaded_file):

    image_bytes = uploaded_file.getvalue()

    pil_image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    rgb = np.array(
        pil_image
    )

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    return image_bytes, rgb, bgr


# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def preprocess_image(
    image,
    grayscale=True,
    denoise=False,
    enhance=True,
    threshold=False,
):

    working = image.copy()

    if grayscale:

        gray = cv2.cvtColor(
            working,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = working

    if denoise:

        gray = cv2.fastNlMeansDenoising(
            gray,
            None,
            5,
            7,
            21
        )

    if enhance and len(gray.shape) == 2:

        clahe = cv2.createCLAHE(
            clipLimit=1.5,
            tileGridSize=(8, 8)
        )

        gray = clahe.apply(gray)

    if threshold and len(gray.shape) == 2:

        gray = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )

    if len(gray.shape) == 2:

        return cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR
        )

    return gray


# ==========================================================
# BOUNDING BOX NORMALIZATION
# ==========================================================

def normalize_bbox(bbox):

    if bbox is None:
        return None

    try:

        points = np.asarray(
            bbox,
            dtype=np.float32
        )

        # Polygon
        if (
            points.ndim == 2
            and points.shape[1] == 2
            and points.shape[0] >= 4
        ):

            points = points[:4]

            return [
                [
                    int(round(x)),
                    int(round(y))
                ]
                for x, y in points
            ]

        # [x1, y1, x2, y2]
        if (
            points.ndim == 1
            and len(points) == 4
        ):

            x1, y1, x2, y2 = points

            return [
                [int(round(x1)), int(round(y1))],
                [int(round(x2)), int(round(y1))],
                [int(round(x2)), int(round(y2))],
                [int(round(x1)), int(round(y2))]
            ]

    except Exception:

        return None

    return None


# ==========================================================
# EASY OCR MODEL
#
# This is executed only once per Streamlit process.
# ==========================================================

@st.cache_resource(
    show_spinner=False
)
def initialize_easyocr():

    import easyocr

    return easyocr.Reader(
        ["en"],
        gpu=False,
        verbose=False
    )


# ==========================================================
# EASY OCR BOX REFINEMENT
# ==========================================================

def refine_easy_bbox(
    bbox,
    image,
    search_margin=6
):

    if bbox is None:
        return None

    try:

        points = np.asarray(
            bbox,
            dtype=np.int32
        )

        if (
            points.ndim != 2
            or points.shape[1] != 2
        ):
            return bbox

        height, width = image.shape[:2]

        x_min = int(
            np.min(points[:, 0])
        )

        y_min = int(
            np.min(points[:, 1])
        )

        x_max = int(
            np.max(points[:, 0])
        )

        y_max = int(
            np.max(points[:, 1])
        )

        x_min = max(
            0,
            min(width - 1, x_min)
        )

        y_min = max(
            0,
            min(height - 1, y_min)
        )

        x_max = max(
            0,
            min(width - 1, x_max)
        )

        y_max = max(
            0,
            min(height - 1, y_max)
        )

        if (
            x_max <= x_min
            or y_max <= y_min
        ):
            return bbox

        # Search around original EasyOCR box.
        sx1 = max(
            0,
            x_min - search_margin
        )

        sy1 = max(
            0,
            y_min - search_margin
        )

        sx2 = min(
            width - 1,
            x_max + search_margin
        )

        sy2 = min(
            height - 1,
            y_max + search_margin
        )

        roi = image[
            sy1:sy2 + 1,
            sx1:sx2 + 1
        ]

        if roi.size == 0:
            return bbox

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        # Dark text -> white foreground.
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV
            + cv2.THRESH_OTSU
        )

        # Remove very small noise.
        kernel = np.ones(
            (2, 2),
            np.uint8
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel
        )

        # Original box in local ROI coordinates.
        bx1 = x_min - sx1
        by1 = y_min - sy1
        bx2 = x_max - sx1
        by2 = y_max - sy1

        local_margin = 4

        rx1 = max(
            0,
            bx1 - local_margin
        )

        ry1 = max(
            0,
            by1 - local_margin
        )

        rx2 = min(
            binary.shape[1] - 1,
            bx2 + local_margin
        )

        ry2 = min(
            binary.shape[0] - 1,
            by2 + local_margin
        )

        candidate = binary[
            ry1:ry2 + 1,
            rx1:rx2 + 1
        ]

        if candidate.size == 0:
            return bbox

        contours, _ = cv2.findContours(
            candidate,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if area < 1:
                continue

            cx, cy, cw, ch = cv2.boundingRect(
                contour
            )

            if cw < 1 or ch < 1:
                continue

            boxes.append(
                (
                    cx + rx1,
                    cy + ry1,
                    cw,
                    ch
                )
            )

        if not boxes:

            return [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ]

        union_x1 = min(
            box[0]
            for box in boxes
        )

        union_y1 = min(
            box[1]
            for box in boxes
        )

        union_x2 = max(
            box[0] + box[2]
            for box in boxes
        )

        union_y2 = max(
            box[1] + box[3]
            for box in boxes
        )

        final_x1 = sx1 + union_x1
        final_y1 = sy1 + union_y1
        final_x2 = sx1 + union_x2
        final_y2 = sy1 + union_y2

        # Prevent refinement from jumping to unrelated text.
        max_shift = max(
            10,
            int(
                max(
                    x_max - x_min,
                    y_max - y_min
                ) * 0.35
            )
        )

        if abs(final_x1 - x_min) > max_shift:
            final_x1 = x_min

        if abs(final_y1 - y_min) > max_shift:
            final_y1 = y_min

        if abs(final_x2 - x_max) > max_shift:
            final_x2 = x_max

        if abs(final_y2 - y_max) > max_shift:
            final_y2 = y_max

        final_x1 = max(
            0,
            min(width - 1, final_x1)
        )

        final_y1 = max(
            0,
            min(height - 1, final_y1)
        )

        final_x2 = max(
            0,
            min(width - 1, final_x2)
        )

        final_y2 = max(
            0,
            min(height - 1, final_y2)
        )

        if (
            final_x2 <= final_x1
            or final_y2 <= final_y1
        ):
            return [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ]

        return [
            [
                int(final_x1),
                int(final_y1)
            ],
            [
                int(final_x2),
                int(final_y1)
            ],
            [
                int(final_x2),
                int(final_y2)
            ],
            [
                int(final_x1),
                int(final_y2)
            ]
        ]

    except Exception:

        return bbox


# ==========================================================
# EASY OCR INFERENCE
#
# Cached separately from model initialization.
# Same image + same settings = no second inference.
# ==========================================================

@st.cache_data(
    show_spinner=False
)
def cached_easyocr(
    image_bytes,
    min_confidence,
    refine_boxes
):

    reader = initialize_easyocr()

    pil_image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    rgb = np.array(
        pil_image
    )

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    start = time.perf_counter()

    results = reader.readtext(
        rgb,
        detail=1,
        paragraph=False,
        width_ths=0.7,
        height_ths=0.5,
        slope_ths=0.15
    )

    detections = []

    for item in results:

        if not isinstance(
            item,
            (list, tuple)
        ):
            continue

        if len(item) < 3:
            continue

        bbox = normalize_bbox(
            item[0]
        )

        text = str(
            item[1]
        ).strip()

        try:
            confidence = float(
                item[2]
            )
        except Exception:
            confidence = 0.0

        if not text:
            continue

        if confidence < min_confidence:
            continue

        if bbox is None:
            continue

        if refine_boxes:

            bbox = refine_easy_bbox(
                bbox,
                bgr
            )

        detections.append(
            {
                "bbox": bbox,
                "text": text,
                "confidence": confidence
            }
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    return {
        "success": True,
        "detections": detections,
        "text": detections_to_text(
            detections
        ),
        "confidence": average_confidence(
            detections
        ),
        "time": elapsed
    }


# ==========================================================
# PADDLE OCR INITIALIZATION
# ==========================================================

@st.cache_resource(
    show_spinner=False
)
def initialize_paddleocr():

    try:

        from paddleocr import PaddleOCR

    except Exception as error:

        raise RuntimeError(
            "PaddleOCR could not be imported.\n\n"
            f"Original error: {error}"
        ) from error

    errors = []

    # ------------------------------------------------------
    # PaddleOCR 3.x
    # ------------------------------------------------------

    try:

        reader = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False
        )

        return reader

    except Exception as error:

        errors.append(
            "PaddleOCR 3.x initialization:\n"
            + str(error)
        )

    # ------------------------------------------------------
    # PaddleOCR 3.x without enable_mkldnn
    # Some versions don't recognize this parameter.
    # ------------------------------------------------------

    try:

        reader = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )

        return reader

    except Exception as error:

        errors.append(
            "PaddleOCR 3.x fallback:\n"
            + str(error)
        )

    # ------------------------------------------------------
    # PaddleOCR 2.x
    # ------------------------------------------------------

    try:

        reader = PaddleOCR(
            lang="en",
            use_angle_cls=False
        )

        return reader

    except Exception as error:

        errors.append(
            "PaddleOCR 2.x initialization:\n"
            + str(error)
        )

    raise RuntimeError(
        "PaddleOCR was installed/imported but "
        "could not be initialized.\n\n"
        "This usually indicates a "
        "PaddlePaddle/PaddleOCR compatibility "
        "problem.\n\n"
        + "\n\n".join(errors)
    )


# ==========================================================
# PADDLE RESULT PARSER
# ==========================================================

def parse_paddle_result(result):

    detections = []

    if result is None:
        return detections

    # ------------------------------------------------------
    # PaddleOCR 3.x result object
    # ------------------------------------------------------

    if hasattr(result, "json"):

        try:

            data = result.json

            if callable(data):
                data = data()

            return parse_paddle_result(
                data
            )

        except Exception:
            pass

    # ------------------------------------------------------
    # Dictionary
    # ------------------------------------------------------

    if isinstance(result, dict):

        texts = (
            result.get("rec_texts")
            or result.get("texts")
            or result.get("text")
        )

        scores = (
            result.get("rec_scores")
            or result.get("scores")
            or result.get("score")
        )

        boxes = (
            result.get("rec_polys")
            or result.get("dt_polys")
            or result.get("boxes")
            or result.get("box")
        )

        if texts is not None:

            if isinstance(
                texts,
                str
            ):
                texts = [texts]

            if scores is None:

                scores = [
                    0.0
                    for _ in texts
                ]

            elif not isinstance(
                scores,
                (list, tuple, np.ndarray)
            ):

                scores = [
                    scores
                    for _ in texts
                ]

            if boxes is None:

                boxes = [
                    None
                    for _ in texts
                ]

            for index, text in enumerate(
                texts
            ):

                text = str(
                    text
                ).strip()

                if not text:
                    continue

                try:

                    confidence = float(
                        scores[index]
                    )

                except Exception:

                    confidence = 0.0

                bbox = None

                if (
                    boxes is not None
                    and index < len(boxes)
                ):

                    bbox = normalize_bbox(
                        boxes[index]
                    )

                detections.append(
                    {
                        "bbox": bbox,
                        "text": text,
                        "confidence": confidence
                    }
                )

            return detections

        for value in result.values():

            nested = parse_paddle_result(
                value
            )

            if nested:
                detections.extend(
                    nested
                )

        return detections

    # ------------------------------------------------------
    # List / Tuple
    # ------------------------------------------------------

    if isinstance(
        result,
        (list, tuple)
    ):

        # Old PaddleOCR:
        #
        # [
        #     bbox,
        #     ["text", confidence]
        # ]

        if len(result) == 2:

            bbox = result[0]
            recognition = result[1]

            if isinstance(
                recognition,
                (list, tuple)
            ) and len(recognition) >= 2:

                text = str(
                    recognition[0]
                ).strip()

                try:

                    confidence = float(
                        recognition[1]
                    )

                except Exception:

                    confidence = 0.0

                if text:

                    detections.append(
                        {
                            "bbox":
                                normalize_bbox(
                                    bbox
                                ),
                            "text": text,
                            "confidence":
                                confidence
                        }
                    )

                    return detections

        for item in result:

            nested = parse_paddle_result(
                item
            )

            if nested:

                detections.extend(
                    nested
                )

        return detections

    return detections


# ==========================================================
# PADDLE OCR EXECUTION
# ==========================================================

def run_paddleocr(
    reader,
    image
):

    # PaddleOCR 3.x
    if hasattr(
        reader,
        "predict"
    ):

        result = reader.predict(
            image
        )

        detections = []

        try:

            for item in result:

                parsed = parse_paddle_result(
                    item
                )

                detections.extend(
                    parsed
                )

        except TypeError:

            detections = parse_paddle_result(
                result
            )

        if detections:

            return detections

    # PaddleOCR 2.x
    if hasattr(
        reader,
        "ocr"
    ):

        result = reader.ocr(
            image,
            cls=False
        )

        detections = parse_paddle_result(
            result
        )

        if detections:

            return detections

    raise RuntimeError(
        "PaddleOCR completed inference but "
        "returned no recognizable text."
    )


# ==========================================================
# CACHED PADDLE OCR
# ==========================================================

@st.cache_data(
    show_spinner=False
)
def cached_paddleocr(
    image_bytes,
    min_confidence
):

    reader = initialize_paddleocr()

    pil_image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    rgb = np.array(
        pil_image
    )

    start = time.perf_counter()

    detections = run_paddleocr(
        reader,
        rgb
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    detections = [
        detection
        for detection in detections
        if detection.get(
            "confidence",
            0.0
        ) >= min_confidence
    ]

    return {
        "success": True,
        "detections": detections,
        "text": detections_to_text(
            detections
        ),
        "confidence": average_confidence(
            detections
        ),
        "time": elapsed
    }


# ==========================================================
# TEXT / CONFIDENCE
# ==========================================================

def detections_to_text(
    detections
):

    lines = []

    for detection in detections:

        text = str(
            detection.get(
                "text",
                ""
            )
        ).strip()

        if text:

            lines.append(
                text
            )

    return clean_text(
        "\n".join(lines)
    )


def average_confidence(
    detections
):

    if not detections:
        return 0.0

    values = []

    for detection in detections:

        try:

            values.append(
                float(
                    detection.get(
                        "confidence",
                        0
                    )
                )
            )

        except Exception:
            pass

    if not values:
        return 0.0

    return sum(values) / len(values)


# ==========================================================
# DRAW DETECTIONS
# ==========================================================

def draw_detections(
    image,
    detections
):

    output = image.copy()

    height, width = output.shape[:2]

    for detection in detections:

        bbox = detection.get(
            "bbox"
        )

        if bbox is None:
            continue

        try:

            points = np.asarray(
                bbox,
                dtype=np.int32
            )

            if (
                points.ndim != 2
                or points.shape[1] != 2
            ):
                continue

            points[:, 0] = np.clip(
                points[:, 0],
                0,
                width - 1
            )

            points[:, 1] = np.clip(
                points[:, 1],
                0,
                height - 1
            )

            cv2.polylines(
                output,
                [points],
                True,
                (40, 180, 80),
                2,
                cv2.LINE_AA
            )

        except Exception:
            continue

    return output


# ==========================================================
# HERO
# ==========================================================

st.title("Document OCR Studio")

st.markdown(
    """
    **A dual-engine optical character recognition workspace
    powered by EasyOCR, PaddleOCR, OpenCV, and Streamlit.**
    """
)

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown(
        "## OCR Configuration"
    )

    engine = st.selectbox(
        "Recognition Engine",
        [
            "EasyOCR",
            "PaddleOCR",
            "Compare Engines"
        ]
    )

    st.markdown("---")

    st.markdown(
        "### Image Preprocessing"
    )

    preprocessing_enabled = st.toggle(
        "Enable preprocessing",
        value=True
    )

    grayscale_enabled = st.checkbox(
        "Grayscale",
        value=True,
        disabled=not preprocessing_enabled
    )

    denoise_enabled = st.checkbox(
        "Denoising",
        value=False,
        disabled=not preprocessing_enabled
    )

    enhancement_enabled = st.checkbox(
        "Contrast enhancement",
        value=True,
        disabled=not preprocessing_enabled
    )

    threshold_enabled = st.checkbox(
        "Adaptive threshold",
        value=False,
        disabled=not preprocessing_enabled
    )

    st.markdown("---")

    min_confidence = st.slider(
        "Minimum confidence",
        0.0,
        1.0,
        0.20,
        0.05
    )

    show_boxes = st.toggle(
        "Show detection boxes",
        value=True
    )

    refine_boxes = st.toggle(
        "Refine EasyOCR boxes",
        value=True
    )

    st.caption(
        "EasyOCR boxes are refined against "
        "the original document."
    )


# ==========================================================
# FILE UPLOAD
# ==========================================================

st.markdown(
    '<div class="section-title">Document Input</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a document image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp"
    ],
    label_visibility="collapsed"
)

if uploaded_file is None:

    st.info(
        "Upload a document image to begin OCR analysis."
    )

    st.stop()


# ==========================================================
# LOAD IMAGE
# ==========================================================

try:

    (
        image_bytes,
        rgb_image,
        bgr_image
    ) = load_image(
        uploaded_file
    )

except Exception as error:

    st.error(
        f"Unable to read image: {error}"
    )

    st.stop()


# ==========================================================
# IMAGE ID
#
# When a new image is uploaded, old OCR results are removed.
# Changing engines does NOT remove results.
# ==========================================================

image_hash = calculate_image_hash(
    image_bytes
)

if (
    st.session_state.current_image_hash
    != image_hash
):

    st.session_state.current_image_hash = (
        image_hash
    )

    st.session_state.ocr_results = {}

    st.session_state.paddle_error = None


# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

if preprocessing_enabled:

    prepared_image = preprocess_image(
        bgr_image,
        grayscale=grayscale_enabled,
        denoise=denoise_enabled,
        enhance=enhancement_enabled,
        threshold=threshold_enabled
    )

else:

    prepared_image = bgr_image.copy()


prepared_rgb = cv2.cvtColor(
    prepared_image,
    cv2.COLOR_BGR2RGB
)


# ==========================================================
# IMAGE PREVIEW
# ==========================================================

st.markdown(
    '<div class="section-title">Image Preview</div>',
    unsafe_allow_html=True
)

preview_col1, preview_col2 = st.columns(2)

with preview_col1:

    st.caption(
        "Original document"
    )

    st.image(
        rgb_image,
        use_container_width=True
    )

with preview_col2:

    st.caption(
        "OCR-ready image"
    )

    st.image(
        prepared_rgb,
        use_container_width=True
    )


# ==========================================================
# OCR CONTROLS
# ==========================================================

st.markdown(
    '<div class="section-title">OCR Analysis</div>',
    unsafe_allow_html=True
)

button_col1, button_col2 = st.columns(2)


with button_col1:

    run_selected = st.button(
        "Run Selected Engine",
        type="primary",
        use_container_width=True
    )


with button_col2:

    run_both = st.button(
        "Run Both Engines",
        use_container_width=True
    )


# ==========================================================
# RUN SELECTED ENGINE
# ==========================================================

if run_selected:

    # ------------------------------------------------------
    # EASY OCR
    # ------------------------------------------------------

    if engine == "EasyOCR":

        try:

            with st.spinner(
                "Running EasyOCR..."
            ):

                result = cached_easyocr(
                    image_bytes,
                    min_confidence,
                    refine_boxes
                )

            st.session_state.ocr_results[
                "EasyOCR"
            ] = result

            st.success(
                "EasyOCR completed."
            )

        except Exception as error:

            st.session_state.ocr_results[
                "EasyOCR"
            ] = {
                "success": False,
                "error": str(error)
            }

    # ------------------------------------------------------
    # PADDLE OCR
    # ------------------------------------------------------

    elif engine == "PaddleOCR":

        try:

            with st.spinner(
                "Running PaddleOCR..."
            ):

                result = cached_paddleocr(
                    image_bytes,
                    min_confidence
                )

            st.session_state.ocr_results[
                "PaddleOCR"
            ] = result

            st.session_state.paddle_error = None

            st.success(
                "PaddleOCR completed."
            )

        except Exception as error:

            error_text = str(
                error
            )

            st.session_state.paddle_error = (
                error_text
            )

            st.session_state.ocr_results[
                "PaddleOCR"
            ] = {
                "success": False,
                "error": error_text
            }

    # ------------------------------------------------------
    # COMPARE
    #
    # Run only missing engines.
    # Existing results remain untouched.
    # ------------------------------------------------------

    else:

        # EasyOCR
        if "EasyOCR" not in st.session_state.ocr_results:

            try:

                with st.spinner(
                    "Running EasyOCR..."
                ):

                    st.session_state.ocr_results[
                        "EasyOCR"
                    ] = cached_easyocr(
                        image_bytes,
                        min_confidence,
                        refine_boxes
                    )

            except Exception as error:

                st.session_state.ocr_results[
                    "EasyOCR"
                ] = {
                    "success": False,
                    "error": str(error)
                }

        # PaddleOCR
        if "PaddleOCR" not in st.session_state.ocr_results:

            try:

                with st.spinner(
                    "Running PaddleOCR..."
                ):

                    st.session_state.ocr_results[
                        "PaddleOCR"
                    ] = cached_paddleocr(
                        image_bytes,
                        min_confidence
                    )

                st.session_state.paddle_error = None

            except Exception as error:

                error_text = str(
                    error
                )

                st.session_state.paddle_error = (
                    error_text
                )

                st.session_state.ocr_results[
                    "PaddleOCR"
                ] = {
                    "success": False,
                    "error": error_text
                }


# ==========================================================
# RUN BOTH ENGINES
#
# IMPORTANT:
# This does NOT clear existing results.
# ==========================================================

if run_both:

    # ------------------------------------------------------
    # EASY OCR
    # ------------------------------------------------------

    try:

        with st.spinner(
            "Running EasyOCR..."
        ):

            st.session_state.ocr_results[
                "EasyOCR"
            ] = cached_easyocr(
                image_bytes,
                min_confidence,
                refine_boxes
            )

    except Exception as error:

        st.session_state.ocr_results[
            "EasyOCR"
        ] = {
            "success": False,
            "error": str(error)
        }

    # ------------------------------------------------------
    # PADDLE OCR
    # ------------------------------------------------------

    try:

        with st.spinner(
            "Running PaddleOCR..."
        ):

            st.session_state.ocr_results[
                "PaddleOCR"
            ] = cached_paddleocr(
                image_bytes,
                min_confidence
            )

        st.session_state.paddle_error = None

    except Exception as error:

        error_text = str(
            error
        )

        st.session_state.paddle_error = (
            error_text
        )

        st.session_state.ocr_results[
            "PaddleOCR"
        ] = {
            "success": False,
            "error": error_text
        }


# ==========================================================
# RESULTS
# ==========================================================

results = st.session_state.ocr_results


if results:

    st.markdown(
        '<div class="section-title">Recognition Results</div>',
        unsafe_allow_html=True
    )


# ==========================================================
# SINGLE ENGINE VIEW
# ==========================================================

if engine in (
    "EasyOCR",
    "PaddleOCR"
):

    result = results.get(
        engine
    )

    if not result:

        st.info(
            f"{engine} has not been run yet."
        )

        st.caption(
            "Run the selected engine above. "
            "Previous results from the other engine "
            "will remain available."
        )

    elif not result.get(
        "success",
        False
    ):

        st.error(
            f"{engine} failed."
        )

        st.code(
            result.get(
                "error",
                "Unknown error"
            )
        )

        if engine == "PaddleOCR":

            st.warning(
                "PaddleOCR is currently unavailable "
                "in this Python environment. "
                "EasyOCR remains available."
            )

    else:

        detections = result.get(
            "detections",
            []
        )

        extracted_text = result.get(
            "text",
            ""
        )

        confidence = result.get(
            "confidence",
            0.0
        )

        elapsed = result.get(
            "time",
            0.0
        )

        metric1, metric2, metric3 = (
            st.columns(3)
        )

        with metric1:

            st.metric(
                "Text Regions",
                len(detections)
            )

        with metric2:

            st.metric(
                "Average Confidence",
                f"{confidence * 100:.1f}%"
            )

        with metric3:

            st.metric(
                "Processing Time",
                f"{elapsed:.2f} sec"
            )

        # --------------------------------------------------
        # Detection image
        # --------------------------------------------------

        if show_boxes:

            detection_image = draw_detections(
                bgr_image,
                detections
            )

            detection_image = cv2.cvtColor(
                detection_image,
                cv2.COLOR_BGR2RGB
            )

        else:

            detection_image = rgb_image

        result_col1, result_col2 = (
            st.columns(
                [1.1, 0.9]
            )
        )

        with result_col1:

            st.caption(
                f"{engine} detected text regions"
            )

            st.image(
                detection_image,
                use_container_width=True
            )

        with result_col2:

            st.caption(
                f"{engine} extracted text"
            )

            st.text_area(
                "OCR result",
                value=extracted_text,
                height=400,
                label_visibility="collapsed",
                key=f"{engine}_single_output"
            )


# ==========================================================
# COMPARE ENGINES VIEW
# ==========================================================

else:

    easy_result = results.get(
        "EasyOCR"
    )

    paddle_result = results.get(
        "PaddleOCR"
    )

    compare_col1, compare_col2 = (
        st.columns(2)
    )


    # ======================================================
    # EASY OCR
    # ======================================================

    with compare_col1:

        st.markdown(
            """
            <div class="engine-card">
                <div class="engine-name">
                    EasyOCR
                </div>
                <div class="engine-description">
                    General-purpose OCR engine with
                    refined document text boxes.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if not easy_result:

            st.info(
                "EasyOCR has not been run yet."
            )

        elif not easy_result.get(
            "success",
            False
        ):

            st.error(
                "EasyOCR failed."
            )

            st.code(
                easy_result.get(
                    "error",
                    "Unknown error"
                )
            )

        else:

            st.metric(
                "Confidence",
                f"{easy_result.get('confidence', 0) * 100:.1f}%"
            )

            st.caption(
                f"Text regions: "
                f"{len(easy_result.get('detections', []))}"
                f" · Processing time: "
                f"{easy_result.get('time', 0):.2f}s"
            )

            st.text_area(
                "EasyOCR output",
                value=easy_result.get(
                    "text",
                    ""
                ),
                height=350,
                label_visibility="collapsed",
                key="easy_compare_output"
            )


    # ======================================================
    # PADDLE OCR
    # ======================================================

    with compare_col2:

        st.markdown(
            """
            <div class="engine-card">
                <div class="engine-name">
                    PaddleOCR
                </div>
                <div class="engine-description">
                    Document-oriented OCR engine.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if not paddle_result:

            st.info(
                "PaddleOCR has not been run yet."
            )

        elif not paddle_result.get(
            "success",
            False
        ):

            st.warning(
                "PaddleOCR is unavailable "
                "in the current environment."
            )

            st.code(
                paddle_result.get(
                    "error",
                    "PaddleOCR failed."
                )
            )

            st.caption(
                "This does not affect EasyOCR."
            )

        else:

            st.metric(
                "Confidence",
                f"{paddle_result.get('confidence', 0) * 100:.1f}%"
            )

            st.caption(
                f"Text regions: "
                f"{len(paddle_result.get('detections', []))}"
                f" · Processing time: "
                f"{paddle_result.get('time', 0):.2f}s"
            )

            st.text_area(
                "PaddleOCR output",
                value=paddle_result.get(
                    "text",
                    ""
                ),
                height=350,
                label_visibility="collapsed",
                key="paddle_compare_output"
            )


    # ======================================================
    # VISUAL COMPARISON
    # ======================================================

    if (
        easy_result
        and easy_result.get("success", False)
    ):

        st.markdown(
            '<div class="section-title">EasyOCR Detection Map</div>',
            unsafe_allow_html=True
        )

        easy_image = draw_detections(
            bgr_image,
            easy_result.get(
                "detections",
                []
            )
        )

        easy_image = cv2.cvtColor(
            easy_image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            easy_image,
            use_container_width=True
        )


    if (
        paddle_result
        and paddle_result.get("success", False)
    ):

        st.markdown(
            '<div class="section-title">PaddleOCR Detection Map</div>',
            unsafe_allow_html=True
        )

        paddle_image = draw_detections(
            bgr_image,
            paddle_result.get(
                "detections",
                []
            )
        )

        paddle_image = cv2.cvtColor(
            paddle_image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            paddle_image,
            use_container_width=True
        )


    # ======================================================
    # COMPARISON TABLE
    # ======================================================

    st.markdown(
        '<div class="section-title">Engine Comparison</div>',
        unsafe_allow_html=True
    )

    rows = []

    for name in [
        "EasyOCR",
        "PaddleOCR"
    ]:

        result = results.get(
            name
        )

        if not result:

            rows.append(
                {
                    "Engine": name,
                    "Status": "Not Run",
                    "Regions": "—",
                    "Confidence": "—",
                    "Time": "—"
                }
            )

        elif not result.get(
            "success",
            False
        ):

            rows.append(
                {
                    "Engine": name,
                    "Status": "Unavailable",
                    "Regions": "—",
                    "Confidence": "—",
                    "Time": "—"
                }
            )

        else:

            rows.append(
                {
                    "Engine": name,
                    "Status": "Success",
                    "Regions": len(
                        result.get(
                            "detections",
                            []
                        )
                    ),
                    "Confidence":
                        f"{result.get('confidence', 0) * 100:.1f}%",
                    "Time":
                        f"{result.get('time', 0):.2f}s"
                }
            )

    st.table(
        rows
    )


# ==========================================================
# PADDLE DIAGNOSTIC
# ==========================================================

if (
    st.session_state.paddle_error
):

    paddle_error_text = (
        st.session_state.paddle_error
    )

    if (
        "ConvertPirAttribute2RuntimeAttribute"
        in paddle_error_text
        or "onednn"
        in paddle_error_text.lower()
        or "mkldnn"
        in paddle_error_text.lower()
        or "pir::"
        in paddle_error_text
    ):

        with st.expander(
            "Why is PaddleOCR unavailable?"
        ):

            st.markdown(
                """
                **PaddleOCR encountered a
                PaddlePaddle runtime compatibility
                error.**

                This occurs inside the PaddlePaddle
                execution runtime rather than in the
                OCR result parser.

                The application disables the
                oneDNN/MKLDNN execution path where
                possible and keeps PaddleOCR failure
                isolated from EasyOCR.

                EasyOCR can continue working normally.
                """
            )

            st.code(
                paddle_error_text
            )


# ==========================================================
# EXPORT
# ==========================================================

successful_results = {
    name: result
    for name, result in results.items()
    if result.get(
        "success",
        False
    )
}


if successful_results:

    st.markdown(
        '<div class="section-title">Export</div>',
        unsafe_allow_html=True
    )

    for name, result in (
        successful_results.items()
    ):

        safe_name = (
            name.lower()
            .replace(
                " ",
                "_"
            )
        )

        filename = (
            os.path.splitext(
                uploaded_file.name
            )[0]
            + f"_{safe_name}_ocr.txt"
        )

        st.download_button(
            f"Download {name} Result",
            result.get(
                "text",
                ""
            ),
            file_name=filename,
            mime="text/plain",
            use_container_width=True,
            key=f"download_{safe_name}"
        )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
    """
    <div class="footer">
        Day 23 · Document OCR Studio
        <br>
        EasyOCR · PaddleOCR · OpenCV · Streamlit
        <br><br>
        Developed by Hadeed Jalani
    </div>
    """,
    unsafe_allow_html=True
)