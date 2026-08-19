# ============================================================
# MLB SUMMER INTERNSHIP - DAY 23
# DOCUMENT OCR STUDIO
#
# EasyOCR + PaddleOCR + OpenCV + Streamlit
#
# CPU-optimized version
# PaddlePaddle 3.2.0
# PaddleOCR 3.2.0
# PaddleX 3.2.0
# ============================================================

# ============================================================
# PADDLE CPU SETTINGS
# MUST BE SET BEFORE IMPORTING PADDLE / PADDLEOCR
# ============================================================

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

os.environ["GLOG_v"] = "0"


# ============================================================
# IMPORTS
# ============================================================

import hashlib
import io
import re
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Document OCR Studio",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2.3rem 2.5rem;
        margin-bottom: 1.8rem;
        border-radius: 18px;
        background: linear-gradient(
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
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 900px;
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


# ============================================================
# SESSION STATE
# ============================================================

if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = {}

if "image_hash" not in st.session_state:
    st.session_state.image_hash = None

if "paddle_error" not in st.session_state:
    st.session_state.paddle_error = None


# ============================================================
# HELPERS
# ============================================================

def clean_text(text):
    if text is None:
        return ""

    text = str(text)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def image_hash(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()


def average_confidence(detections):
    if not detections:
        return 0.0

    values = []

    for detection in detections:
        try:
            values.append(
                float(
                    detection.get(
                        "confidence",
                        0.0
                    )
                )
            )
        except Exception:
            pass

    if not values:
        return 0.0

    return sum(values) / len(values)


def detections_to_text(detections):

    lines = []

    for detection in detections:

        text = str(
            detection.get(
                "text",
                ""
            )
        ).strip()

        if text:
            lines.append(text)

    return clean_text(
        "\n".join(lines)
    )


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(uploaded_file):

    image_bytes = uploaded_file.getvalue()

    pil_image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    rgb = np.array(pil_image)

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    return image_bytes, rgb, bgr


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(
    image,
    grayscale=True,
    denoise=False,
    enhance=True,
    threshold=False
):

    working = image.copy()

    if grayscale:

        gray = cv2.cvtColor(
            working,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = working

    if denoise and len(gray.shape) == 2:

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


# ============================================================
# BOUNDING BOX
# ============================================================

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

        # x1,y1,x2,y2
        if (
            points.ndim == 1
            and len(points) == 4
        ):

            x1, y1, x2, y2 = points

            return [
                [int(x1), int(y1)],
                [int(x2), int(y1)],
                [int(x2), int(y2)],
                [int(x1), int(y2)]
            ]

    except Exception:
        return None

    return None


# ============================================================
# EASY OCR MODEL
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def get_easyocr():

    import easyocr

    return easyocr.Reader(
        ["en"],
        gpu=False,
        verbose=False
    )


# ============================================================
# EASY OCR
# ============================================================

@st.cache_data(
    show_spinner=False
)
def run_easyocr(
    image_bytes,
    min_confidence
):

    reader = get_easyocr()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    rgb = np.array(image)

    start = time.perf_counter()

    results = reader.readtext(
        rgb,
        detail=1,
        paragraph=False,
        width_ths=0.7,
        height_ths=0.5,
        slope_ths=0.15
    )

    elapsed = (
        time.perf_counter()
        - start
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

        detections.append(
            {
                "bbox": bbox,
                "text": text,
                "confidence": confidence
            }
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


# ============================================================
# PADDLE OCR MODEL
#
# IMPORTANT:
# Explicitly use MOBILE models.
#
# This prevents PaddleOCR from selecting:
# PP-OCRv5_server_det
#
# We want:
# PP-OCRv5_mobile_det
# PP-OCRv5_mobile_rec
#
# CPU only.
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def get_paddleocr():

    from paddleocr import PaddleOCR

    # --------------------------------------------------------
    # PaddleOCR 3.x / 3.2.x
    #
    # Mobile models are substantially lighter than server
    # models and are appropriate for this project.
    # --------------------------------------------------------

    reader = PaddleOCR(
        lang="en",

        text_detection_model_name=(
            "PP-OCRv5_mobile_det"
        ),

        text_recognition_model_name=(
            "PP-OCRv5_mobile_rec"
        ),

        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,

        device="cpu",

        enable_mkldnn=False,
    )

    return reader


# ============================================================
# PADDLE RESULT PARSER
# ============================================================

def parse_paddle_result(result):

    detections = []

    if result is None:
        return detections

    # --------------------------------------------------------
    # PaddleOCR result object
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

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

            if boxes is None:

                boxes = [
                    None
                    for _ in texts
                ]

            for i, text in enumerate(texts):

                text = str(
                    text
                ).strip()

                if not text:
                    continue

                try:

                    confidence = float(
                        scores[i]
                    )

                except Exception:

                    confidence = 0.0

                bbox = None

                if (
                    i < len(boxes)
                    and boxes[i] is not None
                ):

                    bbox = normalize_bbox(
                        boxes[i]
                    )

                detections.append(
                    {
                        "bbox": bbox,
                        "text": text,
                        "confidence": confidence
                    }
                )

            return detections

        # Recursive parsing
        for value in result.values():

            nested = parse_paddle_result(
                value
            )

            detections.extend(
                nested
            )

        return detections

    # --------------------------------------------------------
    # List / tuple
    # --------------------------------------------------------

    if isinstance(
        result,
        (list, tuple)
    ):

        # PaddleOCR 2.x style:
        #
        # [bbox, ["text", score]]

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

            detections.extend(
                nested
            )

        return detections

    return detections


# ============================================================
# PADDLE INFERENCE
# ============================================================

def execute_paddleocr(
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

        for item in result:

            parsed = parse_paddle_result(
                item
            )

            detections.extend(
                parsed
            )

        if detections:
            return detections

    # PaddleOCR 2.x fallback
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

    return []


# ============================================================
# PADDLE OCR
# ============================================================

@st.cache_data(
    show_spinner=False
)
def run_paddleocr(
    image_bytes,
    min_confidence
):

    reader = get_paddleocr()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    rgb = np.array(image)

    start = time.perf_counter()

    detections = execute_paddleocr(
        reader,
        rgb
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    detections = [
        d
        for d in detections
        if d.get(
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


# ============================================================
# DRAW BOXES
# ============================================================

def draw_boxes(
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


# ============================================================
# HERO
# ============================================================

st.title("Document OCR Studio")

st.markdown(
    """
    **A dual-engine optical character recognition workspace
    powered by EasyOCR, PaddleOCR, OpenCV, and Streamlit.**
    """
)



# ============================================================
# SIDEBAR
# ============================================================

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

    st.markdown("---")

    st.caption(
        "PaddleOCR is configured for CPU inference "
        "using lightweight PP-OCRv5 mobile models."
    )


# ============================================================
# UPLOAD
# ============================================================

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


# ============================================================
# LOAD IMAGE
# ============================================================

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


# ============================================================
# RESET RESULTS FOR NEW IMAGE
# ============================================================

current_hash = image_hash(
    image_bytes
)

if (
    st.session_state.image_hash
    != current_hash
):

    st.session_state.image_hash = (
        current_hash
    )

    st.session_state.ocr_results = {}

    st.session_state.paddle_error = None


# ============================================================
# PREPROCESS
# ============================================================

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


# ============================================================
# IMAGE PREVIEW
# ============================================================

st.markdown(
    '<div class="section-title">Image Preview</div>',
    unsafe_allow_html=True
)

preview_col1, preview_col2 = st.columns(
    2
)

with preview_col1:

    st.caption(
        "Original document"
    )

    st.image(
        rgb_image,
        width="stretch"
    )


with preview_col2:

    st.caption(
        "OCR-ready image"
    )

    st.image(
        prepared_rgb,
        width="stretch"
    )


# ============================================================
# OCR CONTROLS
# ============================================================

st.markdown(
    '<div class="section-title">OCR Analysis</div>',
    unsafe_allow_html=True
)

button_col1, button_col2 = st.columns(
    2
)

with button_col1:

    run_selected = st.button(
        "Run Selected Engine",
        type="primary",
        width="stretch"
    )

with button_col2:

    run_both = st.button(
        "Run Both Engines",
        width="stretch"
    )


# ============================================================
# RUN EASY OCR
# ============================================================

def execute_easy():

    try:

        with st.spinner(
            "Running EasyOCR..."
        ):

            result = run_easyocr(
                image_bytes,
                min_confidence
            )

        st.session_state.ocr_results[
            "EasyOCR"
        ] = result

        return True

    except Exception as error:

        st.session_state.ocr_results[
            "EasyOCR"
        ] = {
            "success": False,
            "error": str(error)
        }

        return False


# ============================================================
# RUN PADDLE OCR
# ============================================================

def execute_paddle():

    try:

        with st.spinner(
            "Loading PaddleOCR CPU model and running OCR..."
        ):

            result = run_paddleocr(
                image_bytes,
                min_confidence
            )

        st.session_state.ocr_results[
            "PaddleOCR"
        ] = result

        st.session_state.paddle_error = None

        return True

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

        return False


# ============================================================
# SELECTED ENGINE
# ============================================================

if run_selected:

    if engine == "EasyOCR":

        if execute_easy():

            st.success(
                "EasyOCR completed successfully."
            )

    elif engine == "PaddleOCR":

        if execute_paddle():

            st.success(
                "PaddleOCR completed successfully."
            )

    else:

        if "EasyOCR" not in st.session_state.ocr_results:
            execute_easy()

        if "PaddleOCR" not in st.session_state.ocr_results:
            execute_paddle()


# ============================================================
# RUN BOTH
# ============================================================

if run_both:

    execute_easy()
    execute_paddle()


# ============================================================
# RESULTS
# ============================================================

results = st.session_state.ocr_results


if results:

    st.markdown(
        '<div class="section-title">Recognition Results</div>',
        unsafe_allow_html=True
    )


# ============================================================
# SINGLE ENGINE
# ============================================================

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
                "PaddleOCR is unavailable. "
                "EasyOCR remains available."
            )

    else:

        detections = result.get(
            "detections",
            []
        )

        confidence = result.get(
            "confidence",
            0.0
        )

        elapsed = result.get(
            "time",
            0.0
        )

        metric1, metric2, metric3 = st.columns(
            3
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
                "OCR Time",
                f"{elapsed:.2f}s"
            )

        # ----------------------------------------------------
        # DETECTION MAP
        # ----------------------------------------------------

        if show_boxes:

            detection_image = draw_boxes(
                bgr_image,
                detections
            )

            detection_image = cv2.cvtColor(
                detection_image,
                cv2.COLOR_BGR2RGB
            )

        else:

            detection_image = rgb_image

        result_col1, result_col2 = st.columns(
            [1.1, 0.9]
        )

        with result_col1:

            st.caption(
                f"{engine} detection map"
            )

            st.image(
                detection_image,
                width="stretch"
            )

        with result_col2:

            st.caption(
                f"{engine} extracted text"
            )

            st.text_area(
                "OCR result",
                value=result.get(
                    "text",
                    ""
                ),
                height=400,
                label_visibility="collapsed",
                key=f"{engine}_single_result"
            )


# ============================================================
# COMPARE ENGINES
# ============================================================

else:

    easy_result = results.get(
        "EasyOCR"
    )

    paddle_result = results.get(
        "PaddleOCR"
    )

    compare_col1, compare_col2 = st.columns(
        2
    )

    # --------------------------------------------------------
    # EASY OCR
    # --------------------------------------------------------

    with compare_col1:

        st.html("""
<div style="
    background-color: #1b1e26;
    border: 1px solid #30343d;
    border-radius: 12px;
    padding: 25px;
    color: white;
">
    <div style="
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 18px;
    ">
        EasyOCR
    </div>

    <div style="
        font-size: 16px;
        line-height: 1.6;
        color: #e5e7eb;
    ">
        General-purpose OCR engine<br>
        for document text recognition.
    </div>
</div>
""")

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
                f"Regions: "
                f"{len(easy_result.get('detections', []))}"
                f" · Time: "
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

    # --------------------------------------------------------
    # PADDLE OCR
    # --------------------------------------------------------

    with compare_col2:

     st.html("""
<div style="
    background-color: #1b1e26;
    border: 1px solid #30343d;
    border-radius: 12px;
    padding: 25px;
    color: white;
">
    <div style="
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 18px;
    ">
        PaddleOCR
    </div>

    <div style="
        font-size: 16px;
        line-height: 1.6;
        color: #e5e7eb;
    ">
        CPU-optimized document OCR<br>
        using PP-OCRv5 mobile models.
    </div>
</div>
""")


    if not paddle_result:

            st.info(
                "PaddleOCR has not been run yet."
            )

    elif not paddle_result.get(
            "success",
            False
        ):

            st.warning(
                "PaddleOCR failed."
            )

            st.code(
                paddle_result.get(
                    "error",
                    "Unknown error"
                )
            )

    else:

            st.metric(
                "Confidence",
                f"{paddle_result.get('confidence', 0) * 100:.1f}%"
            )

            st.caption(
                f"Regions: "
                f"{len(paddle_result.get('detections', []))}"
                f" · Time: "
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


# ============================================================
# DETECTION MAPS
# ============================================================

if (
    engine == "Compare Engines"
):

    if (
        easy_result
        and easy_result.get(
            "success",
            False
        )
    ):

        st.markdown(
            '<div class="section-title">EasyOCR Detection Map</div>',
            unsafe_allow_html=True
        )

        easy_image = draw_boxes(
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
            width="stretch"
        )

    if (
        paddle_result
        and paddle_result.get(
            "success",
            False
        )
    ):

        st.markdown(
            '<div class="section-title">PaddleOCR Detection Map</div>',
            unsafe_allow_html=True
        )

        paddle_image = draw_boxes(
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
            width="stretch"
        )


# ============================================================
# COMPARISON TABLE
# ============================================================

if engine == "Compare Engines":

    st.markdown(
        '<div class="section-title">Engine Comparison</div>',
        unsafe_allow_html=True
    )

    rows = []

    for name in (
        "EasyOCR",
        "PaddleOCR"
    ):

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
                    "Status": "Failed",
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

    st.table(rows)


# ============================================================
# PADDLE ERROR
# ============================================================

if st.session_state.paddle_error:

    with st.expander(
        "PaddleOCR diagnostic information"
    ):

        st.warning(
            "PaddleOCR encountered an error. "
            "EasyOCR is independent and can continue "
            "working."
        )

        st.code(
            st.session_state.paddle_error
        )


# ============================================================
# EXPORT
# ============================================================

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
        '<div class="section-title">Export Results</div>',
        unsafe_allow_html=True
    )

    for name, result in successful_results.items():

        safe_name = (
            name.lower()
            .replace(
                " ",
                "_"
            )
        )

        original_name = os.path.splitext(
            uploaded_file.name
        )[0]

        filename = (
            f"{original_name}_"
            f"{safe_name}_ocr.txt"
        )

        st.download_button(
            f"Download {name} Result",
            result.get(
                "text",
                ""
            ),
            file_name=filename,
            mime="text/plain",
            width="stretch",
            key=f"download_{safe_name}"
        )


# ============================================================
# FOOTER
# ============================================================

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
