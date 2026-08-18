# ==========================================================
# MLB Summer Internship - Day 22
# Professional OCR Document Reader
# ==========================================================

import io
import os
import re
import time

import cv2
import easyocr
import numpy as np
import streamlit as st
from PIL import Image


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="OCR Document Reader",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# Professional Styling
# ==========================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1450px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    /* Header */

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 16px;
        background:
            linear-gradient(
                135deg,
                rgba(30, 41, 59, 0.98),
                rgba(15, 23, 42, 0.98)
            );
        border: 1px solid rgba(148, 163, 184, 0.18);
        margin-bottom: 1.6rem;
    }

    .hero-title {
        font-size: 2.35rem;
        font-weight: 750;
        color: #f8fafc;
        letter-spacing: -0.8px;
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.65;
    }

    /* Section */

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    /* Cards */

    .info-card {
        padding: 1rem 1.1rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.22);
        background: rgba(128,128,128,0.045);
        min-height: 100px;
    }

    .info-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        opacity: 0.62;
        margin-bottom: 0.35rem;
    }

    .info-value {
        font-size: 1.2rem;
        font-weight: 700;
    }

    /* OCR output */

    .ocr-panel {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.22);
        background: rgba(128,128,128,0.045);
        line-height: 1.75;
        white-space: pre-wrap;
        font-family: "Consolas", "Courier New", monospace;
        font-size: 0.92rem;
    }

    /* Footer */

    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(128,128,128,0.2);
        text-align: center;
        opacity: 0.55;
        font-size: 0.82rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# EasyOCR Model
# ==========================================================

@st.cache_resource(show_spinner=False)
def load_reader():
    """
    Load EasyOCR only once.

    CPU mode is used because Streamlit deployment environments
    generally do not provide CUDA.
    """

    return easyocr.Reader(
        ["en"],
        gpu=False,
        verbose=False,
    )


# ==========================================================
# Utility Functions
# ==========================================================

def resize_for_ocr(image, max_dimension=1600):
    """
    Resize very large images before OCR.

    This significantly reduces CPU processing time while
    preserving enough resolution for normal documents.
    """

    h, w = image.shape[:2]

    largest = max(h, w)

    if largest <= max_dimension:
        return image.copy()

    scale = max_dimension / largest

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )


def upscale_image(image, factor=1.5):
    """
    Upscale small images using Lanczos interpolation.
    """

    if factor <= 1:
        return image

    h, w = image.shape[:2]

    return cv2.resize(
        image,
        (
            int(w * factor),
            int(h * factor),
        ),
        interpolation=cv2.INTER_CUBIC,
    )


def grayscale(image):
    """
    Safely convert any supported image format to grayscale.
    """

    if image is None:
        return None

    if len(image.shape) == 2:
        return image.copy()

    if image.shape[2] == 4:
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2GRAY,
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


def enhanced_contrast(image):
    """
    CLAHE-based contrast enhancement.
    """

    gray = grayscale(image)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(gray)


def denoise_image(image):
    """
    Light denoising without destroying text edges.
    """

    gray = grayscale(image)

    return cv2.fastNlMeansDenoising(
        gray,
        None,
        h=7,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def adaptive_threshold(image):
    """
    Adaptive thresholding for documents with uneven lighting.
    """

    gray = grayscale(image)

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

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
    Otsu thresholding.

    Useful for clean high-contrast printed documents,
    but not ideal for every photograph.
    """

    gray = grayscale(image)

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    _, result = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return result


def prepare_image(
    image,
    method,
    upscale_enabled,
    upscale_factor,
):
    """
    Complete preprocessing pipeline.
    """

    working = image.copy()

    if upscale_enabled:
        working = upscale_image(
            working,
            upscale_factor,
        )

    if method == "Original":
        return working

    if method == "Grayscale":
        return grayscale(working)

    if method == "Enhanced Contrast":
        return enhanced_contrast(working)

    if method == "Denoised":
        return denoise_image(working)

    if method == "Adaptive Threshold":
        return adaptive_threshold(working)

    if method == "Otsu Threshold":
        return otsu_threshold(working)

    return working


def normalize_detection(detection):
    """
    Convert EasyOCR output into a consistent dictionary.

    EasyOCR normally returns:

        (bounding_box, text, confidence)

    Older/custom utility code may return dictionaries.

    This function supports both formats.
    """

    if isinstance(detection, dict):

        return {
            "bbox": detection.get("bbox"),
            "text": str(
                detection.get("text", "")
            ).strip(),
            "confidence": float(
                detection.get("confidence", 0.0)
            ),
        }

    if isinstance(detection, (list, tuple)):

        if len(detection) >= 3:

            return {
                "bbox": detection[0],
                "text": str(
                    detection[1]
                ).strip(),
                "confidence": float(
                    detection[2]
                ),
            }

    return None


def normalize_detections(raw_results):
    """
    Normalize all EasyOCR results.
    """

    normalized = []

    for result in raw_results:

        item = normalize_detection(result)

        if item is not None and item["text"]:

            normalized.append(item)

    return normalized


def get_bbox_metrics(bbox):
    """
    Calculate geometric information from an EasyOCR bounding box.
    """

    points = np.array(
        bbox,
        dtype=np.float32,
    )

    x_values = points[:, 0]
    y_values = points[:, 1]

    left = float(np.min(x_values))
    right = float(np.max(x_values))
    top = float(np.min(y_values))
    bottom = float(np.max(y_values))

    center_x = (left + right) / 2
    center_y = (top + bottom) / 2

    height = max(
        bottom - top,
        1.0,
    )

    width = max(
        right - left,
        1.0,
    )

    return {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "center_x": center_x,
        "center_y": center_y,
        "width": width,
        "height": height,
    }


def sort_reading_order(detections):
    """
    Sort OCR results into natural human reading order.

    EasyOCR does not guarantee that returned detections are
    already ordered from top-to-bottom and left-to-right.

    This function groups detections into approximate text lines.
    """

    if not detections:
        return []

    enriched = []

    for detection in detections:

        if not detection.get("bbox"):
            continue

        metrics = get_bbox_metrics(
            detection["bbox"]
        )

        enriched.append(
            (
                detection,
                metrics,
            )
        )

    if not enriched:
        return detections

    # Estimate typical text height.
    heights = [
        metrics["height"]
        for _, metrics in enriched
    ]

    median_height = float(
        np.median(heights)
    )

    line_tolerance = max(
        median_height * 0.55,
        8,
    )

    # Sort by vertical position first.
    enriched.sort(
        key=lambda item: (
            item[1]["center_y"],
            item[1]["left"],
        )
    )

    lines = []

    for detection, metrics in enriched:

        placed = False

        for line in lines:

            average_y = np.mean(
                [
                    item[1]["center_y"]
                    for item in line
                ]
            )

            if abs(
                metrics["center_y"]
                - average_y
            ) <= line_tolerance:

                line.append(
                    (
                        detection,
                        metrics,
                    )
                )

                placed = True
                break

        if not placed:

            lines.append(
                [
                    (
                        detection,
                        metrics,
                    )
                ]
            )

    # Sort lines vertically.
    lines.sort(
        key=lambda line: np.mean(
            [
                item[1]["center_y"]
                for item in line
            ]
        )
    )

    ordered = []

    for line in lines:

        line.sort(
            key=lambda item:
            item[1]["left"]
        )

        for detection, _ in line:
            ordered.append(
                detection
            )

    return ordered


def clean_text(text):
    """
    Clean common OCR formatting problems without
    aggressively changing the actual OCR output.
    """

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    # Remove excessive spaces.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Remove excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def build_text(
    detections,
    minimum_confidence,
):
    """
    Build final OCR text using confidence filtering
    and natural reading order.
    """

    filtered = [
        item
        for item in detections
        if item["confidence"]
        >= minimum_confidence
    ]

    filtered = sort_reading_order(
        filtered
    )

    lines = []

    previous_y = None

    for detection in filtered:

        text = detection["text"]

        if not text:
            continue

        metrics = get_bbox_metrics(
            detection["bbox"]
        )

        current_y = metrics[
            "center_y"
        ]

        # Detect large vertical gaps.
        if (
            previous_y is not None
            and abs(current_y - previous_y)
            > metrics["height"] * 1.8
        ):
            lines.append("")

        lines.append(text)

        previous_y = current_y

    return clean_text(
        "\n".join(lines)
    ), filtered


def average_confidence(detections):
    """
    Calculate average confidence.
    """

    if not detections:
        return 0.0

    values = [
        item["confidence"]
        for item in detections
    ]

    return float(
        np.mean(values)
    )


def draw_detections(
    image,
    detections,
    minimum_confidence,
):
    """
    Draw detected OCR regions.
    """

    result = image.copy()

    for index, detection in enumerate(
        detections,
        start=1,
    ):

        confidence = detection[
            "confidence"
        ]

        if confidence < minimum_confidence:
            continue

        bbox = np.array(
            detection["bbox"],
            dtype=np.int32,
        )

        cv2.polylines(
            result,
            [bbox],
            True,
            (40, 180, 255),
            2,
        )

        x = int(
            np.min(bbox[:, 0])
        )

        y = int(
            np.min(bbox[:, 1])
        )

        label = (
            f"{index} "
            f"{confidence * 100:.0f}%"
        )

        cv2.putText(
            result,
            label,
            (
                x,
                max(y - 8, 20),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 180, 255),
            2,
            cv2.LINE_AA,
        )

    return result


def rgb_from_bgr(image):
    """
    Convert OpenCV BGR image to RGB.
    """

    if len(image.shape) == 2:
        return image

    if image.shape[2] == 4:
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2RGB,
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )


# ==========================================================
# Session State
# ==========================================================

if "ocr_result" not in st.session_state:
    st.session_state.ocr_result = None


# ==========================================================
# Header
# ==========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">OCR Document Reader</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="
        margin-top:-1.4rem;
        margin-bottom:2rem;
        padding:0 2.5rem 1.5rem 2.5rem;
        color:#cbd5e1;
        font-size:1rem;
        line-height:1.7;
    ">
        A computer vision application for detecting,
        ordering, analyzing, and exporting text from
        real-world images using EasyOCR and OpenCV.
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# Sidebar Controls
# ==========================================================

with st.sidebar:

    st.markdown("## Processing Controls")

    st.caption(
        "Configure the OCR pipeline before analysis."
    )

    st.markdown("---")

    st.markdown("### OCR Region")

    region_mode = st.selectbox(
        "Select image region",
        [
            "Full Image",
            "Top Region",
            "Center Region",
            "Bottom Region",
        ],
        index=0,
        help=(
            "Use a focused region when the image "
            "contains large non-text objects."
        ),
    )

    st.markdown("---")

    st.markdown("### Preprocessing")

    preprocessing_method = st.selectbox(
        "Preprocessing method",
        [
            "Enhanced Contrast",
            "Grayscale",
            "Adaptive Threshold",
            "Otsu Threshold",
            "Denoised",
            "Original",
        ],
        index=0,
    )

    upscale_enabled = st.checkbox(
        "Upscale small images",
        value=True,
        help=(
            "Improves OCR on small text but increases "
            "processing time."
        ),
    )

    upscale_factor = st.slider(
        "Upscale factor",
        min_value=1.0,
        max_value=2.0,
        value=1.5,
        step=0.1,
        disabled=not upscale_enabled,
    )

    st.markdown("---")

    st.markdown("### OCR Quality")

    minimum_confidence = st.slider(
        "Minimum confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help=(
            "Higher values remove uncertain OCR detections."
        ),
    )

    st.markdown("---")

    show_boxes = st.checkbox(
        "Show detection regions",
        value=True,
    )

    st.markdown("---")

    st.caption(
        "Engine: EasyOCR"
    )

    st.caption(
        "Language: English"
    )

    st.caption(
        "CPU-compatible configuration"
    )


# ==========================================================
# Document Input
# ==========================================================

st.markdown(
    '<div class="section-title">Document Input</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp",
    ],
    label_visibility="collapsed",
)


if uploaded_file is None:

    st.info(
        "Upload an image containing printed, handwritten, "
        "or scene text to begin OCR analysis."
    )

    st.stop()


# ==========================================================
# Load Image
# ==========================================================

try:

    image_bytes = uploaded_file.getvalue()

    pil_image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    original_rgb = np.array(
        pil_image
    )

    original_bgr = cv2.cvtColor(
        original_rgb,
        cv2.COLOR_RGB2BGR,
    )

except Exception as error:

    st.error(
        f"Unable to read image: {error}"
    )

    st.stop()


# ==========================================================
# Image Information
# ==========================================================

height, width = original_bgr.shape[:2]

file_size_kb = (
    len(image_bytes) / 1024
)

st.markdown(
    '<div class="section-title">Document Information</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Dimensions",
        f"{width} × {height}",
    )

with col2:
    st.metric(
        "File Size",
        f"{file_size_kb:.1f} KB",
    )

with col3:
    st.metric(
        "Region",
        region_mode,
    )

with col4:
    st.metric(
        "OCR Threshold",
        f"{minimum_confidence:.2f}",
    )


# ==========================================================
# Region Selection
# ==========================================================

def crop_region(
    image,
    mode,
):
    """
    Select a useful OCR region.
    """

    h, w = image.shape[:2]

    if mode == "Top Region":
        return image[
            :int(h * 0.40),
            :
        ]

    if mode == "Center Region":
        start = int(h * 0.30)
        end = int(h * 0.75)

        return image[
            start:end,
            :
        ]

    if mode == "Bottom Region":
        return image[
            int(h * 0.60):,
            :
        ]

    return image


ocr_region = crop_region(
    original_bgr,
    region_mode,
)


# ==========================================================
# Prepare OCR Image
# ==========================================================

ocr_input = resize_for_ocr(
    ocr_region,
    max_dimension=1600,
)

preprocessed = prepare_image(
    ocr_input,
    preprocessing_method,
    upscale_enabled,
    upscale_factor,
)


# ==========================================================
# Preview
# ==========================================================

st.markdown(
    '<div class="section-title">Image Preview</div>',
    unsafe_allow_html=True,
)

preview_col1, preview_col2 = st.columns(2)

with preview_col1:

    st.caption("Original image")

    st.image(
        original_rgb,
        use_container_width=True,
    )

with preview_col2:

    st.caption(
        f"Preprocessed OCR image · "
        f"{preprocessing_method}"
    )

    st.image(
        rgb_from_bgr(preprocessed),
        use_container_width=True,
    )


# ==========================================================
# OCR Action
# ==========================================================

st.markdown(
    '<div class="section-title">OCR Analysis</div>',
    unsafe_allow_html=True,
)

process_button = st.button(
    "Extract Text",
    type="primary",
    use_container_width=True,
)


# ==========================================================
# OCR Processing
# ==========================================================

if process_button:

    reader = load_reader()

    progress = st.progress(
        0,
        text="Preparing OCR engine...",
    )

    start_time = time.perf_counter()

    try:

        progress.progress(
            20,
            text="Preparing image...",
        )

        # --------------------------------------------------
        # Important:
        #
        # EasyOCR works best when the OCR image is not
        # excessively large.
        # --------------------------------------------------

        progress.progress(
            40,
            text="Detecting text regions...",
        )

        raw_results = reader.readtext(
            preprocessed,
            detail=1,
            paragraph=False,
            width_ths=0.7,
            link_threshold=0.3,
            text_threshold=0.55,
            low_text=0.3,
            mag_ratio=1.0,
        )

        progress.progress(
            70,
            text="Ordering detected text...",
        )

        detections = normalize_detections(
            raw_results
        )

        extracted_text, filtered_detections = (
            build_text(
                detections,
                minimum_confidence,
            )
        )

        progress.progress(
            85,
            text="Calculating confidence...",
        )

        confidence = average_confidence(
            filtered_detections
        )

        if show_boxes:

            detection_image = draw_detections(
                ocr_input,
                filtered_detections,
                minimum_confidence,
            )

        else:

            detection_image = ocr_input

        elapsed = (
            time.perf_counter()
            - start_time
        )

        progress.progress(
            100,
            text="OCR analysis complete.",
        )

        time.sleep(0.15)

        progress.empty()

        st.session_state.ocr_result = {
            "text": extracted_text,
            "detections": filtered_detections,
            "confidence": confidence,
            "detection_image": detection_image,
            "elapsed": elapsed,
            "region": region_mode,
            "method": preprocessing_method,
        }

    except Exception as error:

        progress.empty()

        st.error(
            f"OCR processing failed: {error}"
        )

        st.stop()


# ==========================================================
# Results
# ==========================================================

result = st.session_state.ocr_result


if result is not None:

    extracted_text = result["text"]
    detections = result["detections"]
    confidence = result["confidence"]
    detection_image = result[
        "detection_image"
    ]

    elapsed = result["elapsed"]

    st.markdown(
        '<div class="section-title">Analysis Results</div>',
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Text Regions",
            len(detections),
        )

    with col2:

        st.metric(
            "Average Confidence",
            f"{confidence * 100:.1f}%",
        )

    with col3:

        word_count = len(
            extracted_text.split()
        )

        st.metric(
            "Words",
            word_count,
        )

    with col4:

        st.metric(
            "Processing Time",
            f"{elapsed:.2f}s",
        )


    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------

    result_col1, result_col2 = st.columns(
        [1.1, 0.9]
    )

    with result_col1:

        st.caption(
            "Detected text regions"
        )

        st.image(
            rgb_from_bgr(
                detection_image
            ),
            use_container_width=True,
        )

    with result_col2:

        st.caption(
            "Extracted text"
        )

        if extracted_text:

            st.text_area(
                "OCR output",
                value=extracted_text,
                height=420,
                label_visibility="collapsed",
            )

        else:

            st.warning(
                "No text passed the current confidence threshold."
            )


    # ------------------------------------------------------
    # Quality Warning
    # ------------------------------------------------------

    if confidence < 0.50 and detections:

        st.warning(
            "OCR confidence is relatively low. "
            "Try a different preprocessing method, "
            "increase the image resolution, or select "
            "a more focused OCR region."
        )


    # ------------------------------------------------------
    # Detection Details
    # ------------------------------------------------------

    if detections:

        with st.expander(
            "View detection details"
        ):

            for index, detection in enumerate(
                detections,
                start=1,
            ):

                st.write(
                    f"**{index}.** "
                    f"{detection['text']} "
                    f"— "
                    f"{detection['confidence'] * 100:.1f}%"
                )


    # ------------------------------------------------------
    # Export
    # ------------------------------------------------------

    st.markdown(
        '<div class="section-title">Export</div>',
        unsafe_allow_html=True,
    )

    base_name = os.path.splitext(
        uploaded_file.name
    )[0]

    text_filename = (
        f"{base_name}_ocr.txt"
    )

    st.download_button(
        label="Download Extracted Text",
        data=extracted_text,
        file_name=text_filename,
        mime="text/plain",
        use_container_width=True,
    )


# ==========================================================
# Footer
# ==========================================================

st.markdown(
    """
    <div class="footer">
        Day 22 · OCR Document Reader
        <br>
        EasyOCR · OpenCV · Streamlit
        <br><br>
        Developed by Hadeed Jalani
    </div>
    """,
    unsafe_allow_html=True,
)