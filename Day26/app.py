import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from segmentation import (
    binary_threshold,
    adaptive_threshold,
    otsu_threshold,
    foreground_segmentation,
    watershed_segmentation,
    background_removal,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Document & Object Segmentation",
    page_icon="✂️",
    layout="wide",
)


# =========================================================
# HELPERS
# =========================================================

def cv_to_pil(image):
    """
    Convert OpenCV BGR image to PIL RGB.
    """

    if len(image.shape) == 2:
        return Image.fromarray(image)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(image)


def pil_to_cv(image):
    """
    Convert PIL RGB image to OpenCV BGR.
    """

    return cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )


def image_download_bytes(image):
    """
    Convert OpenCV image into PNG bytes.
    """

    success, encoded = cv2.imencode(
        ".png",
        image
    )

    if not success:
        return None

    return encoded.tobytes()


def prepare_display_image(image):
    """
    Make grayscale images compatible with Streamlit display.
    """

    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# =========================================================
# STYLING
# =========================================================

st.markdown(
    """
    <style>

        .main-title {
            font-size: 42px;
            font-weight: 800;
            text-align: center;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            color: #777;
            font-size: 17px;
            margin-bottom: 30px;
        }

        .metric-card {
            padding: 15px;
            border-radius: 12px;
            background: #f5f7fa;
            text-align: center;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '✂️ Document & Object Segmentation Tool'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Segment documents and foreground objects using
        OpenCV thresholding, watershed segmentation,
        and background removal techniques.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "⚙️ Segmentation Settings"
)

method = st.sidebar.selectbox(
    "Choose segmentation method",
    [
        "Binary Thresholding",
        "Adaptive Thresholding",
        "Otsu Thresholding",
        "Foreground Segmentation",
        "Watershed Segmentation",
        "Background Removal",
    ],
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Available Methods**

    • Binary: fixed global threshold

    • Adaptive: local thresholding

    • Otsu: automatic threshold selection

    • Foreground: foreground/background mask

    • Watershed: separates connected regions

    • Background Removal: OpenCV GrabCut
    """
)


# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
)


if uploaded_file is None:

    st.info(
        "Upload an image to begin segmentation."
    )

    st.stop()


# =========================================================
# READ IMAGE
# =========================================================

try:

    original_pil = Image.open(
        uploaded_file
    ).convert("RGB")

    original_cv = pil_to_cv(
        original_pil
    )

except Exception as e:

    st.error(
        f"Could not read the image: {e}"
    )

    st.stop()


# =========================================================
# ORIGINAL IMAGE
# =========================================================

st.subheader(
    "🖼️ Input Image"
)

st.image(
    original_pil,
    width="stretch",
)


# =========================================================
# IMAGE INFORMATION
# =========================================================

height, width = original_cv.shape[:2]

info1, info2, info3 = st.columns(3)

with info1:
    st.metric(
        "Width",
        f"{width}px"
    )

with info2:
    st.metric(
        "Height",
        f"{height}px"
    )

with info3:
    st.metric(
        "Selected Method",
        method
    )


# =========================================================
# PROCESS
# =========================================================

if st.button(
    "🚀 Run Segmentation",
    type="primary",
    width="stretch",
):

    start_time = time.perf_counter()

    try:

        # ---------------------------------------------
        # METHOD SELECTION
        # ---------------------------------------------

        if method == "Binary Thresholding":

            result = binary_threshold(
                original_cv
            )

            mask = result

        elif method == "Adaptive Thresholding":

            result = adaptive_threshold(
                original_cv
            )

            mask = result

        elif method == "Otsu Thresholding":

            result = otsu_threshold(
                original_cv
            )

            mask = result

        elif method == "Foreground Segmentation":

            mask = foreground_segmentation(
                original_cv
            )

            result = cv2.bitwise_and(
                original_cv,
                original_cv,
                mask=mask
            )

        elif method == "Watershed Segmentation":

            result = watershed_segmentation(
                original_cv
            )

            mask = None

        elif method == "Background Removal":

            result, mask = background_removal(
                original_cv
            )

        else:

            raise ValueError(
                "Unknown segmentation method."
            )

        # ---------------------------------------------
        # TIME
        # ---------------------------------------------

        elapsed = (
            time.perf_counter()
            - start_time
        )

        if result is None:

            raise ValueError(
                "Segmentation returned no result."
            )

        # ---------------------------------------------
        # SUCCESS
        # ---------------------------------------------

        st.success(
            f"Segmentation completed using "
            f"**{method}**."
        )

        # ---------------------------------------------
        # ORIGINAL / RESULT
        # ---------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Original Image"
            )

            st.image(
                original_pil,
                width="stretch",
            )

        with col2:

            st.subheader(
                "Segmented Result"
            )

            st.image(
                prepare_display_image(result),
                width="stretch",
            )

        # ---------------------------------------------
        # PROCESSING METRICS
        # ---------------------------------------------

        st.markdown("---")

        metric1, metric2, metric3 = st.columns(3)

        with metric1:

            st.metric(
                "Method",
                method
            )

        with metric2:

            st.metric(
                "Processing Time",
                f"{elapsed:.3f} sec"
            )

        with metric3:

            if mask is not None:

                foreground_pixels = int(
                    np.count_nonzero(mask)
                )

                total_pixels = mask.size

                percentage = (
                    foreground_pixels
                    / total_pixels
                    * 100
                )

                st.metric(
                    "Foreground Area",
                    f"{percentage:.1f}%"
                )

            else:

                st.metric(
                    "Result",
                    "Segmented"
                )

        # ---------------------------------------------
        # MASK
        # ---------------------------------------------

        if mask is not None:

            st.markdown("---")

            st.subheader(
                "🎭 Segmentation Mask"
            )

            st.image(
                mask,
                width="stretch",
            )

        # ---------------------------------------------
        # DOWNLOAD
        # ---------------------------------------------

        st.markdown("---")

        st.subheader(
            "⬇️ Download Result"
        )

        output_bytes = image_download_bytes(
            result
        )

        if output_bytes:

            safe_name = (
                method
                .lower()
                .replace(" ", "_")
                .replace("/", "_")
            )

            st.download_button(
                label=(
                    "⬇️ Download "
                    f"{method} Result"
                ),
                data=output_bytes,
                file_name=(
                    f"{safe_name}_output.png"
                ),
                mime="image/png",
                width="stretch",
            )

    except Exception as e:

        st.error(
            "Segmentation failed. "
            "Please try another image."
        )

        st.exception(e)

