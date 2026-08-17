import io
import math
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Computer Vision Image Processing Studio",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Professional Styling
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        letter-spacing: -0.8px;
        margin-bottom: 8px;
    }

    .app-subtitle {
        font-size: 17px;
        line-height: 1.7;
        color: #aeb4c0;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .info-card {
        padding: 20px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        background: rgba(128, 128, 128, 0.06);
        min-height: 150px;
    }

    .info-card-title {
        font-size: 19px;
        font-weight: 650;
        margin-bottom: 10px;
    }

    .info-card-text {
        color: #aeb4c0;
        line-height: 1.6;
        font-size: 14px;
    }

    .result-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        background: rgba(128, 128, 128, 0.04);
    }

    .shape-result {
        padding: 10px 14px;
        margin: 5px 0;
        border-radius: 8px;
        background: rgba(128, 128, 128, 0.08);
    }

    .small-note {
        color: #9299a6;
        font-size: 13px;
        line-height: 1.5;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">Computer Vision Image Processing Studio</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-subtitle">
        A practical computer vision application built with Python, OpenCV,
        NumPy, PIL, and Streamlit. Upload an image, apply individual
        processing operations, or construct a custom multi-stage pipeline
        and download the final result.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Utility Functions
# ============================================================

def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert an image safely into RGB format.
    Handles grayscale, BGRA, RGBA, BGR and RGB images.
    """

    if image is None:
        raise ValueError("Image is empty.")

    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    channels = image.shape[2]

    if channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def ensure_gray(image: np.ndarray) -> np.ndarray:
    """
    Safely convert an image into grayscale.
    Prevents the OpenCV bad-channel error when the image
    is already grayscale.
    """

    if image is None:
        raise ValueError("Image is empty.")

    if len(image.shape) == 2:
        return image.copy()

    channels = image.shape[2]

    if channels == 1:
        return image[:, :, 0].copy()

    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to BGR for OpenCV operations.
    """

    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to RGB.
    """

    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_to_bytes(image: np.ndarray, format_name="PNG") -> bytes:
    """
    Convert NumPy image into downloadable bytes.
    """

    rgb = ensure_rgb(image)

    pil_image = Image.fromarray(rgb)

    buffer = io.BytesIO()

    pil_image.save(buffer, format=format_name)

    return buffer.getvalue()


def normalize_kernel(value: int) -> int:
    """
    Ensure Gaussian blur kernel is odd and valid.
    """

    value = max(3, int(value))

    if value % 2 == 0:
        value += 1

    return value


# ============================================================
# Image Processing Functions
# ============================================================

def apply_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale safely.
    """

    return ensure_gray(image)


def apply_blur(
    image: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    """
    Apply Gaussian blur.
    """

    kernel_size = normalize_kernel(kernel_size)

    if len(image.shape) == 2:
        return cv2.GaussianBlur(
            image,
            (kernel_size, kernel_size),
            0,
        )

    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0,
    )


def apply_canny(
    image: np.ndarray,
    lower: int,
    upper: int,
) -> np.ndarray:
    """
    Apply Canny edge detection.
    """

    gray = ensure_gray(image)

    if lower >= upper:
        upper = min(255, lower + 1)

    return cv2.Canny(
        gray,
        lower,
        upper,
    )


def apply_brightness_contrast(
    image: np.ndarray,
    brightness: int,
    contrast: float,
) -> np.ndarray:
    """
    Adjust brightness and contrast using OpenCV.
    """

    rgb = ensure_rgb(image)

    result = cv2.convertScaleAbs(
        rgb,
        alpha=float(contrast),
        beta=int(brightness),
    )

    return result


def apply_rotation(
    image: np.ndarray,
    angle: int,
) -> np.ndarray:
    """
    Rotate image while keeping the entire image visible.
    """

    rgb = ensure_rgb(image)

    height, width = rgb.shape[:2]

    center = (
        width / 2,
        height / 2,
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )

    cos_value = abs(matrix[0, 0])
    sin_value = abs(matrix[0, 1])

    new_width = int(
        height * sin_value +
        width * cos_value
    )

    new_height = int(
        height * cos_value +
        width * sin_value
    )

    matrix[0, 2] += (
        new_width / 2
    ) - center[0]

    matrix[1, 2] += (
        new_height / 2
    ) - center[1]

    rotated = cv2.warpAffine(
        rgb,
        matrix,
        (new_width, new_height),
        borderMode=cv2.BORDER_REPLICATE,
    )

    return rotated


def apply_sharpen(
    image: np.ndarray,
    strength: float,
) -> np.ndarray:
    """
    Sharpen image using an unsharp-mask technique.
    """

    rgb = ensure_rgb(image)

    blurred = cv2.GaussianBlur(
        rgb,
        (0, 0),
        3,
    )

    sharpened = cv2.addWeighted(
        rgb,
        1.0 + strength,
        blurred,
        -strength,
        0,
    )

    return np.clip(
        sharpened,
        0,
        255,
    ).astype(np.uint8)


def apply_flip(
    image: np.ndarray,
    direction: str,
) -> np.ndarray:
    """
    Flip image horizontally, vertically,
    or both.
    """

    rgb = ensure_rgb(image)

    if direction == "Horizontal":
        return cv2.flip(rgb, 1)

    if direction == "Vertical":
        return cv2.flip(rgb, 0)

    return cv2.flip(rgb, -1)


def apply_threshold(
    image: np.ndarray,
    threshold: int,
) -> np.ndarray:
    """
    Apply binary thresholding.
    """

    gray = ensure_gray(image)

    _, result = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY,
    )

    return result


def apply_enhancement(
    image: np.ndarray,
    sharpness: float,
    color: float,
    contrast: float,
) -> np.ndarray:
    """
    PIL-based image enhancement.
    """

    rgb = ensure_rgb(image)

    pil_image = Image.fromarray(rgb)

    pil_image = ImageEnhance.Sharpness(
        pil_image
    ).enhance(sharpness)

    pil_image = ImageEnhance.Color(
        pil_image
    ).enhance(color)

    pil_image = ImageEnhance.Contrast(
        pil_image
    ).enhance(contrast)

    return np.array(pil_image)


def apply_contours(
    image: np.ndarray,
    threshold_value: int = 127,
    min_area: int = 500,
) -> Tuple[np.ndarray, int]:
    """
    Detect and draw contours.
    """

    rgb = ensure_rgb(image)

    gray = ensure_gray(rgb)

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    _, binary = cv2.threshold(
        blurred,
        threshold_value,
        255,
        cv2.THRESH_BINARY,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    output = rgb.copy()

    valid_contours = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area >= min_area:
            valid_contours.append(contour)

    cv2.drawContours(
        output,
        valid_contours,
        -1,
        (0, 255, 0),
        2,
    )

    return output, len(valid_contours)


# ============================================================
# Shape Detection
# ============================================================

SHAPE_NAMES = {
    3: "Triangle",
    4: "Quadrilateral",
    5: "Pentagon",
    6: "Hexagon",
    7: "Heptagon",
    8: "Octagon",
    9: "Nonagon",
}


def calculate_angle(
    point_a,
    point_b,
    point_c,
) -> float:
    """
    Calculate angle ABC.
    """

    a = np.array(point_a, dtype=np.float32)
    b = np.array(point_b, dtype=np.float32)
    c = np.array(point_c, dtype=np.float32)

    vector_1 = a - b
    vector_2 = c - b

    norm_1 = np.linalg.norm(vector_1)
    norm_2 = np.linalg.norm(vector_2)

    if norm_1 == 0 or norm_2 == 0:
        return 0.0

    cosine = np.dot(
        vector_1,
        vector_2,
    ) / (
        norm_1 * norm_2
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    return math.degrees(
        math.acos(cosine)
    )


def classify_shape(
    contour: np.ndarray,
    approximation: np.ndarray,
) -> str:
    """
    Classify detected contour.

    Supports:
    Triangle
    Square
    Rectangle
    Pentagon
    Hexagon
    Heptagon
    Octagon
    Nonagon
    Circle
    Polygon
    """

    vertices = len(approximation)

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(
        contour,
        True,
    )

    if perimeter == 0:
        return "Polygon"

    circularity = (
        4
        * math.pi
        * area
        / (perimeter * perimeter)
    )

    # --------------------------------------------------------
    # Circle detection
    # --------------------------------------------------------

    if circularity >= 0.78 and vertices >= 8:
        return "Circle"

    # --------------------------------------------------------
    # Triangle
    # --------------------------------------------------------

    if vertices == 3:
        return "Triangle"

    # --------------------------------------------------------
    # Quadrilateral
    # --------------------------------------------------------

    if vertices == 4:

        points = approximation.reshape(
            4,
            2,
        )

        angles = []

        for i in range(4):

            angle = calculate_angle(
                points[
                    (i - 1) % 4
                ],
                points[i],
                points[
                    (i + 1) % 4
                ],
            )

            angles.append(angle)

        x, y, width, height = cv2.boundingRect(
            approximation
        )

        if height == 0:
            return "Quadrilateral"

        aspect_ratio = width / height

        # A square has approximately equal sides
        # and approximately right angles.

        right_angles = all(
            75 <= angle <= 105
            for angle in angles
        )

        near_square = (
            0.85 <= aspect_ratio <= 1.15
        )

        if right_angles and near_square:
            return "Square"

        if right_angles:
            return "Rectangle"

        return "Quadrilateral"

    # --------------------------------------------------------
    # Regular polygon classification
    # --------------------------------------------------------

    if 5 <= vertices <= 9:
        return SHAPE_NAMES.get(
            vertices,
            "Polygon",
        )

    # --------------------------------------------------------
    # Polygon fallback
    # --------------------------------------------------------

    return "Polygon"


def apply_shape_detection(
    image: np.ndarray,
    canny_lower: int,
    canny_upper: int,
    min_area: int,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Detect and classify geometric shapes.

    Returns:
        processed image
        list of shape information
    """

    rgb = ensure_rgb(image)

    gray = ensure_gray(rgb)

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    edges = cv2.Canny(
        blurred,
        canny_lower,
        canny_upper,
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    output = rgb.copy()

    detected_shapes = []

    image_area = rgb.shape[0] * rgb.shape[1]

    minimum_area = max(
        min_area,
        int(image_area * 0.0005),
    )

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < minimum_area:
            continue

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        if perimeter <= 0:
            continue

        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True,
        )

        shape_name = classify_shape(
            contour,
            approximation,
        )

        x, y, width, height = cv2.boundingRect(
            approximation
        )

        center_x = x + width // 2
        center_y = y + height // 2

        cv2.drawContours(
            output,
            [approximation],
            -1,
            (0, 220, 120),
            3,
        )

        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            (255, 180, 0),
            1,
        )

        label_y = max(
            25,
            y - 8,
        )

        cv2.putText(
            output,
            shape_name,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.circle(
            output,
            (center_x, center_y),
            4,
            (255, 80, 80),
            -1,
        )

        detected_shapes.append(
            {
                "shape": shape_name,
                "area": float(area),
                "vertices": int(
                    len(approximation)
                ),
                "center": (
                    center_x,
                    center_y,
                ),
            }
        )

    return output, detected_shapes


# ============================================================
# Sidebar
# ============================================================

st.sidebar.markdown(
    """
    <div style="font-size:28px;font-weight:700;">
        Processing Controls
    </div>

    <div class="small-note">
        Configure the image source, processing mode,
        operation, and processing parameters.
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")


# ============================================================
# Image Upload
# ============================================================

st.sidebar.subheader("Image Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp",
    ],
    help="Upload an image to process.",
)


# ============================================================
# Main Application Before Upload
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload an image from the sidebar to begin."
    )

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Application Overview</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">
                    Image Processing
                </div>
                <div class="info-card-text">
                    Apply classical computer vision
                    operations such as grayscale,
                    blur, edges, enhancement,
                    thresholding, and sharpening.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">
                    Interactive Controls
                </div>
                <div class="info-card-text">
                    Adjust brightness, contrast,
                    blur strength, edge thresholds,
                    rotation, sharpening, and
                    enhancement parameters.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">
                    Pipeline Processing
                </div>
                <div class="info-card-text">
                    Combine multiple image processing
                    operations and execute them
                    sequentially as a custom pipeline.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:

        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">
                    Shape Detection
                </div>
                <div class="info-card-text">
                    Detect geometric objects including
                    triangles, squares, rectangles,
                    polygons, and circles.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.stop()


# ============================================================
# Read Uploaded Image
# ============================================================

file_bytes = uploaded_file.read()

pil_image = Image.open(
    io.BytesIO(file_bytes)
).convert("RGB")

original_image = np.array(
    pil_image
)


# ============================================================
# Processing Mode
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader("Processing Mode")

processing_mode = st.sidebar.radio(
    "Select processing mode",
    [
        "Single Operation",
        "Chain Multiple Filters",
    ],
)


# ============================================================
# Single Operation Controls
# ============================================================

operation = None

if processing_mode == "Single Operation":

    st.sidebar.markdown("---")

    st.sidebar.subheader("Operation")

    operation = st.sidebar.selectbox(
        "Select an operation",
        [
            "Grayscale",
            "Gaussian Blur",
            "Canny Edge Detection",
            "Image Rotation",
            "Brightness & Contrast",
            "Image Enhancement",
            "Sharpening",
            "Flip",
            "Thresholding",
            "Contour Detection",
            "Shape Detection",
        ],
    )


# ============================================================
# Shared Parameter Defaults
# ============================================================

blur_kernel = 7
canny_lower = 50
canny_upper = 150
rotation_angle = 90
brightness = 0
contrast = 1.0
sharpness = 1.5
color_enhancement = 1.2
enhancement_contrast = 1.2
sharpen_strength = 1.0
flip_direction = "Horizontal"
threshold_value = 127
min_area = 500


# ============================================================
# Interactive Parameter Controls
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader("Processing Parameters")


if (
    operation == "Gaussian Blur"
    or processing_mode == "Chain Multiple Filters"
):

    blur_kernel = st.sidebar.slider(
        "Blur Kernel",
        min_value=3,
        max_value=31,
        value=7,
        step=2,
    )


if (
    operation == "Canny Edge Detection"
    or operation == "Shape Detection"
    or processing_mode == "Chain Multiple Filters"
):

    canny_lower = st.sidebar.slider(
        "Canny Lower Threshold",
        min_value=0,
        max_value=255,
        value=50,
    )

    canny_upper = st.sidebar.slider(
        "Canny Upper Threshold",
        min_value=1,
        max_value=255,
        value=150,
    )


if operation == "Image Rotation":

    rotation_angle = st.sidebar.slider(
        "Rotation Angle",
        min_value=-180,
        max_value=180,
        value=90,
        step=1,
    )


if operation == "Brightness & Contrast":

    brightness = st.sidebar.slider(
        "Brightness",
        min_value=-100,
        max_value=100,
        value=0,
    )

    contrast = st.sidebar.slider(
        "Contrast",
        min_value=0.2,
        max_value=3.0,
        value=1.0,
        step=0.1,
    )


if operation == "Image Enhancement":

    sharpness = st.sidebar.slider(
        "Sharpness",
        min_value=0.0,
        max_value=5.0,
        value=1.5,
        step=0.1,
    )

    color_enhancement = st.sidebar.slider(
        "Color Enhancement",
        min_value=0.0,
        max_value=3.0,
        value=1.2,
        step=0.1,
    )

    enhancement_contrast = st.sidebar.slider(
        "Contrast Enhancement",
        min_value=0.0,
        max_value=3.0,
        value=1.2,
        step=0.1,
    )


if operation == "Sharpening":

    sharpen_strength = st.sidebar.slider(
        "Sharpening Strength",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
    )


if operation == "Flip":

    flip_direction = st.sidebar.selectbox(
        "Flip Direction",
        [
            "Horizontal",
            "Vertical",
            "Both",
        ],
    )


if operation == "Thresholding":

    threshold_value = st.sidebar.slider(
        "Threshold",
        min_value=0,
        max_value=255,
        value=127,
    )


if operation in [
    "Contour Detection",
    "Shape Detection",
]:

    min_area = st.sidebar.slider(
        "Minimum Contour Area",
        min_value=50,
        max_value=10000,
        value=500,
        step=50,
    )


# ============================================================
# Pipeline Mode
# ============================================================

pipeline_operations = []

if processing_mode == "Chain Multiple Filters":

    st.sidebar.markdown("---")

    st.sidebar.subheader("Pipeline Builder")

    available_operations = [
        "Grayscale",
        "Gaussian Blur",
        "Canny Edge Detection",
        "Image Rotation",
        "Brightness & Contrast",
        "Image Enhancement",
        "Sharpening",
        "Flip",
        "Thresholding",
        "Contour Detection",
        "Shape Detection",
    ]

    pipeline_operations = st.sidebar.multiselect(
        "Select operations in execution order",
        available_operations,
        default=[
            "Brightness & Contrast",
            "Gaussian Blur",
            "Canny Edge Detection",
        ],
    )

    st.sidebar.markdown(
        """
        <div class="small-note">
            Operations are applied sequentially from
            top to bottom in the selected order.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Main Image Information
# ============================================================

height, width = original_image.shape[:2]

st.markdown(
    '<div class="section-title">Image Information</div>',
    unsafe_allow_html=True,
)

info_col1, info_col2, info_col3, info_col4 = st.columns(4)

with info_col1:
    st.metric(
        "Width",
        f"{width} px",
    )

with info_col2:
    st.metric(
        "Height",
        f"{height} px",
    )

with info_col3:
    st.metric(
        "Channels",
        "3",
    )

with info_col4:
    st.metric(
        "Format",
        uploaded_file.type.split("/")[-1].upper(),
    )


# ============================================================
# Process Image
# ============================================================

processed_image = original_image.copy()

shape_results = []

processing_steps = []


def execute_operation(
    image: np.ndarray,
    selected_operation: str,
) -> np.ndarray:

    global shape_results

    if selected_operation == "Grayscale":

        processing_steps.append(
            "Grayscale"
        )

        return apply_grayscale(image)

    if selected_operation == "Gaussian Blur":

        processing_steps.append(
            "Gaussian Blur"
        )

        return apply_blur(
            image,
            blur_kernel,
        )

    if selected_operation == "Canny Edge Detection":

        processing_steps.append(
            "Canny Edge Detection"
        )

        return apply_canny(
            image,
            canny_lower,
            canny_upper,
        )

    if selected_operation == "Image Rotation":

        processing_steps.append(
            "Image Rotation"
        )

        return apply_rotation(
            image,
            rotation_angle,
        )

    if selected_operation == "Brightness & Contrast":

        processing_steps.append(
            "Brightness & Contrast"
        )

        return apply_brightness_contrast(
            image,
            brightness,
            contrast,
        )

    if selected_operation == "Image Enhancement":

        processing_steps.append(
            "Image Enhancement"
        )

        return apply_enhancement(
            image,
            sharpness,
            color_enhancement,
            enhancement_contrast,
        )

    if selected_operation == "Sharpening":

        processing_steps.append(
            "Sharpening"
        )

        return apply_sharpen(
            image,
            sharpen_strength,
        )

    if selected_operation == "Flip":

        processing_steps.append(
            "Flip"
        )

        return apply_flip(
            image,
            flip_direction,
        )

    if selected_operation == "Thresholding":

        processing_steps.append(
            "Thresholding"
        )

        return apply_threshold(
            image,
            threshold_value,
        )

    if selected_operation == "Contour Detection":

        processing_steps.append(
            "Contour Detection"
        )

        result, _ = apply_contours(
            image,
            threshold_value=127,
            min_area=min_area,
        )

        return result

    if selected_operation == "Shape Detection":

        processing_steps.append(
            "Shape Detection"
        )

        result, detected = apply_shape_detection(
            image,
            canny_lower,
            canny_upper,
            min_area,
        )

        shape_results.extend(
            detected
        )

        return result

    return image


# ============================================================
# Run Processing
# ============================================================

if processing_mode == "Single Operation":

    processed_image = execute_operation(
        processed_image,
        operation,
    )

else:

    for selected_operation in pipeline_operations:

        processed_image = execute_operation(
            processed_image,
            selected_operation,
        )


# ============================================================
# Results
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">Processing Results</div>',
    unsafe_allow_html=True,
)

if processing_steps:

    st.markdown(
        f"""
        <div class="small-note">
            Processing pipeline:
            {' → '.join(processing_steps)}
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.info(
        "Select at least one processing operation."
    )


# ============================================================
# Original and Processed Images
# ============================================================

result_col1, result_col2 = st.columns(2)

with result_col1:

    st.subheader("Original Image")

    st.image(
        original_image,
        use_container_width=True,
    )


with result_col2:

    st.subheader("Processed Image")

    st.image(
        processed_image,
        use_container_width=True,
    )


# ============================================================
# Shape Detection Results
# ============================================================

if shape_results:

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Shape Detection Results</div>',
        unsafe_allow_html=True,
    )

    # Count shapes

    shape_counts = {}

    for item in shape_results:

        name = item["shape"]

        shape_counts[name] = (
            shape_counts.get(name, 0)
            + 1
        )

    count_columns = st.columns(
        min(
            4,
            max(
                1,
                len(shape_counts),
            ),
        )
    )

    for index, (
        shape_name,
        count,
    ) in enumerate(
        shape_counts.items()
    ):

        with count_columns[
            index % len(count_columns)
        ]:

            st.metric(
                shape_name,
                count,
            )

    st.markdown(
        f"""
        <div class="small-note">
            Total detected objects:
            <strong>{len(shape_results)}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Download Result
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">Export Result</div>',
    unsafe_allow_html=True,
)

download_bytes = image_to_bytes(
    processed_image,
    "PNG",
)

base_name = (
    uploaded_file.name
    .rsplit(".", 1)[0]
)

output_name = (
    f"{base_name}_processed.png"
)

st.download_button(
    label="Download Processed Image",
    data=download_bytes,
    file_name=output_name,
    mime="image/png",
    use_container_width=True,
)


# ============================================================
# Processing Details
# ============================================================

if processing_steps:

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Processing Details</div>',
        unsafe_allow_html=True,
    )

    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:

        st.markdown(
            """
            <div class="result-card">
                <strong>Operations Applied</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for index, step in enumerate(
            processing_steps,
            start=1,
        ):

            st.write(
                f"{index}. {step}"
            )

    with detail_col2:

        st.markdown(
            """
            <div class="result-card">
                <strong>Application Capabilities</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            "• Classical image processing"
        )

        st.write(
            "• Interactive parameter tuning"
        )

        st.write(
            "• Sequential processing pipeline"
        )

        st.write(
            "• Geometric shape detection"
        )

        st.write(
            "• Processed image export"
        )