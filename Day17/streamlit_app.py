import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO


st.set_page_config(
    page_title="Day 17 - Document Image Enhancement",
    page_icon="🖼️",
    layout="wide"
)


# ==========================================================
# Helper Functions
# ==========================================================

def pil_to_cv2(image):
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(image):
    if len(image.shape) == 2:
        return Image.fromarray(image)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE
    )


def resize_image(image, scale):
    height, width = image.shape[:2]

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


def adjust_brightness_contrast(
    image,
    brightness,
    contrast
):
    return cv2.convertScaleAbs(
        image,
        alpha=contrast,
        beta=brightness
    )


def sharpen_image(image, strength):
    kernel = np.array([
        [0, -strength, 0],
        [-strength, 1 + 4 * strength, -strength],
        [0, -strength, 0]
    ])

    return cv2.filter2D(
        image,
        -1,
        kernel
    )


def detect_document(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    image_area = image.shape[0] * image.shape[1]

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < image_area * 0.20:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approx) == 4:
            return approx.reshape(4, 2)

    return None


def order_points(points):

    points = np.array(
        points,
        dtype=np.float32
    )

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    total = points.sum(axis=1)
    difference = np.diff(
        points,
        axis=1
    )

    ordered[0] = points[np.argmin(total)]
    ordered[2] = points[np.argmax(total)]
    ordered[1] = points[np.argmin(difference)]
    ordered[3] = points[np.argmax(difference)]

    return ordered


def perspective_correct(image):

    points = detect_document(image)

    if points is None:
        return image.copy(), False

    rect = order_points(points)

    top_left, top_right, bottom_right, bottom_left = rect

    width_top = np.linalg.norm(
        top_right - top_left
    )

    width_bottom = np.linalg.norm(
        bottom_right - bottom_left
    )

    max_width = int(
        max(width_top, width_bottom)
    )

    height_right = np.linalg.norm(
        bottom_right - top_right
    )

    height_left = np.linalg.norm(
        bottom_left - top_left
    )

    max_height = int(
        max(height_right, height_left)
    )

    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(
        rect,
        destination
    )

    corrected = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height)
    )

    return corrected, True


# ==========================================================
# Header
# ==========================================================

st.title("🖼️ Document Image Enhancement Tool")

st.markdown(
    """
    **Day 17 — OpenCV Image Transformations & Enhancement**

    Upload an image and experiment with common computer vision
    transformations and enhancement techniques.
    """
)

st.divider()


# ==========================================================
# Upload
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is None:

    st.info(
        "Upload a document or image to begin."
    )

    st.stop()


# ==========================================================
# Read Image
# ==========================================================

pil_image = Image.open(
    uploaded_file
).convert("RGB")

original = pil_to_cv2(
    pil_image
)

processed = original.copy()


# ==========================================================
# Sidebar Controls
# ==========================================================

st.sidebar.header("Image Operations")


operation = st.sidebar.selectbox(
    "Choose operation",
    [
        "None",
        "Grayscale",
        "Translation",
        "Rotation",
        "Scaling",
        "Perspective Correction",
        "Brightness & Contrast",
        "Gaussian Blur",
        "Median Blur",
        "Bilateral Filter",
        "Sharpening"
    ]
)


# ==========================================================
# Operation Controls
# ==========================================================

if operation == "Grayscale":

    processed = cv2.cvtColor(
        processed,
        cv2.COLOR_BGR2GRAY
    )


elif operation == "Translation":

    x_shift = st.sidebar.slider(
        "Horizontal shift",
        -500,
        500,
        0
    )

    y_shift = st.sidebar.slider(
        "Vertical shift",
        -500,
        500,
        0
    )

    matrix = np.float32([
        [1, 0, x_shift],
        [0, 1, y_shift]
    ])

    height, width = processed.shape[:2]

    processed = cv2.warpAffine(
        processed,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE
    )


elif operation == "Rotation":

    angle = st.sidebar.slider(
        "Rotation angle",
        -180,
        180,
        0
    )

    processed = rotate_image(
        processed,
        angle
    )


elif operation == "Scaling":

    scale = st.sidebar.slider(
        "Scale",
        0.25,
        2.0,
        1.0,
        0.05
    )

    processed = resize_image(
        processed,
        scale
    )


elif operation == "Perspective Correction":

    processed, detected = perspective_correct(
        processed
    )

    if detected:
        st.sidebar.success(
            "Document boundary detected."
        )
    else:
        st.sidebar.warning(
            "No clear document boundary detected."
        )


elif operation == "Brightness & Contrast":

    brightness = st.sidebar.slider(
        "Brightness",
        -100,
        100,
        0
    )

    contrast = st.sidebar.slider(
        "Contrast",
        0.5,
        3.0,
        1.0,
        0.1
    )

    processed = adjust_brightness_contrast(
        processed,
        brightness,
        contrast
    )


elif operation == "Gaussian Blur":

    kernel_size = st.sidebar.slider(
        "Kernel size",
        1,
        15,
        5,
        2
    )

    processed = cv2.GaussianBlur(
        processed,
        (kernel_size, kernel_size),
        0
    )


elif operation == "Median Blur":

    kernel_size = st.sidebar.slider(
        "Kernel size",
        1,
        15,
        5,
        2
    )

    processed = cv2.medianBlur(
        processed,
        kernel_size
    )


elif operation == "Bilateral Filter":

    diameter = st.sidebar.slider(
        "Diameter",
        1,
        15,
        9
    )

    sigma_color = st.sidebar.slider(
        "Sigma Color",
        10,
        150,
        75
    )

    sigma_space = st.sidebar.slider(
        "Sigma Space",
        10,
        150,
        75
    )

    processed = cv2.bilateralFilter(
        processed,
        diameter,
        sigma_color,
        sigma_space
    )


elif operation == "Sharpening":

    strength = st.sidebar.slider(
        "Sharpening strength",
        0.1,
        2.0,
        1.0,
        0.1
    )

    processed = sharpen_image(
        processed,
        strength
    )


# ==========================================================
# Display
# ==========================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("Original")

    st.image(
        pil_image,
        use_container_width=True
    )


with col2:

    st.subheader("Processed")

    st.image(
        cv2_to_pil(processed),
        use_container_width=True
    )


# ==========================================================
# Image Information
# ==========================================================

st.divider()

st.subheader("Image Information")

info1, info2, info3 = st.columns(3)

height, width = processed.shape[:2]

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
        "Operation",
        operation
    )


# ==========================================================
# Download
# ==========================================================

st.divider()

output_pil = cv2_to_pil(
    processed
)

buffer = BytesIO()

output_pil.save(
    buffer,
    format="PNG"
)

st.download_button(
    label="⬇️ Download Processed Image",
    data=buffer.getvalue(),
    file_name="processed_image.png",
    mime="image/png"
)


st.caption(
    "MLBench Summer Internship — Day 17 | "
    "OpenCV Image Transformations & Enhancement"
)