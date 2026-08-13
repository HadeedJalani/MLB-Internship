# ==========================================================
# MLB Summer Internship - Day 19
# Advanced Shape Detection & Contour Analysis
# ==========================================================

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Shape Detection",
    page_icon="🔷",
    layout="wide"
)


# ==========================================================
# SUPPORTED SHAPES
# ==========================================================

SUPPORTED_SHAPES = [
    "Triangle",
    "Square",
    "Rectangle",
    "Pentagon",
    "Hexagon",
    "Heptagon",
    "Octagon",
    "Nonagon",
    "Circle",
    "Polygon"
]


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .shape-card {
        padding: 12px;
        border-radius: 10px;
        background-color: #151922;
        border: 1px solid #2a2f3a;
        text-align: center;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# TITLE
# ==========================================================

st.markdown(
    '<div class="main-title">'
    '🔷 Advanced Shape Detection System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'OpenCV contour analysis, geometric shape classification, '
    'measurements, confidence scoring, and visual detection.'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# IMAGE CONVERSION
# ==========================================================

def pil_to_cv(image):

    image = np.array(image)

    if image.ndim == 2:

        return cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    if image.shape[2] == 4:

        return cv2.cvtColor(
            image,
            cv2.COLOR_RGBA2BGR
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )


def cv_to_rgb(image):

    if len(image.shape) == 2:

        return image

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ==========================================================
# PREPROCESSING
# ==========================================================

def preprocess_image(
    image,
    blur_size,
    threshold_method,
    morphology,
    kernel_size
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    if blur_size % 2 == 0:

        blur_size += 1

    blurred = cv2.GaussianBlur(
        gray,
        (blur_size, blur_size),
        0
    )

    # ------------------------------------------------------
    # Threshold
    # ------------------------------------------------------

    if threshold_method == "Otsu":

        _, binary = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV +
            cv2.THRESH_OTSU
        )

    elif threshold_method == "Adaptive":

        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            5
        )

    elif threshold_method == "Binary":

        _, binary = cv2.threshold(
            blurred,
            127,
            255,
            cv2.THRESH_BINARY_INV
        )

    else:

        binary = cv2.Canny(
            blurred,
            50,
            150
        )

    # ------------------------------------------------------
    # Morphology
    # ------------------------------------------------------

    if morphology != "None":

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                kernel_size,
                kernel_size
            )
        )

        if morphology == "Opening":

            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                kernel
            )

        elif morphology == "Closing":

            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                kernel
            )

        elif morphology == "Opening + Closing":

            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                kernel
            )

            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                kernel
            )

    return gray, blurred, binary


# ==========================================================
# GEOMETRY FUNCTIONS
# ==========================================================

def calculate_circularity(
    area,
    perimeter
):

    if perimeter <= 0:

        return 0.0

    return float(
        4 *
        np.pi *
        area /
        (perimeter * perimeter)
    )


def calculate_solidity(
    contour
):

    area = cv2.contourArea(
        contour
    )

    hull = cv2.convexHull(
        contour
    )

    hull_area = cv2.contourArea(
        hull
    )

    if hull_area <= 0:

        return 0.0

    return float(
        area / hull_area
    )


def calculate_extent(
    contour
):

    area = cv2.contourArea(
        contour
    )

    x, y, width, height = cv2.boundingRect(
        contour
    )

    box_area = width * height

    if box_area <= 0:

        return 0.0

    return float(
        area / box_area
    )


# ==========================================================
# SHAPE CLASSIFICATION
# ==========================================================

def classify_shape(
    contour
):

    perimeter = cv2.arcLength(
        contour,
        True
    )

    area = cv2.contourArea(
        contour
    )

    if perimeter <= 0 or area <= 0:

        return "Polygon", 0.0, 0

    approximation = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    vertices = len(
        approximation
    )

    x, y, width, height = cv2.boundingRect(
        approximation
    )

    if height <= 0:

        return "Polygon", 0.0, vertices

    aspect_ratio = (
        width /
        float(height)
    )

    circularity = calculate_circularity(
        area,
        perimeter
    )

    solidity = calculate_solidity(
        contour
    )

    extent = calculate_extent(
        contour
    )

    # ------------------------------------------------------
    # Circle
    #
    # Check this before high-vertex polygons.
    # ------------------------------------------------------

    if (
        circularity >= 0.88
        and
        solidity >= 0.94
    ):

        confidence = min(
            0.99,
            circularity
        )

        return (
            "Circle",
            confidence,
            vertices
        )

    # ------------------------------------------------------
    # Triangle
    # ------------------------------------------------------

    if vertices == 3:

        if solidity >= 0.85:

            return (
                "Triangle",
                0.95,
                vertices
            )

        return (
            "Triangle",
            0.80,
            vertices
        )

    # ------------------------------------------------------
    # Four-sided shapes
    # ------------------------------------------------------

    if vertices == 4:

        # Rectangularity check
        if extent >= 0.60:

            # Square
            if (
                0.90 <= aspect_ratio <= 1.10
            ):

                return (
                    "Square",
                    0.95,
                    vertices
                )

            # Rectangle
            return (
                "Rectangle",
                0.95,
                vertices
            )

        return (
            "Rectangle",
            0.75,
            vertices
        )

    # ------------------------------------------------------
    # Pentagon
    # ------------------------------------------------------

    if vertices == 5:

        return (
            "Pentagon",
            0.92,
            vertices
        )

    # ------------------------------------------------------
    # Hexagon
    # ------------------------------------------------------

    if vertices == 6:

        return (
            "Hexagon",
            0.92,
            vertices
        )

    # ------------------------------------------------------
    # Heptagon
    # ------------------------------------------------------

    if vertices == 7:

        return (
            "Heptagon",
            0.90,
            vertices
        )

    # ------------------------------------------------------
    # Octagon
    # ------------------------------------------------------

    if vertices == 8:

        return (
            "Octagon",
            0.90,
            vertices
        )

    # ------------------------------------------------------
    # Nonagon
    # ------------------------------------------------------

    if vertices == 9:

        return (
            "Nonagon",
            0.88,
            vertices
        )

    # ------------------------------------------------------
    # Generic polygon
    # ------------------------------------------------------

    return (
        "Polygon",
        0.80,
        vertices
    )


# ==========================================================
# CONTOUR DETECTION
# ==========================================================

def detect_contours(
    binary,
    min_area,
    max_area_ratio
):

    image_area = (
        binary.shape[0] *
        binary.shape[1]
    )

    max_area = (
        image_area *
        max_area_ratio
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < min_area:

            continue

        if area > max_area:

            continue

        x, y, width, height = cv2.boundingRect(
            contour
        )

        if width < 20 or height < 20:

            continue

        # Reject extremely thin artifacts

        aspect = (
            width /
            float(height)
        )

        if aspect > 10 or aspect < 0.10:

            continue

        valid.append(
            contour
        )

    valid.sort(
        key=cv2.contourArea,
        reverse=True
    )

    return valid


# ==========================================================
# ANALYZE CONTOURS
# ==========================================================

def analyze_contours(
    contours
):

    objects = []

    for index, contour in enumerate(
        contours,
        start=1
    ):

        area = cv2.contourArea(
            contour
        )

        perimeter = cv2.arcLength(
            contour,
            True
        )

        x, y, width, height = cv2.boundingRect(
            contour
        )

        shape, confidence, vertices = (
            classify_shape(
                contour
            )
        )

        circularity = calculate_circularity(
            area,
            perimeter
        )

        solidity = calculate_solidity(
            contour
        )

        extent = calculate_extent(
            contour
        )

        (
            center_x,
            center_y
        ), radius = cv2.minEnclosingCircle(
            contour
        )

        objects.append({

            "Object": index,

            "Shape": shape,

            "Confidence": round(
                confidence * 100,
                1
            ),

            "Area": round(
                float(area),
                2
            ),

            "Perimeter": round(
                float(perimeter),
                2
            ),

            "X": int(x),

            "Y": int(y),

            "Width": int(width),

            "Height": int(height),

            "Circularity": round(
                circularity,
                3
            ),

            "Solidity": round(
                solidity,
                3
            ),

            "Extent": round(
                extent,
                3
            ),

            "Vertices": vertices,

            "Circle Radius": round(
                float(radius),
                2
            ),

            "Contour": contour
        })

    return objects


# ==========================================================
# DRAW CONTOURS
# ==========================================================

def draw_contours(
    image,
    contours
):

    result = image.copy()

    cv2.drawContours(
        result,
        contours,
        -1,
        (0, 255, 0),
        2
    )

    return result


# ==========================================================
# DRAW LABELED SHAPES
# ==========================================================

def draw_labeled_shapes(
    image,
    objects
):

    result = image.copy()

    for obj in objects:

        contour = obj["Contour"]

        shape = obj["Shape"]

        object_id = obj["Object"]

        confidence = obj["Confidence"]

        x = obj["X"]

        y = obj["Y"]

        width = obj["Width"]

        height = obj["Height"]

        # --------------------------------------------------
        # Shape color
        # --------------------------------------------------

        if shape == "Circle":

            color = (
                255,
                0,
                255
            )

        elif shape == "Triangle":

            color = (
                0,
                255,
                255
            )

        elif shape == "Square":

            color = (
                0,
                255,
                0
            )

        elif shape == "Rectangle":

            color = (
                255,
                0,
                0
            )

        elif shape in [
            "Pentagon",
            "Hexagon",
            "Heptagon",
            "Octagon",
            "Nonagon"
        ]:

            color = (
                0,
                165,
                255
            )

        else:

            color = (
                180,
                180,
                180
            )

        # --------------------------------------------------
        # Contour
        # --------------------------------------------------

        cv2.drawContours(
            result,
            [contour],
            -1,
            color,
            3
        )

        # --------------------------------------------------
        # Bounding rectangle
        # --------------------------------------------------

        cv2.rectangle(
            result,
            (x, y),
            (
                x + width,
                y + height
            ),
            color,
            2
        )

        # --------------------------------------------------
        # Main label
        # --------------------------------------------------

        label = (
            f"#{object_id} {shape}"
        )

        label_y = max(
            y - 10,
            25
        )

        cv2.putText(
            result,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )

        # --------------------------------------------------
        # Confidence
        # --------------------------------------------------

        confidence_label = (
            f"{confidence:.0f}%"
        )

        cv2.putText(
            result,
            confidence_label,
            (
                x,
                label_y + 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA
        )

        # --------------------------------------------------
        # Measurements
        # --------------------------------------------------

        measurement = (
            f"A:{obj['Area']:.0f} "
            f"P:{obj['Perimeter']:.0f}"
        )

        measurement_y = min(
            y + height + 20,
            result.shape[0] - 10
        )

        cv2.putText(
            result,
            measurement,
            (x, measurement_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return result


# ==========================================================
# IMAGE BYTES
# ==========================================================

def image_to_bytes(
    image
):

    success, encoded = cv2.imencode(
        ".png",
        image
    )

    if not success:

        return None

    return encoded.tobytes()


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header(
        "⚙️ Detection Settings"
    )

    threshold_method = st.selectbox(
        "Threshold Method",
        [
            "Otsu",
            "Adaptive",
            "Binary",
            "Canny"
        ]
    )

    blur_size = st.slider(
        "Gaussian Blur",
        3,
        11,
        5,
        step=2
    )

    morphology = st.selectbox(
        "Morphological Processing",
        [
            "None",
            "Opening",
            "Closing",
            "Opening + Closing"
        ],
        index=2
    )

    kernel_size = st.slider(
        "Morphology Kernel",
        3,
        9,
        3,
        step=2
    )

    min_area = st.slider(
        "Minimum Object Area",
        50,
        10000,
        500,
        step=50
    )

    max_area_ratio = st.slider(
        "Maximum Object Area Ratio",
        0.20,
        0.95,
        0.85,
        step=0.05
    )

    st.divider()

    st.subheader(
        "🎯 Supported Shapes"
    )

    for shape in SUPPORTED_SHAPES:

        st.write(
            f"• {shape}"
        )


# ==========================================================
# UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "📤 Upload an image containing shapes",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ==========================================================
# MAIN PROCESSING
# ==========================================================

if uploaded_file is not None:

    pil_image = Image.open(
        uploaded_file
    ).convert("RGB")

    image = pil_to_cv(
        pil_image
    )

    gray, blurred, binary = preprocess_image(
        image,
        blur_size,
        threshold_method,
        morphology,
        kernel_size
    )

    contours = detect_contours(
        binary,
        min_area,
        max_area_ratio
    )

    objects = analyze_contours(
        contours
    )

    contours_image = draw_contours(
        image,
        contours
    )

    labeled_image = draw_labeled_shapes(
        image,
        objects
    )

    # ======================================================
    # RESULTS
    # ======================================================

    st.markdown(
        '<div class="section-title">'
        '🖼️ Detection Results'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader(
            "Original"
        )

        st.image(
            cv_to_rgb(image),
            use_container_width=True
        )

    with col2:

        st.subheader(
            "Contours"
        )

        st.image(
            cv_to_rgb(contours_image),
            use_container_width=True
        )

    with col3:

        st.subheader(
            "Shape Detection"
        )

        st.image(
            cv_to_rgb(labeled_image),
            use_container_width=True
        )

    # ======================================================
    # SUMMARY
    # ======================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Detection Summary'
        '</div>',
        unsafe_allow_html=True
    )

    counts = {
        shape: 0
        for shape in SUPPORTED_SHAPES
    }

    for obj in objects:

        shape = obj["Shape"]

        if shape in counts:

            counts[shape] += 1

    # First row
    row1 = st.columns(5)

    for column, shape in zip(
        row1,
        SUPPORTED_SHAPES[:5]
    ):

        with column:

            st.metric(
                shape,
                counts[shape]
            )

    # Second row
    row2 = st.columns(5)

    for column, shape in zip(
        row2,
        SUPPORTED_SHAPES[5:]
    ):

        with column:

            st.metric(
                shape,
                counts[shape]
            )

    st.metric(
        "🔷 Total Objects",
        len(objects)
    )

    # ======================================================
    # PREPROCESSING
    # ======================================================

    with st.expander(
        "🔬 View preprocessing stages"
    ):

        p1, p2, p3 = st.columns(3)

        with p1:

            st.image(
                gray,
                caption="Grayscale",
                use_container_width=True
            )

        with p2:

            st.image(
                blurred,
                caption="Gaussian Blur",
                use_container_width=True
            )

        with p3:

            st.image(
                binary,
                caption="Thresholded Image",
                use_container_width=True
            )

    # ======================================================
    # DETAILS TABLE
    # ======================================================

    st.markdown(
        '<div class="section-title">'
        '🔍 Object Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    if objects:

        table_data = []

        for obj in objects:

            table_data.append({

                "Object":
                    obj["Object"],

                "Shape":
                    obj["Shape"],

                "Confidence":
                    f"{obj['Confidence']:.1f}%",

                "Area":
                    obj["Area"],

                "Perimeter":
                    obj["Perimeter"],

                "Width":
                    obj["Width"],

                "Height":
                    obj["Height"],

                "Circularity":
                    obj["Circularity"],

                "Solidity":
                    obj["Solidity"],

                "Vertices":
                    obj["Vertices"]

            })

        df = pd.DataFrame(
            table_data
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "No objects detected. "
            "Try lowering the minimum area."
        )

    # ======================================================
    # SHAPE DISTRIBUTION
    # ======================================================

    if objects:

        st.markdown(
            '<div class="section-title">'
            '📈 Shape Distribution'
            '</div>',
            unsafe_allow_html=True
        )

        chart_data = pd.DataFrame({

            "Shape": SUPPORTED_SHAPES,

            "Count": [
                counts[shape]
                for shape in SUPPORTED_SHAPES
            ]

        })

        st.bar_chart(
            chart_data.set_index(
                "Shape"
            )
        )

    # ======================================================
    # DOWNLOADS
    # ======================================================

    st.markdown(
        '<div class="section-title">'
        '📥 Export'
        '</div>',
        unsafe_allow_html=True
    )

    d1, d2, d3 = st.columns(3)

    with d1:

        st.download_button(
            "⬇️ Binary Image",
            image_to_bytes(binary),
            "binary_image.png",
            "image/png"
        )

    with d2:

        st.download_button(
            "⬇️ Contour Result",
            image_to_bytes(contours_image),
            "contours.png",
            "image/png"
        )

    with d3:

        st.download_button(
            "⬇️ Final Shape Result",
            image_to_bytes(labeled_image),
            "shape_detection.png",
            "image/png"
        )

    # ------------------------------------------------------
    # CSV
    # ------------------------------------------------------

    if objects:

        csv_data = pd.DataFrame(
            table_data
        ).to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📄 Download Analysis CSV",
            csv_data,
            "shape_analysis.csv",
            "text/csv"
        )


# ==========================================================
# EMPTY STATE
# ==========================================================

else:

    st.info(
        "👆 Upload an image to begin."
    )

    st.markdown(
        """
        ### 🔷 Detectable Shapes

        This application is designed to detect:

        **Triangle · Square · Rectangle · Pentagon ·
        Hexagon · Heptagon · Octagon · Nonagon ·
        Circle · Polygon**

        ### 📐 Measurements

        For each detected object, the application calculates:

        - Area
        - Perimeter
        - Bounding rectangle
        - Width and height
        - Circularity
        - Solidity
        - Number of vertices
        - Detection confidence

        ### 🚀 Processing Pipeline

        **Image → Grayscale → Gaussian Blur → Threshold →
        Morphological Processing → External Contours →
        Shape Classification → Measurements → Visualization**
        """
    )