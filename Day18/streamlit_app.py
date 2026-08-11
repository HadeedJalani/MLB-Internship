# ==========================================================
# MLB Summer Internship - Day 18
# Document Vision Lab
# Edge Detection & Morphological Operations
# ==========================================================

import streamlit as st
import cv2
import numpy as np
import time


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Document Vision Lab",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# Custom Styling
# ==========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #777;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #777;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# Helper Functions
# ==========================================================

def order_points(points):
    """
    Arrange four points in the order:

    top-left
    top-right
    bottom-right
    bottom-left
    """

    points = np.array(
        points,
        dtype=np.float32
    )

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    coordinate_sum = points.sum(axis=1)
    coordinate_difference = np.diff(
        points,
        axis=1
    )

    ordered[0] = points[
        np.argmin(coordinate_sum)
    ]

    ordered[2] = points[
        np.argmax(coordinate_sum)
    ]

    ordered[1] = points[
        np.argmin(coordinate_difference)
    ]

    ordered[3] = points[
        np.argmax(coordinate_difference)
    ]

    return ordered


# ==========================================================
# Perspective Correction
# ==========================================================

def perspective_transform(image, points):

    rect = order_points(points)

    top_left = rect[0]
    top_right = rect[1]
    bottom_right = rect[2]
    bottom_left = rect[3]

    width_top = np.linalg.norm(
        top_right - top_left
    )

    width_bottom = np.linalg.norm(
        bottom_right - bottom_left
    )

    max_width = max(
        int(width_top),
        int(width_bottom)
    )

    height_right = np.linalg.norm(
        bottom_right - top_right
    )

    height_left = np.linalg.norm(
        bottom_left - top_left
    )

    max_height = max(
        int(height_right),
        int(height_left)
    )

    if max_width <= 0 or max_height <= 0:
        return image.copy()

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ],
        dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(
        rect,
        destination
    )

    corrected = cv2.warpPerspective(
        image,
        matrix,
        (
            max_width,
            max_height
        )
    )

    return corrected


# ==========================================================
# Edge Detection
# ==========================================================

def detect_edges(
    image,
    method,
    canny_low,
    canny_high
):

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    blurred = cv2.GaussianBlur(
        grayscale,
        (5, 5),
        0
    )

    if method == "Canny":

        edges = cv2.Canny(
            blurred,
            canny_low,
            canny_high
        )

    elif method == "Sobel":

        sobel_x = cv2.Sobel(
            blurred,
            cv2.CV_64F,
            1,
            0,
            ksize=3
        )

        sobel_y = cv2.Sobel(
            blurred,
            cv2.CV_64F,
            0,
            1,
            ksize=3
        )

        sobel_x = cv2.convertScaleAbs(
            sobel_x
        )

        sobel_y = cv2.convertScaleAbs(
            sobel_y
        )

        edges = cv2.addWeighted(
            sobel_x,
            0.5,
            sobel_y,
            0.5,
            0
        )

    else:

        laplacian = cv2.Laplacian(
            blurred,
            cv2.CV_64F
        )

        edges = cv2.convertScaleAbs(
            laplacian
        )

    return (
        grayscale,
        blurred,
        edges
    )


# ==========================================================
# All Edge Methods
# ==========================================================

def compare_edge_methods(image):

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    blurred = cv2.GaussianBlur(
        grayscale,
        (5, 5),
        0
    )

    # Sobel

    sobel_x = cv2.Sobel(
        blurred,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        blurred,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    sobel_x = cv2.convertScaleAbs(
        sobel_x
    )

    sobel_y = cv2.convertScaleAbs(
        sobel_y
    )

    sobel = cv2.addWeighted(
        sobel_x,
        0.5,
        sobel_y,
        0.5,
        0
    )

    # Laplacian

    laplacian = cv2.Laplacian(
        blurred,
        cv2.CV_64F
    )

    laplacian = cv2.convertScaleAbs(
        laplacian
    )

    # Canny

    canny = cv2.Canny(
        blurred,
        50,
        150
    )

    return (
        sobel,
        laplacian,
        canny
    )


# ==========================================================
# Morphological Operations
# ==========================================================

def apply_morphology(
    edges,
    operation,
    kernel_size,
    iterations
):

    if operation == "None":
        return edges

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            kernel_size,
            kernel_size
        )
    )

    operation_map = {
        "Erosion": cv2.MORPH_ERODE,
        "Dilation": cv2.MORPH_DILATE,
        "Opening": cv2.MORPH_OPEN,
        "Closing": cv2.MORPH_CLOSE,
        "Morphological Gradient": cv2.MORPH_GRADIENT,
        "Top Hat": cv2.MORPH_TOPHAT,
        "Black Hat": cv2.MORPH_BLACKHAT
    }

    if operation not in operation_map:
        return edges

    return cv2.morphologyEx(
        edges,
        operation_map[operation],
        kernel,
        iterations=iterations
    )


# ==========================================================
# Find Document Boundary
# ==========================================================

def find_document_boundary(
    processed_edges,
    original_image
):

    contours, _ = cv2.findContours(
        processed_edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    image_area = (
        original_image.shape[0]
        * original_image.shape[1]
    )

    best_contour = None
    best_approximation = None

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < image_area * 0.10:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approximation) == 4:

            best_contour = contour
            best_approximation = (
                approximation
            )

            break

    result = original_image.copy()

    if best_approximation is None:

        return (
            result,
            None,
            False
        )

    points = best_approximation.reshape(
        4,
        2
    )

    ordered = order_points(
        points
    )

    polygon = ordered.astype(
        np.int32
    ).reshape(
        (-1, 1, 2)
    )

    cv2.polylines(
        result,
        [polygon],
        True,
        (0, 255, 0),
        4
    )

    area = cv2.contourArea(
        best_contour
    )

    perimeter = cv2.arcLength(
        best_contour,
        True
    )

    coverage = (
        area / image_area
    ) * 100

    # ------------------------------------------------------
    # Rectangularity
    # ------------------------------------------------------

    x, y, w, h = cv2.boundingRect(
        best_contour
    )

    bounding_area = w * h

    if bounding_area > 0:

        rectangularity = (
            area / bounding_area
        ) * 100

    else:

        rectangularity = 0

    # ------------------------------------------------------
    # Confidence
    # ------------------------------------------------------

    area_score = min(
        coverage / 70,
        1.0
    )

    rectangularity_score = min(
        rectangularity / 90,
        1.0
    )

    confidence = (
        area_score * 0.5
        +
        rectangularity_score * 0.5
    ) * 100

    confidence = min(
        confidence,
        99
    )

    return (
        result,
        {
            "points": ordered,
            "area": area,
            "perimeter": perimeter,
            "coverage": coverage,
            "rectangularity": rectangularity,
            "confidence": confidence
        },
        True
    )


# ==========================================================
# Draw Corner Points
# ==========================================================

def draw_corners(
    image,
    points
):

    result = image.copy()

    if points is None:
        return result

    for index, point in enumerate(
        points
    ):

        x, y = point.astype(
            int
        )

        cv2.circle(
            result,
            (x, y),
            8,
            (255, 0, 0),
            -1
        )

        cv2.putText(
            result,
            f"P{index + 1}",
            (
                x + 10,
                y - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    return result


# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title(
    "⚙️ Processing Controls"
)

st.sidebar.caption(
    "Configure the Computer Vision pipeline."
)

# ----------------------------------------------------------
# Presets
# ----------------------------------------------------------

preset = st.sidebar.selectbox(
    "Processing Preset",
    [
        "Custom",
        "Clean Scan",
        "Mobile Photo",
        "Shadow / Uneven Lighting",
        "Noisy Document"
    ]
)


# ==========================================================
# Preset Parameters
# ==========================================================

preset_values = {

    "Clean Scan": {
        "method": "Canny",
        "low": 40,
        "high": 120,
        "operation": "Closing",
        "kernel": 5,
        "iterations": 1
    },

    "Mobile Photo": {
        "method": "Canny",
        "low": 50,
        "high": 150,
        "operation": "Closing",
        "kernel": 5,
        "iterations": 2
    },

    "Shadow / Uneven Lighting": {
        "method": "Canny",
        "low": 30,
        "high": 100,
        "operation": "Closing",
        "kernel": 7,
        "iterations": 2
    },

    "Noisy Document": {
        "method": "Canny",
        "low": 70,
        "high": 180,
        "operation": "Opening",
        "kernel": 5,
        "iterations": 2
    }
}


if preset != "Custom":

    settings = preset_values[
        preset
    ]

    edge_method = settings[
        "method"
    ]

    canny_low = settings[
        "low"
    ]

    canny_high = settings[
        "high"
    ]

    morphology_operation = settings[
        "operation"
    ]

    kernel_size = settings[
        "kernel"
    ]

    iterations = settings[
        "iterations"
    ]

    st.sidebar.success(
        f"{preset} preset active"
    )

else:

    edge_method = st.sidebar.selectbox(
        "Edge Detection Method",
        [
            "Canny",
            "Sobel",
            "Laplacian"
        ]
    )

    canny_low = st.sidebar.slider(
        "Canny Lower Threshold",
        0,
        255,
        50
    )

    canny_high = st.sidebar.slider(
        "Canny Upper Threshold",
        0,
        255,
        150
    )

    morphology_operation = (
        st.sidebar.selectbox(
            "Morphological Operation",
            [
                "None",
                "Erosion",
                "Dilation",
                "Opening",
                "Closing",
                "Morphological Gradient",
                "Top Hat",
                "Black Hat"
            ]
        )
    )

    kernel_size = st.sidebar.slider(
        "Kernel Size",
        3,
        15,
        5,
        step=2
    )

    iterations = st.sidebar.slider(
        "Morphology Iterations",
        1,
        5,
        1
    )


# ==========================================================
# Display Options
# ==========================================================

st.sidebar.divider()

st.sidebar.subheader(
    "Display Options"
)

show_corners = st.sidebar.checkbox(
    "Show detected corners",
    value=True
)

show_comparison = st.sidebar.checkbox(
    "Show edge method comparison",
    value=True
)

enable_perspective = st.sidebar.checkbox(
    "Enable perspective correction",
    value=True
)


# ==========================================================
# Header
# ==========================================================

st.markdown(
    '<div class="main-title">'
    '📄 Document Vision Lab'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Edge Detection • Morphological Analysis • '
    'Document Boundary Detection'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Upload a document image and analyze its edges, "
    "morphological structure, and detected boundary."
)


# ==========================================================
# Upload
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload Document Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ==========================================================
# Main Processing
# ==========================================================

if uploaded_file is not None:

    start_time = time.perf_counter()

    file_bytes = np.asarray(
        bytearray(
            uploaded_file.read()
        ),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:

        st.error(
            "Unable to read the uploaded image."
        )

        st.stop()

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    original_image = image.copy()

    # ------------------------------------------------------
    # Edge Detection
    # ------------------------------------------------------

    grayscale, blurred, edges = detect_edges(
        image,
        edge_method,
        canny_low,
        canny_high
    )

    # ------------------------------------------------------
    # Morphology
    # ------------------------------------------------------

    processed_edges = apply_morphology(
        edges,
        morphology_operation,
        kernel_size,
        iterations
    )

    # ------------------------------------------------------
    # Boundary Detection
    # ------------------------------------------------------

    boundary_image, detection_data, detected = (
        find_document_boundary(
            processed_edges,
            image
        )
    )

    # ------------------------------------------------------
    # Processing Time
    # ------------------------------------------------------

    processing_time = (
        time.perf_counter()
        - start_time
    )


    # ======================================================
    # Detection Status
    # ======================================================

    if detected:

        st.success(
            "✓ Document boundary detected successfully."
        )

    else:

        st.warning(
            "No suitable four-sided document boundary "
            "was detected. Try another preset or adjust "
            "the processing parameters."
        )


    # ======================================================
    # Metrics
    # ======================================================

    st.subheader(
        "Detection Analysis"
    )

    metric_columns = st.columns(
        5
    )

    height, width = image.shape[:2]

    with metric_columns[0]:

        st.metric(
            "Resolution",
            f"{width} × {height}"
        )

    with metric_columns[1]:

        if detected:

            st.metric(
                "Confidence",
                f"{detection_data['confidence']:.1f}%"
            )

        else:

            st.metric(
                "Confidence",
                "N/A"
            )

    with metric_columns[2]:

        if detected:

            st.metric(
                "Coverage",
                f"{detection_data['coverage']:.1f}%"
            )

        else:

            st.metric(
                "Coverage",
                "N/A"
            )

    with metric_columns[3]:

        if detected:

            st.metric(
                "Corners",
                "4"
            )

        else:

            st.metric(
                "Corners",
                "0"
            )

    with metric_columns[4]:

        st.metric(
            "Processing",
            f"{processing_time:.3f}s"
        )


    # ======================================================
    # Pipeline Status
    # ======================================================

    st.subheader(
        "Processing Pipeline"
    )

    pipeline_columns = st.columns(
        6
    )

    pipeline_steps = [
        "Grayscale",
        "Gaussian Blur",
        edge_method,
        morphology_operation,
        "Contour Detection",
        "Boundary"
    ]

    for column, step in zip(
        pipeline_columns,
        pipeline_steps
    ):

        with column:

            st.success(
                f"✓ {step}"
            )


    # ======================================================
    # Main Images
    # ======================================================

    st.subheader(
        "Image Processing Results"
    )

    col1, col2 = st.columns(
        2
    )

    with col1:

        st.markdown(
            "**Original Image**"
        )

        st.image(
            original_image,
            use_container_width=True
        )

    with col2:

        st.markdown(
            "**Grayscale Image**"
        )

        st.image(
            grayscale,
            use_container_width=True
        )


    col3, col4 = st.columns(
        2
    )

    with col3:

        st.markdown(
            f"**{edge_method} Edge Detection**"
        )

        st.image(
            edges,
            use_container_width=True
        )

    with col4:

        st.markdown(
            f"**{morphology_operation}**"
        )

        st.image(
            processed_edges,
            use_container_width=True
        )


    # ======================================================
    # Edge Method Comparison
    # ======================================================

    if show_comparison:

        st.subheader(
            "Edge Detection Comparison"
        )

        sobel, laplacian, canny = (
            compare_edge_methods(
                image
            )
        )

        comparison_columns = st.columns(
            3
        )

        with comparison_columns[0]:

            st.markdown(
                "**Sobel**"
            )

            st.image(
                sobel,
                use_container_width=True
            )

        with comparison_columns[1]:

            st.markdown(
                "**Laplacian**"
            )

            st.image(
                laplacian,
                use_container_width=True
            )

        with comparison_columns[2]:

            st.markdown(
                "**Canny**"
            )

            st.image(
                canny,
                use_container_width=True
            )


    # ======================================================
    # Boundary Result
    # ======================================================

    st.subheader(
        "Detected Document Boundary"
    )

    display_boundary = boundary_image.copy()

    if (
        show_corners
        and detected
    ):

        display_boundary = draw_corners(
            display_boundary,
            detection_data["points"]
        )

    st.image(
        display_boundary,
        use_container_width=True
    )


    # ======================================================
    # Contour Information
    # ======================================================

    if detected:

        st.subheader(
            "Contour Information"
        )

        info_columns = st.columns(
            3
        )

        with info_columns[0]:

            st.metric(
                "Contour Area",
                f"{detection_data['area']:.0f} px²"
            )

        with info_columns[1]:

            st.metric(
                "Perimeter",
                f"{detection_data['perimeter']:.0f} px"
            )

        with info_columns[2]:

            st.metric(
                "Rectangularity",
                f"{detection_data['rectangularity']:.1f}%"
            )


    # ======================================================
    # Perspective Correction
    # ======================================================

    corrected_image = None

    if (
        enable_perspective
        and detected
    ):

        st.subheader(
            "Perspective Correction"
        )

        corrected_image = perspective_transform(
            image,
            detection_data["points"]
        )

        corrected_col1, corrected_col2 = (
            st.columns(2)
        )

        with corrected_col1:

            st.markdown(
                "**Detected Boundary**"
            )

            st.image(
                boundary_image,
                use_container_width=True
            )

        with corrected_col2:

            st.markdown(
                "**Perspective Corrected**"
            )

            st.image(
                corrected_image,
                use_container_width=True
            )


    # ======================================================
    # Download Center
    # ======================================================

    st.subheader(
        "Download Center"
    )

    download_columns = st.columns(
        4
    )


    def encode_image(image_data):

        if len(image_data.shape) == 2:

            image_data = cv2.cvtColor(
                image_data,
                cv2.COLOR_GRAY2RGB
            )

        bgr_image = cv2.cvtColor(
            image_data,
            cv2.COLOR_RGB2BGR
        )

        success, encoded = cv2.imencode(
            ".jpg",
            bgr_image
        )

        if success:

            return encoded.tobytes()

        return None


    edge_download = encode_image(
        edges
    )

    morphology_download = encode_image(
        processed_edges
    )

    boundary_download = encode_image(
        display_boundary
    )


    with download_columns[0]:

        if edge_download:

            st.download_button(
                "⬇️ Edge Result",
                edge_download,
                "edge_result.jpg",
                "image/jpeg"
            )


    with download_columns[1]:

        if morphology_download:

            st.download_button(
                "⬇️ Morphology",
                morphology_download,
                "morphology_result.jpg",
                "image/jpeg"
            )


    with download_columns[2]:

        if boundary_download:

            st.download_button(
                "⬇️ Boundary",
                boundary_download,
                "document_boundary.jpg",
                "image/jpeg"
            )


    with download_columns[3]:

        if corrected_image is not None:

            corrected_download = encode_image(
                corrected_image
            )

            if corrected_download:

                st.download_button(
                    "⬇️ Corrected",
                    corrected_download,
                    "perspective_corrected.jpg",
                    "image/jpeg"
                )


# ==========================================================
# Educational Information
# ==========================================================

with st.expander(
    "📚 Computer Vision Concepts"
):

    st.markdown(
        """
        ### Sobel

        Detects image intensity changes in horizontal
        and vertical directions.

        **OpenCV:** `cv2.Sobel()`

        ---

        ### Laplacian

        Detects rapid changes in intensity using the
        second derivative.

        **OpenCV:** `cv2.Laplacian()`

        ---

        ### Canny

        A multi-stage edge detector that generally
        produces cleaner and thinner edges.

        **OpenCV:** `cv2.Canny()`

        ---

        ### Morphological Operations

        **Erosion** — removes small white regions.

        **Dilation** — expands white regions.

        **Opening** — removes small noise.

        **Closing** — closes small gaps and holes.

        **Gradient** — highlights object boundaries.

        **Top Hat** — extracts bright details.

        **Black Hat** — extracts dark details.

        ---

        ### Document Detection Pipeline

        Image → Grayscale → Gaussian Blur → Edge Detection
        → Morphology → Contours → Four-Sided Boundary

        This type of preprocessing can be useful before
        OCR, document scanning, and other Computer Vision
        applications.
        """
    )


# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    "MLBench Summer Internship • Day 18 • "
    "OpenCV Computer Vision"
)