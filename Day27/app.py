
import io
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Object Detection",
    page_icon="🎯",
    layout="wide",
)


# =========================================================
# CONSTANTS
# =========================================================

MODEL_NAME = "yolo11n.pt"


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource(show_spinner="Loading YOLO11n model...")
def load_model():
    """
    Load the YOLO model once and cache it.

    CPU-only inference keeps the app compatible with
    Streamlit Cloud.
    """
    return YOLO(MODEL_NAME)


model = load_model()


# =========================================================
# COLOR GENERATION
# =========================================================

def get_class_color(class_id):
    """
    Generate a deterministic color for each class.
    """

    rng = np.random.default_rng(class_id * 12345)

    color = rng.integers(
        60,
        230,
        size=3,
    ).tolist()

    return tuple(
        int(value)
        for value in color
    )


# =========================================================
# IMAGE CONVERSION
# =========================================================

def uploaded_file_to_cv2(uploaded_file):
    """
    Convert Streamlit uploaded image to OpenCV BGR format.
    """

    image_bytes = uploaded_file.getvalue()

    array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            "Could not decode the uploaded image."
        )

    return image


# =========================================================
# CUSTOM DRAWING
# =========================================================

def draw_detections(image, result):
    """
    Draw bounding boxes, class labels and confidence
    scores with a different color for every class.
    """

    output = image.copy()

    detections = []

    if result.boxes is None:
        return output, detections

    for box in result.boxes:

        class_id = int(
            box.cls[0].item()
        )

        confidence = float(
            box.conf[0].item()
        )

        class_name = result.names[class_id]

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist(),
        )

        color = get_class_color(
            class_id
        )

        # ---------------------------------------------
        # Bounding box
        # ---------------------------------------------

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            3,
        )

        # ---------------------------------------------
        # Label
        # ---------------------------------------------

        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = 0.65
        thickness = 2

        (
            text_width,
            text_height,
        ), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )

        label_y1 = max(
            0,
            y1 - text_height - baseline - 8,
        )

        label_y2 = y1

        cv2.rectangle(
            output,
            (x1, label_y1),
            (
                x1 + text_width + 10,
                label_y2,
            ),
            color,
            -1,
        )

        cv2.putText(
            output,
            label,
            (
                x1 + 5,
                label_y2 - 5,
            ),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        detections.append(
            {
                "class": class_name,
                "confidence": confidence,
                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2,
                ),
            }
        )

    return output, detections


# =========================================================
# IMAGE ENCODING
# =========================================================

def image_to_bytes(image):
    """
    Convert OpenCV image to PNG bytes.
    """

    success, encoded = cv2.imencode(
        ".png",
        image,
    )

    if not success:
        return None

    return encoded.tobytes()


# =========================================================
# VIDEO PROCESSING
# =========================================================

def process_video(
    input_path,
    output_path,
    confidence,
):
    """
    Process a video frame-by-frame using YOLO.
    """

    capture = cv2.VideoCapture(
        input_path
    )

    if not capture.isOpened():
        raise ValueError(
            "Could not open the uploaded video."
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()

        raise ValueError(
            "Could not create output video."
        )

    progress_bar = st.progress(
        0,
        text="Processing video...",
    )

    frame_number = 0

    while True:

        success, frame = capture.read()

        if not success:
            break

        frame_number += 1

        results = model.predict(
            source=frame,
            conf=confidence,
            device="cpu",
            verbose=False,
        )

        result = results[0]

        annotated_frame, _ = draw_detections(
            frame,
            result,
        )

        writer.write(
            annotated_frame
        )

        if total_frames > 0:

            progress = min(
                frame_number / total_frames,
                1.0,
            )

            progress_bar.progress(
                progress,
                text=(
                    f"Processing frame "
                    f"{frame_number}/{total_frames}"
                ),
            )

    capture.release()
    writer.release()

    progress_bar.empty()


# =========================================================
# STYLING
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 44px;
        font-weight: 800;
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
    """
    <div class="main-title">
        🎯 Smart Object Detection
    </div>

    <div class="subtitle">
        Real-time object detection powered by
        YOLO11n, OpenCV and Streamlit.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "⚙️ Detection Settings"
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05,
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ### 🤖 Model

    **YOLO11n**

    Lightweight YOLO model designed for
    fast inference.

    ### 💻 Runtime

    CPU-compatible configuration for
    Streamlit Cloud deployment.
    """
)


# =========================================================
# INPUT TYPE
# =========================================================

input_type = st.radio(
    "Choose input type",
    [
        "🖼️ Image",
        "🎥 Video",
    ],
    horizontal=True,
)


# =========================================================
# IMAGE MODE
# =========================================================

if input_type == "🖼️ Image":

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
            "Upload an image to begin object detection."
        )

        st.stop()

    try:

        original_image = uploaded_file_to_cv2(
            uploaded_file
        )

    except Exception as error:

        st.error(
            f"Could not read image: {error}"
        )

        st.stop()

    # =====================================================
    # INPUT / OUTPUT PREVIEW
    # =====================================================

    image_input_col, image_info_col = st.columns(
        [2, 1]
    )

    with image_input_col:

        st.subheader(
            "🖼️ Input Image"
        )

        st.image(
            cv2.cvtColor(
                original_image,
                cv2.COLOR_BGR2RGB,
            ),
            width="stretch",
        )

    with image_info_col:

        st.subheader(
            "📋 Image Information"
        )

        height, width = original_image.shape[:2]

        st.metric(
            "Width",
            f"{width}px",
        )

        st.metric(
            "Height",
            f"{height}px",
        )

        st.metric(
            "Confidence",
            f"{confidence:.0%}",
        )

    if st.button(
        "🚀 Run Object Detection",
        type="primary",
        width="stretch",
    ):

        with st.spinner(
            "Running YOLO object detection..."
        ):

            results = model.predict(
                source=original_image,
                conf=confidence,
                device="cpu",
                verbose=False,
            )

            result = results[0]

            output_image, detections = (
                draw_detections(
                    original_image,
                    result,
                )
            )

        st.success(
            "Object detection completed successfully."
        )

        # ---------------------------------------------
        # Results
        # ---------------------------------------------

        st.markdown("---")

        col1, col2 = st.columns(
            2,
            gap="large",
        )

        with col1:

            st.subheader(
                "📥 Original Image"
            )

            st.image(
                cv2.cvtColor(
                    original_image,
                    cv2.COLOR_BGR2RGB,
                ),
                width="stretch",
            )

        with col2:

            st.subheader(
                "🎯 Detected Objects"
            )

            st.image(
                cv2.cvtColor(
                    output_image,
                    cv2.COLOR_BGR2RGB,
                ),
                width="stretch",
            )

        # ---------------------------------------------
        # Metrics
        # ---------------------------------------------

        st.markdown("---")

        metric1, metric2, metric3 = (
            st.columns(3)
        )

        with metric1:

            st.metric(
                "Total Objects",
                len(detections),
            )

        with metric2:

            unique_classes = len(
                set(
                    item["class"]
                    for item in detections
                )
            )

            st.metric(
                "Object Classes",
                unique_classes,
            )

        with metric3:

            st.metric(
                "Confidence Threshold",
                f"{confidence:.0%}",
            )

        # ---------------------------------------------
        # Detection table
        # ---------------------------------------------

        st.markdown("---")

        st.subheader(
            "📋 Detection Results"
        )

        if detections:

            rows = []

            for index, detection in enumerate(
                detections,
                start=1,
            ):

                x1, y1, x2, y2 = (
                    detection["bbox"]
                )

                rows.append(
                    {
                        "#": index,
                        "Class": detection[
                            "class"
                        ],
                        "Confidence": (
                            f"{detection['confidence']:.2%}"
                        ),
                        "Bounding Box": (
                            f"({x1}, {y1}) → "
                            f"({x2}, {y2})"
                        ),
                    }
                )

            st.dataframe(
                rows,
                width="stretch",
                hide_index=True,
            )

            # -----------------------------------------
            # Class summary
            # -----------------------------------------

            counts = Counter(
                item["class"]
                for item in detections
            )

            st.subheader(
                "📊 Detected Classes"
            )

            for class_name, count in (
                counts.most_common()
            ):

                st.write(
                    f"**{class_name}** — "
                    f"{count}"
                )

        else:

            st.warning(
                "No objects were detected. "
                "Try lowering the confidence threshold."
            )

        # ---------------------------------------------
        # Download
        # ---------------------------------------------

        output_bytes = image_to_bytes(
            output_image
        )

        if output_bytes:

            st.markdown("---")

            st.download_button(
                "⬇️ Download Detected Image",
                data=output_bytes,
                file_name="detected_image.png",
                mime="image/png",
                width="stretch",
            )


# =========================================================
# VIDEO MODE
# =========================================================

else:

    uploaded_video = st.file_uploader(
        "🎥 Upload a short video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv",
        ],
    )

    if uploaded_video is None:

        st.info(
            "Upload a short video to run YOLO detection."
        )

        st.stop()

    # =====================================================
    # INPUT VIDEO
    # =====================================================

    st.markdown("---")

    video_input_col, video_info_col = st.columns(
        [2, 1],
        gap="large",
    )

    with video_input_col:

        st.subheader(
            "📥 Input Video"
        )

        # Streamlit's video component provides
        # browser controls automatically.
        st.video(
            uploaded_video
        )

    with video_info_col:

        st.subheader(
            "🎥 Video Information"
        )

        video_size_mb = (
            uploaded_video.size / (1024 * 1024)
        )

        st.metric(
            "File Size",
            f"{video_size_mb:.2f} MB",
        )

        st.metric(
            "Confidence",
            f"{confidence:.0%}",
        )

        st.info(
            "The uploaded video is preserved above "
            "so you can compare it with the processed "
            "YOLO output."
        )

    # =====================================================
    # PROCESS BUTTON
    # =====================================================

    if st.button(
        "🚀 Run Video Detection",
        type="primary",
        width="stretch",
    ):

        input_suffix = Path(
            uploaded_video.name
        ).suffix

        input_path = None
        output_path = None

        try:

            # ---------------------------------------------
            # Save uploaded video to temporary input file
            # ---------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=input_suffix,
            ) as input_temp:

                input_temp.write(
                    uploaded_video.getvalue()
                )

                input_path = input_temp.name

            # ---------------------------------------------
            # Create temporary output file
            # ---------------------------------------------

            output_temp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4",
            )

            output_path = output_temp.name

            output_temp.close()

            # ---------------------------------------------
            # Process video
            # ---------------------------------------------

            with st.spinner(
                "Processing video with YOLO..."
            ):

                process_video(
                    input_path,
                    output_path,
                    confidence,
                )

            # ---------------------------------------------
            # Verify output
            # ---------------------------------------------

            if not Path(output_path).exists():

                raise ValueError(
                    "The processed video file was not created."
                )

            output_size = Path(
                output_path
            ).stat().st_size

            if output_size == 0:

                raise ValueError(
                    "The processed video file is empty."
                )

            # ---------------------------------------------
            # Read processed video into memory
            # ---------------------------------------------

            with open(
                output_path,
                "rb",
            ) as video_file:

                video_bytes = (
                    video_file.read()
                )

            # ---------------------------------------------
            # Store in session state
            #
            # This ensures the processed video remains
            # available during Streamlit reruns caused
            # by download interactions.
            # ---------------------------------------------

            st.session_state[
                "processed_video_bytes"
            ] = video_bytes

            st.session_state[
                "processed_video_name"
            ] = "detected_video.mp4"

            st.success(
                "Video detection completed successfully."
            )

        except Exception as error:

            st.error(
                "Video processing failed."
            )

            st.exception(error)

        finally:

            # Input temporary file is no longer required
            # after processing.
            if input_path:

                try:

                    Path(input_path).unlink(
                        missing_ok=True
                    )

                except Exception:
                    pass

            # Do NOT immediately remove output_path
            # before the Streamlit UI has consumed it.
            #
            # It is cleaned after the bytes have been
            # successfully loaded into session state.
            if output_path:

                try:

                    Path(output_path).unlink(
                        missing_ok=True
                    )

                except Exception:
                    pass

    # =====================================================
    # PROCESSED VIDEO RESULT
    # =====================================================

    processed_video = st.session_state.get(
        "processed_video_bytes"
    )

    if processed_video:

        st.markdown("---")

        st.subheader(
            "🎯 Video Detection Results"
        )

        result_input_col, result_output_col = (
            st.columns(
                2,
                gap="large",
            )
        )

        # ---------------------------------------------
        # Original video
        # ---------------------------------------------

        with result_input_col:

            st.markdown(
                "#### 📥 Original Video"
            )

            # Re-display the uploaded source video
            # in the left column.
            st.video(
                uploaded_video
            )

        # ---------------------------------------------
        # Processed video
        # ---------------------------------------------

        with result_output_col:

            st.markdown(
                "#### 🎯 YOLO Detection Output"
            )

            # Browser video controls are enabled
            # automatically by Streamlit.
            st.video(
                processed_video
            )

        # ---------------------------------------------
        # Download
        # ---------------------------------------------

        st.markdown("---")

        st.download_button(
            "⬇️ Download Processed Video",
            data=processed_video,
            file_name=st.session_state.get(
                "processed_video_name",
                "detected_video.mp4",
            ),
            mime="video/mp4",
            width="stretch",
        )


# =========================================================
# FOOTER / PIPELINE
# =========================================================

st.markdown(
    """
    ### 📌 Detection Pipeline

    **Image / Video → YOLO11n → Bounding Boxes →
    Class Labels → Confidence Scores → Results**
    """
)

