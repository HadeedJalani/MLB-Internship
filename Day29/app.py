import os
import tempfile
import time

import streamlit as st

from tracker.object_tracker import ObjectTracker


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Object Tracking System",
    page_icon="🎯",
    layout="wide",
)


# =========================================================
# CONSTANTS
# =========================================================

MODEL_PATH = "yolo11n.pt"


# =========================================================
# COMPATIBILITY HELPERS
# =========================================================

def compat_widget(func, **kwargs):
    """
    Call a Streamlit widget function, adapting the width/container-width
    keyword to whatever the installed Streamlit version actually supports.

    Newest Streamlit  -> width="stretch"
    Slightly older    -> use_container_width=True
    Oldest            -> neither (falls back to default sizing)
    """
    try:
        return func(width="stretch", **kwargs)
    except TypeError:
        pass
    try:
        return func(use_container_width=True, **kwargs)
    except TypeError:
        pass
    return func(**kwargs)


def render_html(raw_html):
    """
    st.markdown(..., unsafe_allow_html=True) silently fails to render HTML
    as HTML if any line has leading indentation -- Markdown treats a blank
    line followed by an indented line as the start of a preformatted code
    block. textwrap.dedent() alone isn't enough here because nested tags
    are still indented relative to each other, so strip every line
    individually instead.
    """
    lines = [line.strip() for line in raw_html.strip().splitlines()]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

@st.cache_resource
def load_tracker(
    confidence,
    tracker_name,
):
    """
    Load and cache the YOLO tracking system.
    """

    return ObjectTracker(
        model_path=MODEL_PATH,
        confidence=confidence,
        tracker=tracker_name,
    )


def read_file_bytes(file_path):
    """
    Read file as bytes.
    """

    with open(
        file_path,
        "rb",
    ) as file:

        return file.read()


# =========================================================
# CUSTOM CSS
# =========================================================

render_html(
    """
    <style>

    .hero {
        padding: 35px;
        border-radius: 18px;
        text-align: center;
        margin-bottom: 25px;
        background: linear-gradient(
            135deg,
            #0f172a,
            #1e293b
        );
        color: white;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 17px;
        opacity: 0.85;
    }

    .info-card {
        padding: 18px;
        border-radius: 12px;
        background-color: #f5f7fa;
        margin-bottom: 10px;
    }

    </style>
    """
)


# =========================================================
# HEADER
# =========================================================

render_html(
    """
    <div class="hero">

        <div class="hero-title">
            🎯 Smart Object Tracking System
        </div>

        <div class="hero-subtitle">
            Multi-object tracking powered by
            YOLO11, ByteTrack, BoT-SORT,
            OpenCV and Streamlit.
        </div>

    </div>
    """
)


# =========================================================
# SIDEBAR SETTINGS
# =========================================================

st.sidebar.header(
    "⚙️ Tracking Settings"
)


tracker_name = st.sidebar.selectbox(
    "Select Tracking Algorithm",
    [
        "ByteTrack",
        "BoT-SORT",
    ],
)


confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.40,
    step=0.05,
)


st.sidebar.markdown("---")


st.sidebar.subheader(
    "🧠 About Trackers"
)


st.sidebar.info(
    """
    **ByteTrack**

    ⚡ Fast and lightweight  
    🚗 Excellent for traffic scenes  
    💻 Recommended for CPU deployment
    """
)


st.sidebar.info(
    """
    **BoT-SORT**

    🎯 Strong identity tracking  
    🔄 Better occlusion handling  
    🧠 More computationally intensive
    """
)


# =========================================================
# FILE UPLOAD
# =========================================================

st.subheader(
    "📤 Upload Video"
)


uploaded_file = st.file_uploader(
    "Upload a video for object tracking",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv",
    ],
)


if uploaded_file is None:

    st.info(
        "Upload a video to begin object tracking."
    )

    st.stop()


# =========================================================
# SAVE INPUT VIDEO
# =========================================================

input_suffix = os.path.splitext(
    uploaded_file.name
)[1]

if not input_suffix:
    input_suffix = ".mp4"


with tempfile.NamedTemporaryFile(
    delete=False,
    suffix=input_suffix,
) as temp_input:

    temp_input.write(
        uploaded_file.getbuffer()
    )

    input_path = temp_input.name


# =========================================================
# DISPLAY INPUT VIDEO
# =========================================================

st.markdown("---")

st.subheader(
    "🎥 Input Preview"
)

st.video(
    uploaded_file.getvalue()
)


# =========================================================
# START TRACKING
# =========================================================

start_clicked = compat_widget(
    st.button,
    label="🚀 Start Object Tracking",
    type="primary",
)

if start_clicked:

    try:

        # -------------------------------------------------
        # Progress UI
        # -------------------------------------------------

        progress_bar = st.progress(0)

        status_text = st.empty()

        status_text.info(
            "Loading tracking model..."
        )

        # -------------------------------------------------
        # Load Tracker
        # -------------------------------------------------

        tracker = load_tracker(
            confidence=confidence,
            tracker_name=tracker_name,
        )

        # -------------------------------------------------
        # Output File
        # -------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4",
        ) as temp_output:

            output_path = temp_output.name

        # -------------------------------------------------
        # Progress callback
        # -------------------------------------------------

        def update_progress(progress):

            percentage = int(
                progress * 100
            )

            progress_bar.progress(
                percentage
            )

            status_text.info(
                f"Tracking objects... "
                f"{percentage}%"
            )

        # -------------------------------------------------
        # Process Video
        # -------------------------------------------------

        start_time = time.perf_counter()

        stats = tracker.process_video(
            input_path=input_path,
            output_path=output_path,
            progress_callback=update_progress,
        )

        total_time = (
            time.perf_counter()
            - start_time
        )

        progress_bar.progress(100)

        status_text.success(
            "Tracking completed successfully!"
        )

        # -------------------------------------------------
        # Results
        # -------------------------------------------------

        st.markdown("---")

        st.subheader(
            "🎬 Tracking Results"
        )

        # Horizontal Input / Output
        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### 📥 Original Video"
            )

            st.video(
                uploaded_file.getvalue()
            )

        with col2:

            st.markdown(
                "### 🎯 Tracked Output"
            )

            output_bytes = read_file_bytes(
                output_path
            )

            st.video(
                output_bytes
            )

        # -------------------------------------------------
        # Statistics
        # -------------------------------------------------

        st.markdown("---")

        st.subheader(
            "📊 Tracking Statistics"
        )

        metric1, metric2, metric3, metric4 = (
            st.columns(4)
        )

        metric1.metric(
            "Frames Processed",
            stats["total_frames"],
        )

        metric2.metric(
            "Unique Objects",
            stats["unique_objects"],
        )

        metric3.metric(
            "Classes Detected",
            stats["unique_classes"],
        )

        metric4.metric(
            "Processing Time",
            f"{total_time:.2f}s",
        )

        # -------------------------------------------------
        # Additional Information
        # -------------------------------------------------

        st.markdown("---")

        info_col1, info_col2 = st.columns(2)

        with info_col1:

            st.markdown(
                "### 🆔 Tracking IDs"
            )

            if stats["tracking_ids"]:

                st.write(
                    stats["tracking_ids"]
                )

            else:

                st.info(
                    "No tracking IDs detected."
                )

        with info_col2:

            st.markdown(
                "### 🏷️ Classes Detected"
            )

            if stats["classes_detected"]:

                for class_name in (
                    stats["classes_detected"]
                ):

                    st.write(
                        f"• {class_name}"
                    )

            else:

                st.info(
                    "No objects detected."
                )

        # -------------------------------------------------
        # Download
        # -------------------------------------------------

        st.markdown("---")

        compat_widget(
            st.download_button,
            label="⬇️ Download Tracked Video",
            data=output_bytes,
            file_name=(
                "tracked_"
                + uploaded_file.name
            ),
            mime="video/mp4",
        )

    except Exception as error:

        st.error(
            "Video tracking failed."
        )

        st.exception(error)