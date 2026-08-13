# ==========================================================
# MLB Summer Internship - Day 20
# Video Processing with OpenCV
# ==========================================================

import os
import tempfile
import time

import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from video_operations import (
    get_video_info,
    process_video,
)

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Day 20 | Video Processing",
    page_icon="🎥",
    layout="wide",
)

# ==========================================================
# Styling
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
        font-size: 18px;
        color: #777;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(128,128,128,0.08);
        margin-bottom: 15px;
    }

    .pipeline-box {
        padding: 20px;
        border-radius: 12px;
        background-color: rgba(60,120,200,0.15);
        border: 1px solid rgba(60,120,200,0.3);
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# Header
# ==========================================================

st.markdown(
    '<div class="main-title">🎥 Day 20 Video Processing with OpenCV</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Process recorded videos and live webcam frames using
    grayscale conversion, Gaussian blur, and Canny edge detection.
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("⚙️ Processing Settings")

processing_mode = st.sidebar.selectbox(
    "Processing Mode",
    [
        "Original",
        "Grayscale",
        "Gaussian Blur",
        "Canny Edge Detection",
    ],
)

blur_kernel = st.sidebar.slider(
    "Blur Strength",
    min_value=3,
    max_value=21,
    value=7,
    step=2,
)

st.sidebar.markdown("---")

st.sidebar.subheader("Canny Settings")

canny_lower = st.sidebar.slider(
    "Lower Threshold",
    min_value=0,
    max_value=255,
    value=50,
)

canny_upper = st.sidebar.slider(
    "Upper Threshold",
    min_value=0,
    max_value=255,
    value=150,
)

# ==========================================================
# Processing Pipeline
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div class="pipeline-box">

    ### 🔬 Processing Pipeline

    **Recorded Video**

    🎥 Video  
    ↓  
    🖼️ Frame Extraction  
    ↓  
    ⚫ Grayscale  
    ↓  
    🌫️ Gaussian Blur  
    ↓  
    ✨ Canny Edge Detection  
    ↓  
    🎬 Processed Video

    **Webcam**

    📷 Camera  
    ↓  
    🖼️ Live Frame  
    ↓  
    ⚫ Grayscale  
    ↓  
    🌫️ Gaussian Blur  
    ↓  
    ✨ Canny  
    ↓  
    📺 Live Output

    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# Helper
# ==========================================================


def mode_to_internal_name(mode):

    mapping = {
        "Original": "original",
        "Grayscale": "grayscale",
        "Gaussian Blur": "blur",
        "Canny Edge Detection": "canny",
    }

    return mapping[mode]


# ==========================================================
# Webcam Processor
# ==========================================================


class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.mode = "canny"
        self.blur_kernel = 7
        self.canny_lower = 50
        self.canny_upper = 150

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        # ----------------------------------------------
        # Original
        # ----------------------------------------------

        if self.mode == "original":

            processed = image

        # ----------------------------------------------
        # Grayscale
        # ----------------------------------------------

        elif self.mode == "grayscale":

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

            processed = cv2.cvtColor(
                gray,
                cv2.COLOR_GRAY2BGR,
            )

        # ----------------------------------------------
        # Gaussian Blur
        # ----------------------------------------------

        elif self.mode == "blur":

            blurred = cv2.GaussianBlur(
                image,
                (
                    self.blur_kernel,
                    self.blur_kernel,
                ),
                0,
            )

            processed = blurred

        # ----------------------------------------------
        # Canny
        # ----------------------------------------------

        elif self.mode == "canny":

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

            blurred = cv2.GaussianBlur(
                gray,
                (
                    self.blur_kernel,
                    self.blur_kernel,
                ),
                0,
            )

            edges = cv2.Canny(
                blurred,
                self.canny_lower,
                self.canny_upper,
            )

            processed = cv2.cvtColor(
                edges,
                cv2.COLOR_GRAY2BGR,
            )

        else:

            processed = image

        return av.VideoFrame.from_ndarray(
            processed,
            format="bgr24",
        )


# ==========================================================
# Main Tabs
# ==========================================================

tab_video, tab_webcam = st.tabs(
    [
        "🎬 Process Recorded Video",
        "📷 Live Webcam Processing",
    ]
)

# ==========================================================
# TAB 1 — RECORDED VIDEO
# ==========================================================

with tab_video:

    st.markdown(
        '<div class="section-title">📤 Upload a Video</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload an MP4, AVI, MOV, MKV, or WEBM video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv",
            "webm",
        ],
        key="video_uploader",
    )

    # ------------------------------------------------------
    # No Video
    # ------------------------------------------------------

    if uploaded_file is None:

        st.info(
            "👆 Upload a video to begin processing."
        )

        st.markdown("---")

        st.subheader(
            "📚 What this application demonstrates"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.markdown(
                """
                ### 🎞️ Frames

                A video is processed as a
                sequence of individual frames.
                """
            )

        with col2:

            st.markdown(
                """
                ### ⚫ Grayscale

                Converts color frames into
                intensity-based images.
                """
            )

        with col3:

            st.markdown(
                """
                ### 🌫️ Gaussian Blur

                Reduces noise and smooths
                each video frame.
                """
            )

        with col4:

            st.markdown(
                """
                ### ✨ Canny

                Detects edges caused by
                intensity changes.
                """
            )

    # ------------------------------------------------------
    # Video Uploaded
    # ------------------------------------------------------

    else:

        # ==================================================
        # Save uploaded file temporarily
        # ==================================================

        file_extension = os.path.splitext(
            uploaded_file.name
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension,
        ) as temporary_file:

            temporary_file.write(
                uploaded_file.getbuffer()
            )

            input_path = temporary_file.name

        # ==================================================
        # Video Information
        # ==================================================

        try:

            info = get_video_info(
                input_path
            )

            st.subheader(
                "📊 Video Information"
            )

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:

                st.metric(
                    "FPS",
                    info["fps"],
                )

            with col2:

                st.metric(
                    "Width",
                    f"{info['width']} px",
                )

            with col3:

                st.metric(
                    "Height",
                    f"{info['height']} px",
                )

            with col4:

                st.metric(
                    "Frames",
                    info["total_frames"],
                )

            with col5:

                st.metric(
                    "Duration",
                    f"{info['duration']} sec",
                )

        except Exception as error:

            st.error(
                f"Unable to read video information: {error}"
            )

            if os.path.exists(input_path):

                os.unlink(input_path)

            st.stop()

        # ==================================================
        # Processing Configuration
        # ==================================================

        st.subheader(
            "🔧 Processing Configuration"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.write(
                f"**Mode:** {processing_mode}"
            )

        with col2:

            st.write(
                f"**Blur Kernel:** {blur_kernel}"
            )

        with col3:

            st.write(
                f"**Canny:** "
                f"{canny_lower} / {canny_upper}"
            )

        # ==================================================
        # Original + Preview
        # ==================================================

        st.markdown("---")

        st.subheader(
            "🎬 Original Video"
        )

        st.video(
            uploaded_file
        )

        # ==================================================
        # Process Button
        # ==================================================

        process_button = st.button(
            "🚀 Process Video",
            type="primary",
            use_container_width=True,
            key="process_video_button",
        )

        if process_button:

            internal_mode = mode_to_internal_name(
                processing_mode
            )

            output_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4",
            )

            output_path = output_file.name

            output_file.close()

            progress_bar = st.progress(0)

            status_text = st.empty()

            start_time = time.time()

            # --------------------------------------------------
            # Progress Callback
            # --------------------------------------------------

            def update_progress(progress):

                progress = min(
                    max(progress, 0.0),
                    1.0,
                )

                progress_bar.progress(
                    progress
                )

                status_text.write(
                    f"Processing: "
                    f"{progress * 100:.1f}%"
                )

            # --------------------------------------------------
            # Process
            # --------------------------------------------------

            try:

                result = process_video(
                    input_path=input_path,
                    output_path=output_path,
                    mode=internal_mode,
                    blur_kernel=blur_kernel,
                    canny_lower=canny_lower,
                    canny_upper=canny_upper,
                    progress_callback=update_progress,
                )

                processing_time = (
                    time.time()
                    - start_time
                )

                progress_bar.progress(1.0)

                status_text.success(
                    "✅ Video processing completed successfully."
                )

                # ==================================================
                # Processing Statistics
                # ==================================================

                st.subheader(
                    "📈 Processing Statistics"
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Frames Processed",
                        result["processed_frames"],
                    )

                with col2:

                    st.metric(
                        "Processing Time",
                        f"{processing_time:.2f} sec",
                    )

                with col3:

                    st.metric(
                        "FPS",
                        result["fps"],
                    )

                with col4:

                    st.metric(
                        "Resolution",
                        f"{result['width']} × "
                        f"{result['height']}",
                    )

                # ==================================================
                # SIDE-BY-SIDE VIDEO COMPARISON
                # ==================================================

                st.markdown("---")

                st.subheader(
                    "🔍 Original vs Processed"
                )

                original_col, processed_col = st.columns(2)

                with original_col:

                    st.markdown(
                        "### 🎬 Original"
                    )

                    # Use the uploaded bytes instead of the
                    # UploadedFile object because Streamlit's
                    # file pointer may already have been consumed.

                    uploaded_file.seek(0)

                    original_bytes = (
                        uploaded_file.read()
                    )

                    st.video(
                        original_bytes
                    )

                with processed_col:

                    st.markdown(
                        f"### ✨ Processed — {processing_mode}"
                    )

                    if os.path.exists(
                        output_path
                    ):

                        with open(
                            output_path,
                            "rb",
                        ) as video_file:

                            processed_bytes = (
                                video_file.read()
                            )

                        st.video(
                            processed_bytes
                        )

                    else:

                        st.error(
                            "Processed video could not be created."
                        )

                # ==================================================
                # Download
                # ==================================================

                st.markdown("---")

                st.subheader(
                    "📥 Export"
                )

                if os.path.exists(
                    output_path
                ):

                    with open(
                        output_path,
                        "rb",
                    ) as video_file:

                        video_bytes = (
                            video_file.read()
                        )

                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=video_bytes,
                        file_name=(
                            f"{os.path.splitext(uploaded_file.name)[0]}"
                            f"_{internal_mode}.mp4"
                        ),
                        mime="video/mp4",
                        use_container_width=True,
                    )

            except Exception as error:

                st.error(
                    f"❌ Video processing failed: {error}"
                )

        # ==================================================
        # Cleanup
        # ==================================================

        # Input files are temporary and are only used during
        # the current Streamlit session.


# ==========================================================
# TAB 2 — LIVE WEBCAM
# ==========================================================

with tab_webcam:

    st.markdown(
        '<div class="section-title">📷 Live Webcam Processing</div>',
        unsafe_allow_html=True,
    )

    st.info(
        """
        Allow camera access when your browser asks for permission.
        The webcam frames are processed live using OpenCV.
        """
    )

    # ------------------------------------------------------
    # Webcam Mode
    # ------------------------------------------------------

    webcam_mode = st.selectbox(
        "Webcam Processing Mode",
        [
            "Original",
            "Grayscale",
            "Gaussian Blur",
            "Canny Edge Detection",
        ],
        index=3,
        key="webcam_mode",
    )

    # ------------------------------------------------------
    # Webcam Processor
    # ------------------------------------------------------

    ctx = webrtc_streamer(
        key="day20-webcam",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    # ------------------------------------------------------
    # Update Webcam Processor Settings
    # ------------------------------------------------------

    if ctx.video_processor:

        ctx.video_processor.mode = mode_to_internal_name(
            webcam_mode
        )

        ctx.video_processor.blur_kernel = (
            blur_kernel
        )

        ctx.video_processor.canny_lower = (
            canny_lower
        )

        ctx.video_processor.canny_upper = (
            canny_upper
        )

    # ------------------------------------------------------
    # Webcam Information
    # ------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🎛️ Live Processing Settings"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Mode",
            webcam_mode,
        )

    with col2:

        st.metric(
            "Blur Kernel",
            blur_kernel,
        )

    with col3:

        st.metric(
            "Canny Threshold",
            f"{canny_lower} / {canny_upper}",
        )

    st.markdown("---")

    st.markdown(
        """
        ### 🔬 Webcam Pipeline

        **Camera Feed**
        → **Frame Extraction**
        → **Grayscale**
        → **Gaussian Blur**
        → **Canny Edge Detection**
        → **Live Processed Output**

        The processing happens frame-by-frame in real time.
        """
    )

    
