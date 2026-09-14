from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from security_monitoring import (
    process_image,
    process_video,
)


st.set_page_config(
    page_title="Optimized Security Monitoring",
    page_icon="🎥",
    layout="wide",
)


st.title("Optimized Security Monitoring")
st.caption(
    "YOLOv8 person detection + persistent tracking + ROI monitoring + event analytics"
)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def save_uploaded_file(uploaded_file) -> str:
    """
    Save a Streamlit uploaded file to a temporary local file.
    """
    suffix = Path(uploaded_file.name).suffix or ".mp4"

    temporary_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    temporary_file.write(uploaded_file.getbuffer())
    temporary_file.close()

    return temporary_file.name


def render_statistics(stats: dict) -> None:
    """
    Display the main processing statistics.
    """
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Peak people in ROI", stats.get("peak_roi", 0))
    col2.metric("Unique tracked IDs", stats.get("unique_ids", 0))
    col3.metric("Entries", stats.get("entries", 0))
    col4.metric("Exits", stats.get("exits", 0))

    col5, col6, col7 = st.columns(3)

    col5.metric(
        "Average dwell time",
        f"{stats.get('avg_dwell_s', 0.0):.2f} seconds",
    )

    col6.metric(
        "Processing FPS",
        f"{stats.get('avg_fps', 0.0):.2f}",
    )

    col7.metric(
        "Frames processed",
        stats.get("frames_processed", 0),
    )


# ---------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Processing Settings")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.45,
        step=0.05,
        help="Higher values reduce weak detections but may miss difficult objects.",
    )

    iou = st.slider(
        "IoU threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Controls non-maximum suppression overlap.",
    )

    image_size = st.select_slider(
        "Inference image size",
        options=[320, 416, 512, 640],
        value=512,
        help="Smaller values are faster; larger values may improve small-object detection.",
    )

    frame_skip = st.slider(
        "Process every Nth frame",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
        help="Use 2 or 3 for faster processing on long videos.",
    )

    tracker_name = st.selectbox(
        "Tracking algorithm",
        options=["ByteTrack", "BoT-SORT"],
        help="Both trackers are supported by Ultralytics.",
    )

    show_trails = st.checkbox(
        "Show tracking trails",
        value=True,
    )

    st.markdown("---")
    st.info(
        "For the crossing-case requirement, test two videos where "
        "people cross paths and compare whether IDs remain consistent."
    )


# ---------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------

image_tab, video_tab = st.tabs(["🖼️ Image Monitoring", "🎥 Video Monitoring"])


# ---------------------------------------------------------------------
# Image workflow
# ---------------------------------------------------------------------

with image_tab:
    st.subheader("Image Detection")

    image_upload = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="image_upload",
    )

    if image_upload is not None:
        try:
            input_image = Image.open(image_upload).convert("RGB")

            st.image(
                input_image,
                caption="Input image",
                width="stretch",
            )

            if st.button("Run image monitoring", key="run_image"):
                with st.spinner("Running optimized image inference..."):
                    processed_image, image_stats = process_image(
                        image=input_image,
                        conf=confidence,
                        iou=iou,
                        imgsz=image_size,
                    )

                st.success("Image processing completed.")

                st.image(
                    processed_image,
                    caption="Processed image",
                    width="stretch",
                )

                st.metric(
                    "People detected in image",
                    image_stats["roi_count"],
                )

                st.download_button(
                    label="Download processed image",
                    data=processed_image,
                    file_name="day39_processed_image.png",
                    mime="image/png",
                )

        except Exception as error:
            st.error(f"Image processing failed: {error}")


# ---------------------------------------------------------------------
# Video workflow
# ---------------------------------------------------------------------

with video_tab:
    st.subheader("Video Security Monitoring")

    video_upload = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv"],
        key="video_upload",
    )

    if video_upload is not None:
        temporary_video_path = None

        try:
            temporary_video_path = save_uploaded_file(video_upload)

            st.video(video_upload)

            if st.button("Run optimized video monitoring", key="run_video"):
                progress_bar = st.progress(
                    0,
                    text="Preparing video processing...",
                )

                status_area = st.empty()

                def update_progress(progress: float, message: str) -> None:
                    progress_bar.progress(
                        min(max(progress, 0.0), 1.0),
                        text=message,
                    )
                    status_area.info(message)

                with st.spinner("Processing video with persistent tracking..."):
                    output_video_path, csv_path, video_stats = process_video(
                        video_path=temporary_video_path,
                        conf=confidence,
                        iou=iou,
                        imgsz=image_size,
                        frame_skip=frame_skip,
                        tracker_name=tracker_name,
                        show_trails=show_trails,
                        progress_callback=update_progress,
                    )

                progress_bar.progress(
                    1.0,
                    text="Processing completed.",
                )

                st.success("Video processing completed successfully.")

                render_statistics(video_stats)

                st.subheader("Processed Video")
                st.video(output_video_path.read_bytes())

                st.download_button(
                    label="Download processed video",
                    data=output_video_path.read_bytes(),
                    file_name=output_video_path.name,
                    mime="video/mp4",
                )

                st.subheader("Event Report")

                event_csv_bytes = csv_path.read_bytes()

                st.download_button(
                    label="Download event CSV",
                    data=event_csv_bytes,
                    file_name=csv_path.name,
                    mime="text/csv",
                )

                st.caption(
                    "The CSV contains ENTRY and EXIT events, timestamps, "
                    "track IDs, and dwell-time information."
                )

        except Exception as error:
            st.error(f"Video processing failed: {error}")

        finally:
            if temporary_video_path and os.path.exists(temporary_video_path):
                try:
                    os.unlink(temporary_video_path)
                except OSError:
                    pass


st.markdown("---")

st.caption(
    "Day 39 optimization focus: configurable inference, faster processing, "
    "persistent tracking, ROI-based events, and downloadable analytics."
)