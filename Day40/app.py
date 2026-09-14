from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from video_analytics import VideoConfig, process_video

st.set_page_config(
    page_title="Smart Video Analytics",
    page_icon="🎥",
    layout="wide",
)

st.title("🎥Smart Video Analytics System")
st.write(
    "Upload a recorded video and analyze it with YOLO detection, object tracking, "
    "ROI monitoring, entry/exit detection, and performance analytics."
)

with st.sidebar:
    st.header("⚙️ Analytics Settings")

    confidence = st.slider("Confidence threshold", 0.10, 0.95, 0.40, 0.05)
    iou = st.slider("IoU threshold", 0.10, 0.95, 0.50, 0.05)

    image_size = st.select_slider(
        "Inference image size",
        options=[480, 640],
        value=640,
    )

    frame_skip = st.slider(
        "Frame skip",
        1, 5, 1,
        help="1 = every frame, 2 = every second frame, etc.",
    )

    tracker = st.selectbox(
        "Tracking algorithm",
        ["ByteTrack", "BoT-SORT"],
    )

    st.markdown("---")
    st.subheader("📐 ROI")

    roi_x1 = st.slider("ROI left", 0.0, 0.90, 0.20, 0.05)
    roi_y1 = st.slider("ROI top", 0.0, 0.90, 0.20, 0.05)
    roi_x2 = st.slider("ROI right", 0.10, 1.00, 0.80, 0.05)
    roi_y2 = st.slider("ROI bottom", 0.10, 1.00, 0.85, 0.05)

    show_trails = st.checkbox("Show tracking trails", True)

uploaded_video = st.file_uploader(
    "Upload a 15–30 second video",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded_video is None:
    st.info("Upload a short people, traffic, or vehicle video to begin.")
    st.markdown(
        "**Pipeline:** Video → Frames → YOLO → Tracking → ROI → Events → Analytics → Output"
    )
else:
    st.subheader("Input Video")
    st.video(uploaded_video)

    if st.button("▶ Start Video Analytics", type="primary"):
        temp_path = None

        try:
            suffix = Path(uploaded_video.name).suffix or ".mp4"

            tmp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            )

            tmp.write(uploaded_video.getbuffer())
            tmp.close()
            temp_path = tmp.name

            config = VideoConfig(
                conf=confidence,
                iou=iou,
                imgsz=image_size,
                frame_skip=frame_skip,
                tracker=tracker,
                roi_x1=roi_x1,
                roi_y1=roi_y1,
                roi_x2=roi_x2,
                roi_y2=roi_y2,
                show_trails=show_trails,
            )

            progress = st.progress(0, text="Starting...")
            status = st.empty()

            def callback(value: float, message: str) -> None:
                progress.progress(min(max(value, 0.0), 1.0), text=message)
                status.info(message)

            with st.spinner("Running YOLO detection and tracking..."):
                output_video, events_csv, summary = process_video(
                    temp_path,
                    "outputs",
                    config,
                    callback,
                )

            progress.progress(1.0, text="Processing completed.")
            status.success("Video analytics completed successfully.")

            st.subheader("📊 Analytics Summary")

            cols = st.columns(3)
            cols[0].metric("Total Objects", summary["total_objects"])
            cols[1].metric("Total Entries", summary["total_entries"])
            cols[2].metric("Total Exits", summary["total_exits"])

            cols = st.columns(3)
            cols[0].metric(
                "Maximum Objects in ROI",
                summary["maximum_objects_in_roi"],
            )
            cols[1].metric(
                "Average FPS",
                f"{summary['average_fps']:.2f}",
            )
            cols[2].metric(
                "Processing Time",
                f"{summary['processing_time_s']:.2f}s",
            )

            cols = st.columns(3)
            cols[0].metric(
                "Video Duration",
                f"{summary['video_duration_s']:.2f}s",
            )
            cols[1].metric(
                "Average ROI Count",
                f"{summary['average_objects_in_roi']:.2f}",
            )
            cols[2].metric(
                "Average Dwell Time",
                f"{summary['average_dwell_time_s']:.2f}s",
            )

            st.subheader("🎞️ Processed Video")
            video_bytes = output_video.read_bytes()

            st.video(video_bytes)

            st.download_button(
                "⬇ Download Processed Video",
                data=video_bytes,
                file_name=output_video.name,
                mime="video/mp4",
            )

            st.subheader("📄 events.csv")
            csv_bytes = events_csv.read_bytes()

            st.download_button(
                "⬇ Download events.csv",
                data=csv_bytes,
                file_name="events.csv",
                mime="text/csv",
            )

            event_df = pd.read_csv(events_csv)
            st.dataframe(event_df, width="stretch")

        except Exception as error:
            st.error(f"Video processing failed: {error}")

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
