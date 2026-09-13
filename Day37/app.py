from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from people_counter import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    CountingLine,
    ROI,
    load_model,
    process_image,
    process_video,
)

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_videos"
OUTPUT_DIR = ROOT / "outputs"
HOSTED_FRAME_LIMIT = 450


st.set_page_config(
    page_title="Smart People Counting System",
    page_icon="👥",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1350px; padding-top: 1.25rem;}
    .hero {
        padding: 1.15rem 1.25rem;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,.25);
        background: linear-gradient(115deg, rgba(35,100,160,.10), rgba(80,160,110,.08));
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>👥 Smart People Counting System</h1>
      <p>
        YOLOv8 person detection + ByteTrack tracking + live count,
        stable IDs, line crossing, ROI counting and peak occupancy.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading YOLOv8n...")
def get_model():
    return load_model()


try:
    model = get_model()
except Exception as exc:
    st.error(f"Could not load the detector: {exc}")
    st.stop()


with st.sidebar:
    st.header("Input source")
    source = st.radio(
        "Source",
        ["Sample video", "Upload image/video"],
    )

    selected_path = None
    uploaded = None

    if source == "Sample video":
        sample_files = sorted(
            [
                p for p in SAMPLE_DIR.glob("*")
                if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
            ]
        ) if SAMPLE_DIR.exists() else []

        if sample_files:
            selected_path = st.selectbox(
                "Sample video",
                sample_files,
                format_func=lambda p: p.stem.replace("_", " ").title(),
            )
        else:
            st.warning(
                "No sample videos found. Add 5+ legal-to-use clips to sample_videos/."
            )
    else:
        uploaded = st.file_uploader(
            "Upload image or video",
            type=["png", "jpg", "jpeg", "mp4", "avi", "mov", "mkv"],
        )

    st.header("Detection")
    conf = st.slider("Confidence", 0.05, 0.90, DEFAULT_CONF, 0.05)
    iou = st.slider("IoU", 0.10, 0.90, DEFAULT_IOU, 0.05)

    st.header("Counting line")
    enable_line = st.checkbox("Enable counting line", value=True)
    line = None
    if enable_line:
        orientation = st.radio(
            "Orientation",
            ["horizontal", "vertical"],
            horizontal=True,
        )
        position = st.slider(
            "Line position",
            0.05,
            0.95,
            0.50,
            0.05,
        )
        line = CountingLine(orientation, position)

    st.header("ROI")
    enable_roi = st.checkbox("Enable ROI counting", value=False)
    roi = None
    if enable_roi:
        x_min, x_max = st.slider(
            "ROI X range",
            0.0, 1.0, (0.10, 0.90), 0.05,
        )
        y_min, y_max = st.slider(
            "ROI Y range",
            0.0, 1.0, (0.10, 0.90), 0.05,
        )
        roi = ROI(x_min, x_max, y_min, y_max)

    st.caption(
        "For local full-video processing, set HOSTED_FRAME_LIMIT = None in app.py."
    )

if selected_path is None and uploaded is None:
    st.info("Choose a sample video or upload an image/video.")
    st.stop()

if uploaded is not None:
    suffix = Path(uploaded.name).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    input_path = Path(tmp.name)
    input_name = Path(uploaded.name).stem
else:
    input_path = selected_path
    input_name = selected_path.stem

is_video = input_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}

if st.button("▶ Run analysis", type="primary", use_container_width=True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = st.progress(0.0, text="Starting analysis...")

    if is_video:
        output_path = OUTPUT_DIR / f"{input_name}_people_counted.mp4"

        def progress_cb(done: int, total: int):
            progress.progress(
                min(done / max(total, 1), 1.0),
                text=f"Processing frame {done}/{total}",
            )

        try:
            result = process_video(
                model=model,
                in_path=input_path,
                out_path=output_path,
                conf=conf,
                iou=iou,
                line=line,
                roi=roi,
                progress_cb=progress_cb,
                max_frames=HOSTED_FRAME_LIMIT,
            )
        except Exception as exc:
            progress.empty()
            st.error(f"Video processing failed: {exc}")
            st.stop()

        progress.empty()

        st.session_state["result"] = result
        st.session_state["output"] = output_path
        st.session_state["kind"] = "video"
    else:
        output_path = OUTPUT_DIR / f"{input_name}_people_counted.png"
        try:
            annotated, count = process_image(
                model=model,
                image_path=input_path,
                out_path=output_path,
                conf=conf,
                iou=iou,
                roi=roi,
            )
        except Exception as exc:
            progress.empty()
            st.error(f"Image processing failed: {exc}")
            st.stop()

        progress.empty()

        st.session_state["image_result"] = (annotated, count)
        st.session_state["output"] = output_path
        st.session_state["kind"] = "image"

if "output" not in st.session_state:
    st.info("Press **Run analysis** to create the processed output.")
    st.stop()

output_path = Path(st.session_state["output"])

if st.session_state["kind"] == "video":
    result = st.session_state["result"]

    st.subheader("Processed video")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("People now", result.current_count)
    c2.metric("Peak", result.peak_count)
    c3.metric("Unique seen", result.total_people_seen)
    c4.metric("Crossings", result.line_crossings)
    c5.metric("Entries", result.entries)
    c6.metric("Exits", result.exits)

    st.video(str(output_path))

    st.download_button(
        "⬇ Download processed video",
        data=output_path.read_bytes(),
        file_name=output_path.name,
        mime="video/mp4",
        use_container_width=True,
    )

    chart = pd.DataFrame(
        {
            "Frame": range(1, len(result.people_per_frame) + 1),
            "People": result.people_per_frame,
        }
    ).set_index("Frame")

    st.subheader("People per frame")
    st.line_chart(chart)

    st.write(
        {
            "Frames processed": result.n_frames,
            "FPS": round(result.fps, 2),
            "Processing time (s)": round(result.elapsed_s, 2),
            "ROI count on final frame": result.roi_count,
            "Output": str(output_path),
        }
    )
else:
    annotated, count = st.session_state["image_result"]

    st.subheader("Processed image")
    st.metric("People detected", count)
    st.image(annotated, channels="BGR", use_container_width=True)

    st.download_button(
        "⬇ Download processed image",
        data=output_path.read_bytes(),
        file_name=output_path.name,
        mime="image/png",
        use_container_width=True,
    )
