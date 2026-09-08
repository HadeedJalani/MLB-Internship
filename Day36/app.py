from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from analytics import traffic_status
from traffic_violation import (
    DEFAULT_ANGLE_THRESHOLD_DEG,
    DEFAULT_CONFIRM_RATIO,
    DEFAULT_CONFIRM_WINDOW,
    COMPASS,
    RegionBox,
    RoadDirection,
    ViolationEvent,
    load_config,
    process_video,
)
from tracker import DEFAULT_CONF, DEFAULT_IOU, DEFAULT_MODEL, load_model

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_videos"
OUTPUT_ROOT = ROOT / "outputs"
MODEL_NAME = DEFAULT_MODEL
MAX_UPLOAD_FRAMES = 300

st.set_page_config(
    page_title="AI Traffic Violation Monitoring",
    page_icon="🚦",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.hero {padding: 1.0rem 1.1rem; border: 1px solid rgba(128,128,128,.25);
       border-radius: 16px; background: linear-gradient(120deg, rgba(40,80,120,.08), rgba(120,50,50,.06));}
.small-muted {color: rgba(128,128,128,.9); font-size:.9rem;}
.badge {display:inline-block; padding:.25rem .55rem; border-radius:999px;
        border:1px solid rgba(128,128,128,.25); margin-right:.35rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <h1>🚦 AI Traffic Violation Monitoring System</h1>
      <p class="small-muted">
        YOLO vehicle detection + ByteTrack tracking + rule-based traffic analytics.
        Analyze uploaded or sample traffic videos without a live camera.
      </p>
      <span class="badge">YOLOv8n</span>
      <span class="badge">ByteTrack</span>
      <span class="badge">Wrong-way</span>
      <span class="badge">Restricted zone</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("1. Demo variant")
variant_label = st.sidebar.radio(
    "Choose output",
    ["Wrong-Way Detection", "Traffic Violation Analytics"],
)
variant = "wrong_way" if variant_label == "Wrong-Way Detection" else "analytics"

st.sidebar.header("2. Input video")
source = st.sidebar.radio("Video source", ["Upload video", "Sample video"])

video_path: Path | None = None
display_name = "uploaded_video"

if source == "Upload video":
    upload = st.sidebar.file_uploader(
        "Upload MP4 / AVI / MOV / MKV",
        type=["mp4", "avi", "mov", "mkv"],
    )
    if upload is not None:
        tmp = tempfile.NamedTemporaryFile(
            suffix=Path(upload.name).suffix.lower() or ".mp4",
            delete=False,
        )
        tmp.write(upload.getbuffer())
        tmp.close()
        video_path = Path(tmp.name)
        display_name = Path(upload.name).stem
else:
    samples = sorted(SAMPLE_DIR.glob("*.mp4"))
    if not samples:
        st.sidebar.info(
            "No sample MP4s are committed yet. Add 3–5 public traffic videos "
            "to sample_videos/ and rerun the app."
        )
    else:
        selected = st.sidebar.selectbox(
            "Sample",
            samples,
            format_func=lambda p: p.stem.replace("_", " ").title(),
        )
        video_path = selected
        display_name = selected.stem

def _config_for_sample(path: Path):
    cfg_path = SAMPLE_DIR / f"{path.stem}_config.json"
    if cfg_path.exists():
        return load_config(cfg_path)
    direction, zone, direction_roi = RoadDirection("down"), None, None
    return direction, zone, direction_roi

direction = RoadDirection("down")
zone: RegionBox | None = None
direction_roi: RegionBox | None = None

if video_path is not None:
    st.sidebar.header("3. Traffic rules")

    use_config = False
    if source == "Sample":
        cfg = SAMPLE_DIR / f"{display_name}_config.json"
        if cfg.exists():
            use_config = st.sidebar.checkbox(
                "Use sample's calibrated config",
                value=True,
            )

    if use_config:
        direction, zone, direction_roi = _config_for_sample(video_path)
    else:
        direction_name = st.sidebar.selectbox(
            "Normal road direction",
            list(COMPASS.keys()),
            index=list(COMPASS.keys()).index("down"),
        )
        direction = RoadDirection(
            name=direction_name,
            angle_threshold_deg=float(
                st.sidebar.slider(
                    "Wrong-way angle threshold",
                    min_value=90,
                    max_value=175,
                    value=int(DEFAULT_ANGLE_THRESHOLD_DEG),
                    step=5,
                    help="Higher means a vehicle must be more strongly opposite the normal direction.",
                )
            ),
            confirm_window=int(
                st.sidebar.slider(
                    "Confirmation window",
                    min_value=5,
                    max_value=30,
                    value=DEFAULT_CONFIRM_WINDOW,
                    step=1,
                    help="Number of recent frames used to debounce wrong-way detection.",
                )
            ),
            confirm_ratio=float(
                st.sidebar.slider(
                    "Confirmation ratio",
                    min_value=0.50,
                    max_value=1.00,
                    value=DEFAULT_CONFIRM_RATIO,
                    step=0.05,
                    help="Fraction of the recent window that must be wrong-way.",
                )
            ),
        )

        add_zone = st.sidebar.checkbox("Enable restricted zone", value=True)
        if add_zone:
            zx = st.sidebar.slider(
                "Zone X range",
                0.0, 1.0, (0.70, 0.95), 0.01
            )
            zy = st.sidebar.slider(
                "Zone Y range",
                0.0, 1.0, (0.40, 0.95), 0.01
            )
            zone = RegionBox(zx[0], zx[1], zy[0], zy[1])

        scope_direction = st.sidebar.checkbox(
            "Scope wrong-way to a region",
            value=False,
            help="Useful on divided roads where opposite carriageways are both visible.",
        )
        if scope_direction:
            rx = st.sidebar.slider(
                "Direction ROI X",
                0.0, 1.0, (0.0, 1.0), 0.01
            )
            ry = st.sidebar.slider(
                "Direction ROI Y",
                0.0, 1.0, (0.0, 1.0), 0.01
            )
            direction_roi = RegionBox(rx[0], rx[1], ry[0], ry[1])

    st.sidebar.header("4. Detector")
    conf = st.sidebar.slider(
        "Confidence",
        0.05, 0.90, float(DEFAULT_CONF), 0.05
    )
    iou = st.sidebar.slider(
        "IoU / NMS",
        0.15, 0.90, float(DEFAULT_IOU), 0.05
    )

    st.sidebar.caption(
        "Hosted CPU runs are intentionally capped. For longer videos, "
        "run locally or raise MAX_UPLOAD_FRAMES in app.py."
    )

    if st.sidebar.button("▶ Run analysis", type="primary", use_container_width=True):
        st.session_state["run_requested"] = True

if video_path is None:
    st.info(
        "Choose a sample or upload a traffic video. The app supports unseen videos "
        "because direction and zone rules are configuration inputs rather than hard-coded detections."
    )
    st.stop()

if st.session_state.get("run_requested", False):
    st.session_state["run_requested"] = False

    with st.spinner("Loading YOLO detector..."):
        model = load_model(MODEL_NAME)

    variant_dir = OUTPUT_ROOT / ("variant-1" if variant == "wrong_way" else "variant-2")
    variant_dir.mkdir(parents=True, exist_ok=True)
    output_path = variant_dir / f"{display_name}_{variant}.mp4"

    progress = st.progress(0.0, text="Starting video analysis...")

    def on_progress(done: int, total: int) -> None:
        progress.progress(
            min(done / max(total, 1), 1.0),
            text=f"Processing frame {done}/{total}",
        )

    result = process_video(
        model,
        video_path,
        output_path,
        direction,
        zone,
        direction_roi,
        variant=variant,
        conf=conf,
        iou=iou,
        max_frames=MAX_UPLOAD_FRAMES,
        progress_cb=on_progress,
    )
    progress.empty()

    st.session_state["last_result"] = result
    st.session_state["last_output"] = output_path

result = st.session_state.get("last_result")
output_path = st.session_state.get("last_output")

if result is None or output_path is None:
    st.info("Press **Run analysis** in the sidebar to generate the selected demo variant.")
    st.stop()

st.divider()
st.subheader(f"Output — {variant_label}")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total vehicles", result.total_vehicles)
m2.metric("Total violations", result.total_violations)
m3.metric("Wrong-way", result.wrong_way_count)
m4.metric("Restricted zone", result.restricted_zone_count)
m5.metric("Traffic status", traffic_status(result.total_violations))

st.video(str(output_path))

st.download_button(
    "⬇ Download processed video",
    data=output_path.read_bytes(),
    file_name=output_path.name,
    mime="video/mp4",
    use_container_width=True,
)

events = result.events
events_df = pd.DataFrame(
    {
        "Vehicle ID": [event.track_id for event in events],
        "Vehicle Type": [event.class_name for event in events],
        "Violation": [
            "Wrong-way"
            if event.violation_type == "wrong_way"
            else "Restricted zone"
            for event in events
        ],
        "Timestamp (s)": [round(event.time_s, 2) for event in events],
        "Frame": [event.frame_idx for event in events],
    }
)

tab1, tab2, tab3 = st.tabs(
    ["Violation events", "Vehicle type statistics", "Processing summary"]
)

with tab1:
    if events_df.empty:
        st.success("No rule violations were recorded in this clip.")
    else:
        st.dataframe(events_df, use_container_width=True, hide_index=True)

with tab2:
    class_df = pd.Series(
        result.vehicle_counts_by_class,
        name="count",
    ).sort_values(ascending=False)
    st.bar_chart(class_df)

with tab3:
    st.write(
        {
            "Frames processed": result.n_frames,
            "Video FPS": round(result.fps, 2),
            "Elapsed seconds": round(result.elapsed_s, 2),
            "Variant": variant_label,
            "Normal direction": f"{direction.name} {direction.arrow}",
            "Restricted zone": "Enabled" if zone is not None else "Disabled",
            "Direction ROI": "Enabled" if direction_roi is not None else "Disabled",
            "Output path": str(output_path),
        }
    )
