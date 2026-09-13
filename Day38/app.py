from __future__ import annotations

from pathlib import Path
import json
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from monitoring import (
    VIDEO_EXTENSIONS,
    process_video,
    read_first_frame,
)
from segmentation import METHODS, segment_image


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
SAMPLE_VIDEO_DIR = ROOT / "sample_videos"
SAMPLE_IMAGE_DIR = ROOT / "sample_input_images"

st.set_page_config(
    page_title="Security Monitoring & Segmentation",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1400px; padding-top: 1.1rem;}
    .hero {
        padding: 1.15rem 1.25rem;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,.24);
        background: linear-gradient(120deg, rgba(60,90,160,.12), rgba(50,160,110,.08));
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🛡️ Intelligent Security Monitoring</h1>
      <p>
        YOLO + ByteTrack person tracking, polygon ROI entry/exit events,
        CSV event logs, and Binary / Adaptive / Otsu image segmentation.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

monitor_tab, segmentation_tab = st.tabs(
    ["🎥 Security Monitoring", "🖼️ Image Segmentation"]
)

with monitor_tab:
    st.header("Intelligent Security Monitoring")

    with st.sidebar:
        st.header("Security input")
        source = st.radio(
            "Source",
            ["Sample folder", "Upload one or more videos"],
            key="monitor_source",
        )

        conf = st.slider(
            "YOLO confidence",
            0.10,
            0.90,
            0.35,
            0.05,
            key="monitor_conf",
        )
        process_every = st.slider(
            "Process every Nth frame",
            1,
            5,
            1,
            key="monitor_stride",
        )
        min_stable = st.slider(
            "Stable frames before ENTRY/EXIT",
            1,
            10,
            3,
            key="monitor_stable",
        )

        roi_default = json.dumps(
            [
                {
                    "name": "Main Entrance",
                    "points": [
                        [80, 80],
                        [1180, 80],
                        [1180, 620],
                        [80, 620],
                    ],
                }
            ],
            indent=2,
        )
        roi_text = st.text_area(
            "ROI polygon JSON",
            value=roi_default,
            height=200,
            key="roi_json",
        )

        monitor_run = st.button(
            "▶ Run security monitoring",
            type="primary",
            width="stretch",
        )

    video_paths: list[Path] = []

    if source == "Sample folder":
        video_paths = sorted(
            p for p in SAMPLE_VIDEO_DIR.glob("*")
            if p.suffix.lower() in VIDEO_EXTENSIONS
        ) if SAMPLE_VIDEO_DIR.exists() else []

        if video_paths:
            st.info(
                f"{len(video_paths)} video(s) found in sample_videos/. "
                "Running this mode processes the entire folder."
            )
            selected_names = st.multiselect(
                "Videos to process",
                [p.name for p in video_paths],
                default=[p.name for p in video_paths],
            )
            video_paths = [
                p for p in video_paths
                if p.name in selected_names
            ]
        else:
            st.warning(
                "No videos are present in sample_videos/. "
                "Add 5+ traffic/security videos or use the upload option."
            )
    else:
        uploads = st.file_uploader(
            "Upload one or more videos",
            type=sorted({e.lstrip(".") for e in VIDEO_EXTENSIONS}),
            accept_multiple_files=True,
            key="monitor_uploads",
        )
        if uploads:
            for upload in uploads:
                tmp = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(upload.name).suffix.lower(),
                )
                tmp.write(upload.getbuffer())
                tmp.close()
                video_paths.append(Path(tmp.name))

            st.info(
                f"{len(video_paths)} video(s) selected. "
                "They will all be processed in one run."
            )

    if video_paths:
        preview_path = video_paths[0]
        first = read_first_frame(preview_path)
        if first is not None:
            h, w = first.shape[:2]
            st.image(
                cv2.cvtColor(first, cv2.COLOR_BGR2RGB),
                caption=f"Preview: {preview_path.name} — {w} × {h}px",
                width="stretch",
            )

    if monitor_run:
        if not video_paths:
            st.error("No video selected.")
        else:
            try:
                rois = json.loads(roi_text)
                if not isinstance(rois, list) or not rois:
                    raise ValueError("ROI JSON must be a non-empty list.")

                progress = st.progress(0.0, text="Starting...")
                run_dir = OUTPUT_DIR / "security_monitoring"
                run_dir.mkdir(parents=True, exist_ok=True)

                results = []

                for index, path in enumerate(video_paths, start=1):
                    def child_progress(done, total):
                        overall = ((index - 1) + done / max(total, 1)) / len(video_paths)
                        progress.progress(
                            min(overall, 1.0),
                            text=f"Video {index}/{len(video_paths)} — frame {int(done)}/{int(total)}",
                        )

                    if source == "Sample folder":
                        source_for_processing = path
                    else:
                        source_for_processing = path

                    result = process_video(
                        source_for_processing,
                        rois,
                        conf=conf,
                        process_every=process_every,
                        min_stable_frames=min_stable,
                        output_dir=run_dir,
                        progress_cb=child_progress,
                    )
                    result["display_name"] = path.name
                    results.append(result)

                progress.empty()
                st.session_state["monitor_results"] = results
                st.success(
                    f"Completed {len(results)} video(s). "
                    "Processed MP4s and CSV logs are saved in outputs/security_monitoring/."
                )

            except Exception as exc:
                st.error(f"Monitoring failed: {exc}")

    results = st.session_state.get("monitor_results", [])

    if results:
        st.subheader("Batch results")

        summary = pd.DataFrame(
            [
                {
                    "Video": r["display_name"],
                    "Unique people": r["unique_people"],
                    "Entries": r["entries"],
                    "Exits": r["exits"],
                    "Maximum active": r["max_active"],
                    "Event records": len(r["events"]),
                    "Completed sessions": int((r["events"]["event"] == "EXIT").sum()),
                    "Frames": r["processed_frames"],
                }
                for r in results
            ]
        )
        st.dataframe(summary, width="stretch", hide_index=True)

        for result in results:
            with st.expander(result["display_name"], expanded=(len(results) == 1)):
                st.video(result["video_path"])

                d1, d2 = st.columns(2)
                with d1:
                    st.download_button(
                        "⬇ Download processed video",
                        data=Path(result["video_path"]).read_bytes(),
                        file_name=Path(result["video_path"]).name,
                        mime="video/mp4",
                        width="stretch",
                        key=f"video_{result['video_path']}",
                    )
                with d2:
                    st.download_button(
                        "⬇ Download event CSV",
                        data=Path(result["csv_path"]).read_bytes(),
                        file_name=Path(result["csv_path"]).name,
                        mime="text/csv",
                        width="stretch",
                        key=f"csv_{result['csv_path']}",
                    )

                st.dataframe(
                    result["events"],
                    width="stretch",
                    hide_index=True,
                )

with segmentation_tab:
    st.header("Image Segmentation")

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        key="segmentation_upload",
    )

    method = st.selectbox(
        "Segmentation method",
        METHODS,
    )
    threshold = st.slider(
        "Binary threshold",
        0,
        255,
        127,
    )
    block_size = st.slider(
        "Adaptive block size",
        3,
        51,
        11,
        2,
    )
    c_value = st.slider(
        "Adaptive C",
        -20,
        20,
        2,
    )

    if uploaded_image:
        raw = np.frombuffer(
            uploaded_image.getvalue(),
            dtype=np.uint8,
        )
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Could not decode the image.")
        else:
            output = segment_image(
                image,
                method,
                threshold=threshold,
                block_size=block_size,
                c_value=c_value,
            )

            left, right = st.columns(2)
            with left:
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    caption="Original",
                    width="stretch",
                )
            with right:
                st.image(
                    output,
                    caption=f"{method} segmentation",
                    width="stretch",
                )

            ok, encoded = cv2.imencode(".png", output)
            if ok:
                OUTPUT_DIR.joinpath("segmentation").mkdir(
                    parents=True,
                    exist_ok=True,
                )
                output_path = OUTPUT_DIR / "segmentation" / (
                    f"{Path(uploaded_image.name).stem}_{method.lower()}.png"
                )
                output_path.write_bytes(encoded.tobytes())

                st.download_button(
                    "⬇ Download segmented image",
                    data=encoded.tobytes(),
                    file_name=output_path.name,
                    mime="image/png",
                    width="stretch",
                )

                st.caption(f"Saved locally to: {output_path}")
