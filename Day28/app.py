"""
Day 28 Construction Equipment Detection
Streamlit app: upload an image or video, run inference with the custom-trained
YOLO model, view detections with confidence scores, download the result.
"""

import os
import tempfile
import time

import cv2
import streamlit as st
from ultralytics import YOLO

MODEL_PATH = os.path.join("models", "best.pt")


# ---------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(model_path: str) -> YOLO:
    if not os.path.exists(model_path):
        st.error(
            f"Model file not found at `{model_path}`. "
            "Make sure `best.pt` from Colab training is placed in the `models/` folder."
        )
        st.stop()
    return YOLO(model_path)


def run_image_inference(model: YOLO, image_path: str, conf: float, imgsz: int):
    results = model.predict(source=image_path, conf=conf, imgsz=imgsz, save=False)
    result = results[0]
    annotated = result.plot()  # BGR numpy array with boxes drawn
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    detections = [
        {
            "class": model.names[int(box.cls)],
            "confidence": float(box.conf),
        }
        for box in result.boxes
    ]
    return annotated_rgb, detections


def run_video_inference(model: YOLO, video_path: str, conf: float, imgsz: int, progress_bar):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    all_detections = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)
        result = results[0]
        annotated = result.plot()
        writer.write(annotated)

        for box in result.boxes:
            all_detections.append(model.names[int(box.cls)])

        frame_idx += 1
        progress_bar.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    writer.release()
    return out_path, all_detections


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Construction Equipment Detection", layout="wide")
st.title("🏗️ Construction Equipment Detection")
st.caption("Custom-trained YOLO model — upload an image or video to detect construction equipment.")

with st.sidebar:
    st.header("Settings")
    conf_threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    img_size = st.selectbox("Inference image size", [640, 960, 1280], index=0)
    st.markdown("---")
    st.markdown("**Classes detected:**")
    _model_preview = load_model(MODEL_PATH)
    for name in _model_preview.names.values():
        st.markdown(f"- {name}")

model = load_model(MODEL_PATH)

tab_image, tab_video = st.tabs(["📷 Image", "🎥 Video"])

# --- Image tab ---
with tab_image:
    uploaded_image = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"], key="image_uploader"
    )
    if uploaded_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_image.name)[1]) as tmp:
            tmp.write(uploaded_image.read())
            tmp_path = tmp.name

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(tmp_path, use_column_width=True)

        with st.spinner("Running inference..."):
            annotated_rgb, detections = run_image_inference(model, tmp_path, conf_threshold, img_size)

        with col2:
            st.subheader("Detections")
            st.image(annotated_rgb, use_column_width=True)

        if detections:
            st.markdown("**Detected objects:**")
            for d in detections:
                st.markdown(f"- `{d['class']}` — confidence: `{d['confidence']:.2f}`")
        else:
            st.info("No objects detected above the current confidence threshold.")

        out_img_path = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
        cv2.imwrite(out_img_path, cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
        with open(out_img_path, "rb") as f:
            st.download_button(
                "⬇️ Download annotated image",
                data=f,
                file_name=f"detected_{uploaded_image.name}",
                mime="image/jpeg",
            )

        os.remove(tmp_path)

# --- Video tab ---
with tab_video:
    uploaded_video = st.file_uploader(
        "Upload a video", type=["mp4", "mov", "avi", "mkv"], key="video_uploader"
    )
    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1]) as tmp:
            tmp.write(uploaded_video.read())
            tmp_path = tmp.name

        st.video(tmp_path)

        if st.button("Run detection on video"):
            progress_bar = st.progress(0.0)
            start = time.time()
            with st.spinner("Processing video frame by frame — this can take a while..."):
                out_path, all_detections = run_video_inference(
                    model, tmp_path, conf_threshold, img_size, progress_bar
                )
            elapsed = time.time() - start
            st.success(f"Done in {elapsed:.1f}s — {len(all_detections)} total detections across all frames.")

            st.subheader("Result")
            st.video(out_path)

            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇️ Download annotated video",
                    data=f,
                    file_name=f"detected_{uploaded_video.name.rsplit('.', 1)[0]}.mp4",
                    mime="video/mp4",
                )

        os.remove(tmp_path)

st.markdown("---")
st.caption("Day 28 Custom Object Detection System | YOLO11 trained on Roboflow construction equipment dataset")