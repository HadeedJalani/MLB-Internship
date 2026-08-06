import os
import tempfile
import time
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="VisionAI | YOLOv8 Object Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Professional UI Enhancement
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #0F172A;
            color: #94A3B8;
            text-align: center;
            padding: 0.6rem 0;
            font-size: 0.85rem;
            z-index: 100;
            line-height: 1.3;
        }
        .stButton>button {
            width: 100%;
            border-radius: 6px;
            font-weight: 600;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 2. SESSION STATE INITIALIZATION
# ==========================================
if "detection_history" not in st.session_state:
  st.session_state.detection_history = []


# ==========================================
# 3. MODEL LOADING & CACHING
# ==========================================
@st.cache_resource
def load_yolo_model(model_size="yolov8n.pt"):
  """Load YOLOv8 model and cache it in resource memory."""
  return YOLO(model_size)


# ==========================================
# 4. SIDEBAR NAVIGATION & SETTINGS
# ==========================================
st.sidebar.image(
    "https://raw.githubusercontent.com/ultralytics/assets/main/logo/YOLOv8-Header-Title.png",
    use_container_width=True,
)
st.sidebar.title("Navigation & Controls")

page = st.sidebar.radio(
    "Go to",
    ["Object Detection", "Analytics Dashboard", "Model Details", "About Project"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Configuration")

model_choice = st.sidebar.selectbox(
    "Select YOLOv8 Variant",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
    help="Nano is fastest; Medium provides higher accuracy at the cost of speed.",
)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Minimum confidence score required to display a detected object.",
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold (NMS)",
    min_value=0.1,
    max_value=1.0,
    value=0.45,
    step=0.05,
    help="Intersection over Union threshold for Non-Maximum Suppression.",
)

# Load selected model instance
try:
  model = load_yolo_model(model_choice)
  st.sidebar.markdown("### Active Model Specs")
  st.sidebar.metric("Model", model_choice.replace(".pt", "").upper())
  st.sidebar.metric("Classes", len(model.names))
except Exception as e:
  st.sidebar.error(f"Failed to load model: {e}")
  st.stop()


# ==========================================
# 5. HELPER FUNCTIONS
# ==========================================
def process_detection_results(results, source_type, file_name):
  """Extract detection bounding box metrics, save to session state, and return Pandas DataFrame."""
  records = []
  boxes = results[0].boxes

  for box in boxes:
    cls_id = int(box.cls[0].item())
    class_name = model.names[cls_id]
    confidence = float(box.conf[0].item())
    xyxy = box.xyxy[0].tolist()

    record = {
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Source Type": source_type,
        "File Name": file_name,
        "Class": class_name,
        "Confidence": round(confidence, 4),
        "X1": round(xyxy[0], 1),
        "Y1": round(xyxy[1], 1),
        "X2": round(xyxy[2], 1),
        "Y2": round(xyxy[3], 1),
    }
    records.append(record)
    st.session_state.detection_history.append(record)

  return pd.DataFrame(records)


def convert_pil_to_bytes(img, fmt="JPEG"):
  """Utility to convert PIL Image into downloadable byte stream."""
  buf = BytesIO()
  img.save(buf, format=fmt)
  return buf.getvalue()


# ==========================================
# 6. PAGE 1: OBJECT DETECTION (IMAGE & VIDEO)
# ==========================================
if page == "Object Detection":
  st.markdown(
      '<div class="main-header">Real-Time Object Detection System</div>',
      unsafe_allow_html=True,
  )
  st.markdown(
      '<div class="sub-header">Upload an image or video file to run YOLOv8 object detection, view metric breakdowns, and export outputs.</div>',
      unsafe_allow_html=True,
  )

  tab_img, tab_vid = st.tabs(["🖼️ Image Detection", "🎥 Video Detection"])

  # ----------------------------------------
  # TAB 1: IMAGE DETECTION
  # ----------------------------------------
  with tab_img:
    uploaded_image = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_uploader",
    )

    if uploaded_image is not None:
      col_orig, col_det = st.columns(2)

      image = Image.open(uploaded_image).convert("RGB")

      with col_orig:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

      # Precise inference timing
      start_time = time.perf_counter()
      with st.spinner("Processing image through YOLOv8 pipeline..."):
        results = model.predict(
            source=image, conf=conf_threshold, iou=iou_threshold
        )
      end_time = time.perf_counter()
      inference_time_ms = (end_time - start_time) * 1000

      res_plotted = results[0].plot()
      annotated_image = Image.fromarray(res_plotted[:, :, ::-1])

      with col_det:
        st.subheader("Detected Output")
        st.image(annotated_image, use_container_width=True)

      # Extract Detections Data
      df_detections = process_detection_results(
          results, "Image", uploaded_image.name
      )

      st.markdown("---")
      st.subheader("Detection Summary Cards")

      total_objects = len(df_detections)
      unique_classes = (
          df_detections["Class"].nunique() if total_objects > 0 else 0
      )
      avg_conf = (
          df_detections["Confidence"].mean() if total_objects > 0 else 0.0
      )

      m1, m2, m3, m4, m5 = st.columns(5)
      m1.metric("Total Objects", total_objects)
      m2.metric("Unique Classes", unique_classes)
      m3.metric("Avg Confidence", f"{avg_conf:.1%}")
      m4.metric("Inference Time", f"{inference_time_ms:.1f} ms")
      m5.metric(
          "Model Variant", model_choice.replace(".pt", "").upper()
      )

      if total_objects > 0:
        st.markdown("### Detailed Detection Log")
        st.dataframe(
            df_detections[
                ["Class", "Confidence", "X1", "Y1", "X2", "Y2"]
            ],
            use_container_width=True,
        )

        st.markdown("### Export Results")
        d_col1, d_col2 = st.columns(2)

        with d_col1:
          img_bytes = convert_pil_to_bytes(annotated_image, fmt="JPEG")
          st.download_button(
              label="📥 Download Annotated Image",
              data=img_bytes,
              file_name=f"detected_{uploaded_image.name}",
              mime="image/jpeg",
          )

        with d_col2:
          csv_data = df_detections.to_csv(index=False).encode("utf-8")
          st.download_button(
              label="📊 Download Detections CSV",
              data=csv_data,
              file_name=f"detections_{uploaded_image.name}.csv",
              mime="text/csv",
          )
      else:
        st.info("No objects detected at the current confidence threshold.")

  # ----------------------------------------
  # TAB 2: VIDEO DETECTION
  # ----------------------------------------
  with tab_vid:
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
    )

    if uploaded_video is not None:
      tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
      tfile.write(uploaded_video.read())

      v_col1, v_col2 = st.columns(2)

      with v_col1:
        st.subheader("Original Video")
        st.video(tfile.name)

      process_btn = st.button("▶️ Process Video Frame-by-Frame")

      if process_btn:
        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ).name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(
            out_path, fourcc, fps, (width, height)
        )

        progress_bar = st.progress(0)
        status_text = st.empty()

        frame_count = 0
        all_video_records = []

        while cap.isOpened():
          ret, frame = cap.read()
          if not ret:
            break

          frame_count += 1
          results = model.predict(
              source=frame,
              conf=conf_threshold,
              iou=iou_threshold,
              verbose=False,
          )
          annotated_frame = results[0].plot()
          out_writer.write(annotated_frame)

          for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            class_name = model.names[cls_id]
            confidence = float(box.conf[0].item())
            all_video_records.append({
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Source Type": "Video",
                "File Name": uploaded_video.name,
                "Class": class_name,
                "Confidence": round(confidence, 4),
            })

          if total_frames > 0:
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.info(
                f"Processing Frame {frame_count}/{total_frames}"
            )

        # Properly indented inside the process block
        cap.release()
        out_writer.release()

        status_text.success("Video processing completed successfully!")

        # Summarize video detections into Session History (prevents session bloat)
        df_vid = pd.DataFrame(all_video_records)
        v_total = len(df_vid)

        if v_total > 0:
          for cls_name, grp in df_vid.groupby("Class"):
            st.session_state.detection_history.append({
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Source Type": "Video",
                "File Name": uploaded_video.name,
                "Class": cls_name,
                "Confidence": round(grp["Confidence"].mean(), 4),
                "X1": 0.0,
                "Y1": 0.0,
                "X2": 0.0,
                "Y2": 0.0,
            })
        else:
          st.session_state.detection_history.append({
              "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
              "Source Type": "Video",
              "File Name": uploaded_video.name,
              "Class": "None Detected",
              "Confidence": 0.0,
              "X1": 0.0,
              "Y1": 0.0,
              "X2": 0.0,
              "Y2": 0.0,
          })

        with v_col2:
          st.subheader("Detected Video Output")
          with open(out_path, "rb") as vf:
            video_bytes = vf.read()
          st.video(video_bytes)

        st.markdown("---")
        st.subheader("Video Detection Metrics")

        v_classes = df_vid["Class"].nunique() if v_total > 0 else 0
        v_avg_conf = (
            df_vid["Confidence"].mean() if v_total > 0 else 0.0
        )

        vm1, vm2, vm3, vm4 = st.columns(4)
        vm1.metric("Total Frames Processed", frame_count)
        vm2.metric("Total Detections Across Frames", v_total)
        vm3.metric("Unique Detected Classes", v_classes)
        vm4.metric("Avg Frame Detection Confidence", f"{v_avg_conf:.1%}")

        st.markdown("### Export Video Outputs")
        vd_col1, vd_col2 = st.columns(2)

        with vd_col1:
          st.download_button(
              label="📥 Download Annotated Video (MP4)",
              data=video_bytes,
              file_name=f"detected_{uploaded_video.name}",
              mime="video/mp4",
          )

        with vd_col2:
          if v_total > 0:
            v_csv_data = df_vid.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📊 Download Video Detections CSV",
                data=v_csv_data,
                file_name=f"detections_{uploaded_video.name}.csv",
                mime="text/csv",
            )


# ==========================================
# 7. PAGE 2: ANALYTICS DASHBOARD
# ==========================================
elif page == "Analytics Dashboard":
  st.markdown(
      '<div class="main-header">Detection Analytics & Session Insights</div>',
      unsafe_allow_html=True,
  )
  st.markdown(
      '<div class="sub-header">Interactive data visualizations generated from current session detection history.</div>',
      unsafe_allow_html=True,
  )

  if len(st.session_state.detection_history) == 0:
    st.warning(
        "No detection history recorded yet. Please run Object Detection on an image or video first."
    )
  else:
    df_history = pd.DataFrame(st.session_state.detection_history)

    st.subheader("Global Session Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Session Entries", len(df_history))
    c2.metric("Unique Classes Identified", df_history["Class"].nunique())
    c3.metric(
        "Overall Mean Confidence",
        f"{df_history['Confidence'].mean():.1%}",
    )
    c4.metric(
        "Files Processed", df_history["File Name"].nunique()
    )

    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
      st.subheader("Object Class Frequency")
      class_counts = df_history["Class"].value_counts().reset_index()
      class_counts.columns = ["Class", "Count"]

      fig_bar = px.bar(
          class_counts,
          x="Count",
          y="Class",
          orientation="h",
          color="Count",
          text="Count",
          color_continuous_scale="Viridis",
          title="Frequency of Detected Classes",
      )
      fig_bar.update_layout(
          showlegend=False, margin=dict(l=20, r=20, t=40, b=20)
      )
      st.plotly_chart(fig_bar, use_container_width=True)

    with chart_col2:
      st.subheader("Class Distribution")
      fig_pie = px.pie(
          df_history,
          names="Class",
          title="Percentage Share of Detected Categories",
          hole=0.4,
          color_discrete_sequence=px.colors.qualitative.Pastel,
      )
      fig_pie.update_traces(textposition="inside", textinfo="percent+label")
      fig_pie.update_layout(margin=dict(l=20, r=20, t=40, b=20))
      st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("Detection Confidence Distribution")
    fig_hist = px.histogram(
        df_history,
        x="Confidence",
        nbins=20,
        title="Confidence Score Spread Across Detections",
        color_discrete_sequence=["#6366F1"],
    )
    fig_hist.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    st.subheader("Full Session History Log")
    st.dataframe(df_history, use_container_width=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
      csv_session = df_history.to_csv(index=False).encode("utf-8")
      st.download_button(
          label="📥 Download Session Report (CSV)",
          data=csv_session,
          file_name="session_report.csv",
          mime="text/csv",
      )

    with col_s2:
      if st.button("🗑️ Clear Session History"):
        st.session_state.detection_history = []
        st.experimental_rerun()


# ==========================================
# 8. PAGE 3: MODEL DETAILS
# ==========================================
elif page == "Model Details":
  st.markdown(
      '<div class="main-header">YOLOv8 Architecture & Specifications</div>',
      unsafe_allow_html=True,
  )
  st.markdown(
      '<div class="sub-header">Technical specifications, runtime parameters, and class taxonomy for Ultralytics YOLOv8 models.</div>',
      unsafe_allow_html=True,
  )

  st.subheader("System Information")
  s1, s2, s3, s4 = st.columns(4)
  s1.metric("Framework", "Ultralytics")
  s2.metric("Backend", "PyTorch")
  s3.metric("Dataset", "COCO")
  s4.metric("Classes", len(model.names))

  st.markdown("---")
  col_m1, col_m2 = st.columns(2)

  with col_m1:
    st.subheader("Model Overview")
    st.write(
        """
        **YOLOv8** (You Only Look Once v8) is a state-of-the-art, real-time object detection and image segmentation model created by Ultralytics.
        
        Key innovations in YOLOv8 include:
        - **Anchor-Free Detection:** Predicts object centers directly, reducing bounding box proposals and speeding up NMS.
        - **New Backbone & Neck Network:** Uses C2f modules to replace C3, improving gradient flow and feature extraction.
        - **Decoupled Head:** Separates classification and regression branches for higher precision.
        """
    )

  with col_m2:
    st.subheader("YOLOv8 Family Comparison")
    model_data = {
        "Model": ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"],
        "Parameters (M)": [3.2, 11.2, 25.9, 43.7, 68.2],
        "FLOPs (B)": [8.7, 28.6, 78.9, 165.2, 257.8],
        "mAP 50-95 (COCO)": [37.3, 44.9, 50.2, 52.9, 53.9],
    }
    st.table(pd.DataFrame(model_data))

  st.markdown("---")
  st.subheader("Supported COCO Classes (80 Categories)")

  coco_classes = list(model.names.values())
  col_a, col_b, col_c, col_d = st.columns(4)

  quarter = len(coco_classes) // 4
  col_a.write("\n".join([f"- {c}" for c in coco_classes[:quarter]]))
  col_b.write(
      "\n".join([f"- {c}" for c in coco_classes[quarter : quarter * 2]])
  )
  col_c.write(
      "\n".join([f"- {c}" for c in coco_classes[quarter * 2 : quarter * 3]])
  )
  col_d.write("\n".join([f"- {c}" for c in coco_classes[quarter * 3 :]]))


# ==========================================
# 9. PAGE 4: ABOUT PROJECT
# ==========================================
elif page == "About Project":
  st.markdown(
      '<div class="main-header">About MLB Internship Day 15 Project</div>',
      unsafe_allow_html=True,
  )
  st.markdown(
      '<div class="sub-header">Production-ready Streamlit Application with YOLOv8 Computer Vision Pipeline.</div>',
      unsafe_allow_html=True,
  )

  st.markdown(
      """
    ### Project Scope
    This application fulfills the **MLB Internship Day 15** milestone requirements, establishing an end-to-end computer vision web platform. It enables seamless image and video object detection, real-time analytics generation, and structured reporting.

    ### Key Features
    1. **Dual Detection Engines:** Frame-by-frame video processing and static image object detection.
    2. **Configurable Hyperparameters:** Real-time user adjustment of Confidence and IoU non-maximum suppression thresholds.
    3. **Automated Export Pipelines:** Download annotated media (JPEG, MP4) along with bounding box coordinate logs in CSV format.
    4. **Session State Analytics:** Interactive Plotly dashboards summarizing object frequencies and confidence distributions.
    5. **Production UI Quality:** Styled sidebar navigation, responsive layout grids, and metric summary cards.

    ### Technology Stack
    - **Developer:** Hadeed Jalani
    - **Frontend / Framework:** Streamlit
    - **Object Detection Model:** Ultralytics YOLOv8
    - **Computer Vision & Video Processing:** OpenCV (`opencv-python`), Pillow
    - **Data Processing & Analytics:** Pandas, NumPy, Plotly Express
    """
  )


# ==========================================
# 10. GLOBAL FOOTER
# ==========================================
st.markdown(
    """
    <div class="footer">
        Developed by <b>Hadeed Jalani</b> | MLB Summer Internship<br>
        Day 15 • Object Detection using YOLOv8 | Powered by Streamlit & Ultralytics
    </div>
""",
    unsafe_allow_html=True,
)