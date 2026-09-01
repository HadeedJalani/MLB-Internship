import os
import sys
import tempfile
import time

# Ensure current directory is in Python path for module resolution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

try:
    from counter.vehicle_counter import VehicleCounter
except ModuleNotFoundError:
    from vehicle_counter import VehicleCounter


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Smart Vehicle Counting System",
    page_icon="🚦",
    layout="wide",
)


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #777;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    """
    <div class="main-title">
        🚦 Smart Vehicle Counting System
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Detect, track, and count vehicles crossing a virtual
        counting line using YOLO11 and ByteTrack / BoT-SORT.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

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

tracker_options = [
    "ByteTrack",
    "BoT-SORT",
]

tracker_name = st.sidebar.selectbox(
    "Tracking Algorithm",
    tracker_options,
    index=0,
    key="tracker_select_box",
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Supported Vehicles**

    🚗 Car  
    🚌 Bus  
    🚛 Truck  
    🏍 Motorcycle  

    Vehicles are counted once when their
    tracking ID crosses the virtual line.
    """
)


# ---------------------------------------------------------
# Upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload Traffic Video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv",
    ],
)


if uploaded_file is None:

    st.info(
        "Upload a traffic video to begin vehicle counting."
    )

    st.stop()


# ---------------------------------------------------------
# Save uploaded video
# ---------------------------------------------------------

input_temp = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".mp4",
)

input_temp.write(
    uploaded_file.getbuffer()
)

input_temp.close()


# ---------------------------------------------------------
# Display Input
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">📹 Input Video</div>',
    unsafe_allow_html=True,
)

st.video(
    uploaded_file
)


# ---------------------------------------------------------
# Start processing
# ---------------------------------------------------------

if st.button(
    "🚀 Start Vehicle Counting",
    type="primary",
    use_container_width=True,
):

    output_temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4",
    )

    output_temp.close()

    progress_bar = st.progress(0)

    status_text = st.empty()

    status_text.info(
        "Initializing YOLO model..."
    )

    try:

        counter = VehicleCounter(
            model_path="yolo11n.pt",
            confidence=confidence,
            tracker=tracker_name,
        )

        status_text.info(
            "Processing video and tracking vehicles..."
        )

        start_time = time.perf_counter()

        def progress_callback(progress):

            progress_bar.progress(
                int(progress * 100)
            )

        stats = counter.process_video(
            input_path=input_temp.name,
            output_path=output_temp.name,
            progress_callback=progress_callback,
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        progress_bar.progress(100)

        status_text.success(
            "Vehicle counting completed successfully!"
        )

        # -------------------------------------------------
        # Results
        # -------------------------------------------------

        st.markdown("---")

        st.subheader(
            "🎬 Processing Results"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### 📥 Original Video"
            )

            st.video(
                input_temp.name
            )

        with col2:

            st.markdown(
                "### 📤 Vehicle Counting Output"
            )

            st.video(
                output_temp.name
            )

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        st.markdown("---")

        st.subheader(
            "📊 Counting Statistics"
        )

        metric1, metric2, metric3, metric4 = (
            st.columns(4)
        )

        with metric1:

            st.metric(
                "Total Vehicles",
                stats["total_vehicles"],
            )

        with metric2:

            st.metric(
                "Unique Tracking IDs",
                stats[
                    "unique_tracking_ids"
                ],
            )

        with metric3:

            st.metric(
                "Frames Processed",
                stats[
                    "total_frames"
                ],
            )

        with metric4:

            st.metric(
                "Processing Time",
                f"{elapsed:.2f}s",
            )

        # -------------------------------------------------
        # Per vehicle counts
        # -------------------------------------------------

        st.markdown("---")

        st.subheader(
            "🚗 Vehicle Breakdown"
        )

        vehicle_columns = st.columns(4)

        vehicles = [
            ("🚗", "Car", "car"),
            ("🚌", "Bus", "bus"),
            ("🚛", "Truck", "truck"),
            ("🏍️", "Motorcycle", "motorcycle"),
        ]

        for column, (
            icon,
            display_name,
            vehicle_name,
        ) in zip(
            vehicle_columns,
            vehicles,
        ):

            count = (
                stats["vehicle_counts"]
                .get(
                    vehicle_name,
                    0,
                )
            )

            with column:

                st.metric(
                    f"{icon} {display_name}",
                    count,
                )

        # -------------------------------------------------
        # Direction statistics
        # -------------------------------------------------

        st.markdown("---")

        st.subheader(
            "↕️ Direction-wise Counting"
        )

        direction_col1, direction_col2 = (
            st.columns(2)
        )

        with direction_col1:

            st.markdown(
                "### ⬆️ Upward Movement"
            )

            if stats["up_counts"]:

                st.json(
                    stats["up_counts"]
                )

            else:

                st.info(
                    "No vehicles counted upward."
                )

        with direction_col2:

            st.markdown(
                "### ⬇️ Downward Movement"
            )

            if stats["down_counts"]:

                st.json(
                    stats["down_counts"]
                )

            else:

                st.info(
                    "No vehicles counted downward."
                )

        # -------------------------------------------------
        # Download
        # -------------------------------------------------

        st.markdown("---")

        with open(
            output_temp.name,
            "rb",
        ) as video_file:

            video_bytes = (
                video_file.read()
            )

        st.download_button(
            label=(
                "⬇️ Download Processed Video"
            ),
            data=video_bytes,
            file_name=(
                "vehicle_counting_output.mp4"
            ),
            mime="video/mp4",
            use_container_width=True,
        )

    except Exception as error:

        st.error(
            "Vehicle counting failed."
        )

        st.exception(error)

    finally:

        # Do not delete output immediately because
        # Streamlit may still need it for playback.

        pass