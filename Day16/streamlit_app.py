# ==========================================================
# MLB Summer Internship - Day 16
# Vision Studio
# Professional Image Processing Toolkit
# ==========================================================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import datetime

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(

    page_title="Vision Studio",

    page_icon="🖼️",

    layout="wide",

    initial_sidebar_state="expanded"

)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""

<style>

.main{
    background:#f6f8fb;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.hero{

    background:linear-gradient(90deg,#0f172a,#2563eb);

    padding:30px;

    border-radius:18px;

    color:white;

    margin-bottom:25px;

}

.metric-card{

    background:white;

    padding:15px;

    border-radius:15px;

    box-shadow:0px 4px 12px rgba(0,0,0,0.12);

}

.footer{

    text-align:center;

    color:gray;

    font-size:14px;

    margin-top:40px;

}

</style>

""", unsafe_allow_html=True)

# ==========================================================
# SESSION STATE
# ==========================================================

if "original_image" not in st.session_state:

    st.session_state.original_image = None

if "processed_image" not in st.session_state:

    st.session_state.processed_image = None

if "operations" not in st.session_state:

    st.session_state.operations = []

if "history" not in st.session_state:

    st.session_state.history = []

# ==========================================================
# HERO SECTION
# ==========================================================

st.markdown("""

<div class="hero">

<h1>🖼️ Vision Studio</h1>

<h3>Professional OpenCV Image Processing Toolkit</h3>

<p>

Upload an image and perform professional image processing
operations including resizing, cropping, flipping,
rotation, drawing, brightness enhancement,
color conversion and more.

</p>

</div>

""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Vision Studio")

page = st.sidebar.radio(

    "Navigation",

    [

        "🖼️ Image Toolkit",

        "📊 Image Information",

        "🎨 Drawing Tools",

        "⚙️ Color Processing",

        "ℹ️ About"

    ]

)

st.sidebar.markdown("---")

st.sidebar.success("OpenCV Toolkit")

st.sidebar.info("MLB Internship - Day 16")

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def load_image(uploaded_file):

    image = Image.open(uploaded_file).convert("RGB")

    image = np.array(image)

    image = cv2.cvtColor(

        image,

        cv2.COLOR_RGB2BGR

    )

    return image


def show_images():

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        original = cv2.cvtColor(

            st.session_state.original_image,

            cv2.COLOR_BGR2RGB

        )

        st.image(

            original,

            use_container_width=True

        )

    with col2:

        st.subheader("Processed Image")

        processed = cv2.cvtColor(

            st.session_state.processed_image,

            cv2.COLOR_BGR2RGB

        )

        st.image(

            processed,

            use_container_width=True

        )


def reset_image():

    st.session_state.processed_image = (

        st.session_state.original_image.copy()

    )

    st.session_state.operations = []

    st.session_state.history = []


def save_operation(name):

    st.session_state.operations.append(name)


def save_history():

    st.session_state.history.append(

        st.session_state.processed_image.copy()

    )


def download_image():

    temp = tempfile.NamedTemporaryFile(

        delete=False,

        suffix=".jpg"

    )

    cv2.imwrite(

        temp.name,

        st.session_state.processed_image

    )

    with open(temp.name, "rb") as file:

        st.download_button(

            "⬇ Download Image",

            file,

            file_name="processed_image.jpg",

            mime="image/jpeg",

            use_container_width=True

        )


def render_action_bar():

    col_undo, col_reset, col_download = st.columns(3)

    with col_undo:

        if st.button(

            "↩ Undo",

            use_container_width=True

        ):

            if len(st.session_state.history) > 0:

                st.session_state.processed_image = (

                    st.session_state.history.pop()

                )

                if len(st.session_state.operations) > 0:

                    st.session_state.operations.pop()

                st.rerun()

            else:

                st.warning("Nothing to undo.")

    with col_reset:

        if st.button(

            "Reset Image",

            use_container_width=True

        ):

            reset_image()

            st.rerun()

    with col_download:

        download_image()

# ==========================================================
# IMAGE TOOLKIT
# ==========================================================

if page == "🖼️ Image Toolkit":

    st.header("Image Processing Toolkit")

    uploaded_file = st.file_uploader(

        "Upload an Image",

        type=["jpg","jpeg","png","bmp"]

    )

    if uploaded_file is not None:

        if st.session_state.original_image is None:

            image = load_image(uploaded_file)

            st.session_state.original_image = image

            st.session_state.processed_image = image.copy()

        show_images()

        st.divider()

        st.subheader("Image Operations")

        col1, col2 = st.columns(2)

        # ======================================================
        # Resize
        # ======================================================

        with col1:

            st.markdown("### Resize")

            resize_width = st.number_input(

                "Width",

                min_value=1,

                max_value=3000,

                value=st.session_state.processed_image.shape[1]

            )

            resize_height = st.number_input(

                "Height",

                min_value=1,

                max_value=3000,

                value=st.session_state.processed_image.shape[0]

            )

            if st.button("Resize Image", use_container_width=True):

                save_history()

                resized = cv2.resize(

                    st.session_state.processed_image,

                    (resize_width, resize_height)

                )

                st.session_state.processed_image = resized

                save_operation("Resize")

                st.rerun()

        # ======================================================
        # Rotation
        # ======================================================

        with col2:

            st.markdown("### Rotate")

            rotation = st.selectbox(

                "Angle",

                [

                    "90°",

                    "180°",

                    "270°"

                ]

            )

            if st.button("Rotate Image", use_container_width=True):

                save_history()

                if rotation == "90°":

                    rotated = cv2.rotate(

                        st.session_state.processed_image,

                        cv2.ROTATE_90_CLOCKWISE

                    )

                elif rotation == "180°":

                    rotated = cv2.rotate(

                        st.session_state.processed_image,

                        cv2.ROTATE_180

                    )

                else:

                    rotated = cv2.rotate(

                        st.session_state.processed_image,

                        cv2.ROTATE_90_COUNTERCLOCKWISE

                    )

                st.session_state.processed_image = rotated

                save_operation("Rotate")

                st.rerun()

        st.divider()

        col3, col4 = st.columns(2)

        # ======================================================
        # Flip
        # ======================================================

        with col3:

            st.markdown("### Flip")

            flip_option = st.selectbox(

                "Flip Type",

                [

                    "Horizontal",

                    "Vertical",

                    "Both"

                ]

            )

            if st.button("Flip Image", use_container_width=True):

                save_history()

                if flip_option == "Horizontal":

                    flipped = cv2.flip(

                        st.session_state.processed_image,

                        1

                    )

                elif flip_option == "Vertical":

                    flipped = cv2.flip(

                        st.session_state.processed_image,

                        0

                    )

                else:

                    flipped = cv2.flip(

                        st.session_state.processed_image,

                        -1

                    )

                st.session_state.processed_image = flipped

                save_operation("Flip")

                st.rerun()

        # ======================================================
        # Crop
        # ======================================================

        with col4:

            st.markdown("### Center Crop")

            crop_percent = st.slider(

                "Crop Size",

                30,

                90,

                60

            )

            if st.button("Crop Image", use_container_width=True):

                save_history()

                img = st.session_state.processed_image

                h, w = img.shape[:2]

                new_w = int(w * crop_percent / 100)

                new_h = int(h * crop_percent / 100)

                x = (w - new_w) // 2

                y = (h - new_h) // 2

                cropped = img[

                    y:y+new_h,

                    x:x+new_w

                ]

                st.session_state.processed_image = cropped

                save_operation("Crop")

                st.rerun()

        st.divider()

        st.subheader("Processing Summary")

        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(

                "Width",

                st.session_state.processed_image.shape[1]

            )

        with m2:

            st.metric(

                "Height",

                st.session_state.processed_image.shape[0]

            )

        with m3:

            st.metric(

                "Operations",

                len(st.session_state.operations)

            )

        st.write("Operations Applied:")

        if len(st.session_state.operations) == 0:

            st.info("No operations performed yet.")

        else:

            st.success(" → ".join(st.session_state.operations))

        st.divider()

        render_action_bar()

# ==========================================================
# DRAWING TOOLS
# ==========================================================

elif page == "🎨 Drawing Tools":

    st.header("🎨 Drawing Tools")

    if st.session_state.processed_image is None:

        st.warning("Please upload an image first.")

    else:

        show_images()

        image = st.session_state.processed_image.copy()

        st.divider()

        tool = st.selectbox(

            "Choose Drawing Tool",

            [

                "Rectangle",

                "Circle",

                "Line",

                "Polygon",

                "Custom Text"

            ]

        )

        color = st.color_picker(

            "Choose Color",

            "#00FF00"

        )

        rgb = tuple(

            int(color[i:i+2], 16)

            for i in (1, 3, 5)

        )

        bgr = (rgb[2], rgb[1], rgb[0])

        # ==========================================
        # Rectangle
        # ==========================================

        if tool == "Rectangle":

            thickness = st.slider(

                "Thickness",

                1,

                20,

                3

            )

            if st.button("Draw Rectangle", use_container_width=True):

                save_history()

                h, w = image.shape[:2]

                cv2.rectangle(

                    image,

                    (50,50),

                    (w-50,h-50),

                    bgr,

                    thickness

                )

                st.session_state.processed_image = image

                save_operation("Rectangle")

                st.rerun()

        # ==========================================
        # Circle
        # ==========================================

        elif tool == "Circle":

            radius = st.slider(

                "Radius",

                20,

                300,

                100

            )

            thickness = st.slider(

                "Thickness",

                1,

                20,

                4

            )

            if st.button("Draw Circle", use_container_width=True):

                save_history()

                h, w = image.shape[:2]

                cv2.circle(

                    image,

                    (w//2, h//2),

                    radius,

                    bgr,

                    thickness

                )

                st.session_state.processed_image = image

                save_operation("Circle")

                st.rerun()

        # ==========================================
        # Line
        # ==========================================

        elif tool == "Line":

            thickness = st.slider(

                "Line Thickness",

                1,

                20,

                4

            )

            if st.button("Draw Line", use_container_width=True):

                save_history()

                h, w = image.shape[:2]

                cv2.line(

                    image,

                    (0,0),

                    (w,h),

                    bgr,

                    thickness

                )

                st.session_state.processed_image = image

                save_operation("Line")

                st.rerun()

        # ==========================================
        # Polygon
        # ==========================================

        elif tool == "Polygon":

            thickness = st.slider(

                "Thickness",

                1,

                20,

                4

            )

            if st.button("Draw Polygon", use_container_width=True):

                save_history()

                pts = np.array(

                    [

                        [200,80],

                        [350,180],

                        [300,350],

                        [120,300]

                    ],

                    np.int32

                )

                pts = pts.reshape((-1,1,2))

                cv2.polylines(

                    image,

                    [pts],

                    True,

                    bgr,

                    thickness

                )

                st.session_state.processed_image = image

                save_operation("Polygon")

                st.rerun()

        # ==========================================
        # Text
        # ==========================================

        else:

            custom_text = st.text_input(

                "Enter Text",

                f"Hadeed Jalani | {datetime.date.today()}"

            )

            font_scale = st.slider(

                "Font Size",

                0.5,

                3.0,

                1.0

            )

            thickness = st.slider(

                "Text Thickness",

                1,

                10,

                2

            )

            if st.button("Add Text", use_container_width=True):

                save_history()

                cv2.putText(

                    image,

                    custom_text,

                    (40,60),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    font_scale,

                    bgr,

                    thickness

                )

                st.session_state.processed_image = image

                save_operation("Text")

                st.rerun()

        st.divider()

        render_action_bar()

# ==========================================================
# COLOR PROCESSING
# ==========================================================

elif page == "⚙️ Color Processing":

    st.header("⚙️ Color Processing")

    if st.session_state.processed_image is None:

        st.warning("Please upload an image first.")

    else:

        show_images()

        image = st.session_state.processed_image.copy()

        st.divider()

        operation = st.selectbox(

            "Select Operation",

            [

                "Grayscale",

                "Brightness",

                "Contrast",

                "RGB Comparison"

            ]

        )

        # ==========================================
        # Grayscale
        # ==========================================

        if operation == "Grayscale":

            if st.button("Convert", use_container_width=True):

                save_history()

                gray = cv2.cvtColor(

                    image,

                    cv2.COLOR_BGR2GRAY

                )

                gray = cv2.cvtColor(

                    gray,

                    cv2.COLOR_GRAY2BGR

                )

                st.session_state.processed_image = gray

                save_operation("Grayscale")

                st.rerun()

        # ==========================================
        # Brightness
        # ==========================================

        elif operation == "Brightness":

            value = st.slider(

                "Brightness",

                -100,

                100,

                20

            )

            if st.button("Apply", use_container_width=True):

                save_history()

                bright = cv2.convertScaleAbs(

                    image,

                    alpha=1,

                    beta=value

                )

                st.session_state.processed_image = bright

                save_operation("Brightness")

                st.rerun()

        # ==========================================
        # Contrast
        # ==========================================

        elif operation == "Contrast":

            alpha = st.slider(

                "Contrast",

                0.5,

                3.0,

                1.5

            )

            if st.button("Apply", use_container_width=True):

                save_history()

                contrast = cv2.convertScaleAbs(

                    image,

                    alpha=alpha,

                    beta=0

                )

                st.session_state.processed_image = contrast

                save_operation("Contrast")

                st.rerun()

        # ==========================================
        # RGB Comparison
        # ==========================================

        else:

            rgb = cv2.cvtColor(

                image,

                cv2.COLOR_BGR2RGB

            )

            c1,c2 = st.columns(2)

            with c1:

                st.subheader("BGR")

                st.image(

                    cv2.cvtColor(image,cv2.COLOR_BGR2RGB),

                    use_container_width=True

                )

            with c2:

                st.subheader("RGB")

                st.image(

                    rgb,

                    use_container_width=True

                )

            st.info(

                "OpenCV stores images in BGR format internally. RGB is mainly used for visualization."

            )

        st.divider()

        render_action_bar()

# ==========================================================
# IMAGE INFORMATION
# ==========================================================

elif page == "📊 Image Information":

    st.header("📊 Image Information")

    if st.session_state.processed_image is None:

        st.warning("Please upload an image first.")

    else:

        image = st.session_state.processed_image

        show_images()

        height, width = image.shape[:2]

        channels = image.shape[2]

        image_size = round(image.nbytes / 1024, 2)

        st.divider()

        m1, m2, m3, m4 = st.columns(4)

        with m1:

            st.metric("Width", width)

        with m2:

            st.metric("Height", height)

        with m3:

            st.metric("Channels", channels)

        with m4:

            st.metric("Image Size", f"{image_size} KB")

        st.divider()

        st.subheader("Image Details")

        info = {

            "Property":[

                "Width",

                "Height",

                "Channels",

                "Color Format",

                "Data Type"

            ],

            "Value":[

                width,

                height,

                channels,

                "BGR (OpenCV)",

                image.dtype

            ]

        }

        st.table(info)

        st.divider()

        st.subheader("Operations Applied")

        if len(st.session_state.operations)==0:

            st.info("No processing operations performed.")

        else:

            st.success(

                " ➜ ".join(st.session_state.operations)

            )

        st.divider()

        render_action_bar()

# ==========================================================
# ABOUT
# ==========================================================

elif page == "ℹ️ About":

    st.header("ℹ️ About Vision Studio")

    st.markdown("""

### Vision Studio

Vision Studio is a professional image processing toolkit developed as part of the **MLB Summer Internship – Day 16**.

The application demonstrates the fundamental concepts of OpenCV by allowing users to interactively manipulate images through a clean web interface.

---

### Features

Upload Image

Undo Support

Reset Image

Resize

Crop

Rotate

Flip

Draw Rectangle

Draw Circle

Draw Line

Draw Polygon

Add Custom Text

Convert to Grayscale

Brightness Adjustment

Contrast Adjustment

RGB vs BGR Comparison

Download Processed Image

---

### Technologies Used

- Python

- OpenCV

- Streamlit

- NumPy

- Pillow

---

### Learning Outcomes

This project demonstrates how images are represented and manipulated using OpenCV.

It also serves as the foundation before moving towards advanced Computer Vision topics such as Feature Detection, Image Segmentation and Object Detection.

""")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(

"""
<div class="footer">

Made by Hadeed Jalani | 2026 | Vision Studio • MLB Internship Day 16
Built using OpenCV • Streamlit • Python

</div>

""",

unsafe_allow_html=True

)