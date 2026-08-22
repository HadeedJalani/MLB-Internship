import cv2
import numpy as np
import streamlit as st

from feature_detection import (
    detect_harris_corners,
    detect_orb_keypoints
)

from feature_matching import (
    match_orb_features
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Day 25 - Feature Detection",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #777;
        margin-bottom: 30px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        text-align: center;
        margin-bottom: 10px;
    }

    .metric-number {
        font-size: 30px;
        font-weight: 700;
    }

    .metric-label {
        color: #777;
        font-size: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🔍 Image Feature Detection & Matching</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Day 25 — Harris Corner Detection, ORB Keypoints & Feature Matching'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    nfeatures = st.slider(
        "ORB Features",
        min_value=100,
        max_value=3000,
        value=1000,
        step=100
    )

    harris_threshold = st.slider(
        "Harris Threshold",
        min_value=0.001,
        max_value=0.10,
        value=0.01,
        step=0.001
    )

    ratio_threshold = st.slider(
        "Match Ratio",
        min_value=0.50,
        max_value=0.95,
        value=0.75,
        step=0.05
    )

    st.divider()

    st.info(
        "ORB uses binary descriptors and Hamming distance "
        "for feature matching."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("📷 Upload Two Images")

col1, col2 = st.columns(2)

with col1:

    uploaded1 = st.file_uploader(
        "Upload Image 1",
        type=["jpg", "jpeg", "png"],
        key="image1"
    )

with col2:

    uploaded2 = st.file_uploader(
        "Upload Image 2",
        type=["jpg", "jpeg", "png"],
        key="image2"
    )


# ============================================================
# IMAGE CONVERSION
# ============================================================

def uploaded_to_cv2(uploaded_file):

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    return image


# ============================================================
# MAIN PROCESSING
# ============================================================

if uploaded1 is not None and uploaded2 is not None:

    image1 = uploaded_to_cv2(uploaded1)
    image2 = uploaded_to_cv2(uploaded2)

    if image1 is None or image2 is None:

        st.error(
            "Could not read one of the uploaded images."
        )

        st.stop()


    # --------------------------------------------------------
    # ORIGINAL IMAGES
    # --------------------------------------------------------

    st.subheader("🖼️ Original Images")

    original_col1, original_col2 = st.columns(2)

    with original_col1:

        st.image(
            cv2.cvtColor(image1, cv2.COLOR_BGR2RGB),
            caption="Image 1",
            width="stretch"
        )

    with original_col2:

        st.image(
            cv2.cvtColor(image2, cv2.COLOR_BGR2RGB),
            caption="Image 2",
            width="stretch"
        )


    st.divider()


    # ========================================================
    # HARRIS DETECTION
    # ========================================================

    st.subheader("📍 Harris Corner Detection")

    harris_col1, harris_col2 = st.columns(2)

    with st.spinner("Detecting Harris corners..."):

        harris1, harris_count1 = detect_harris_corners(
            image1,
            threshold=harris_threshold
        )

        harris2, harris_count2 = detect_harris_corners(
            image2,
            threshold=harris_threshold
        )


    with harris_col1:

        st.image(
            cv2.cvtColor(
                harris1,
                cv2.COLOR_BGR2RGB
            ),
            caption=f"Image 1 — {harris_count1:,} corners",
            width="stretch"
        )

    with harris_col2:

        st.image(
            cv2.cvtColor(
                harris2,
                cv2.COLOR_BGR2RGB
            ),
            caption=f"Image 2 — {harris_count2:,} corners",
            width="stretch"
        )


    # ========================================================
    # ORB DETECTION
    # ========================================================

    st.divider()

    st.subheader("🟢 ORB Keypoint Detection")

    orb_col1, orb_col2 = st.columns(2)

    with st.spinner("Detecting ORB keypoints..."):

        orb1, keypoints1, descriptors1 = detect_orb_keypoints(
            image1,
            nfeatures=nfeatures
        )

        orb2, keypoints2, descriptors2 = detect_orb_keypoints(
            image2,
            nfeatures=nfeatures
        )


    orb_count1 = len(keypoints1)
    orb_count2 = len(keypoints2)


    with orb_col1:

        st.image(
            cv2.cvtColor(
                orb1,
                cv2.COLOR_BGR2RGB
            ),
            caption=f"Image 1 — {orb_count1:,} keypoints",
            width="stretch"
        )

    with orb_col2:

        st.image(
            cv2.cvtColor(
                orb2,
                cv2.COLOR_BGR2RGB
            ),
            caption=f"Image 2 — {orb_count2:,} keypoints",
            width="stretch"
        )


    # ========================================================
    # FEATURE MATCHING
    # ========================================================

    st.divider()

    st.subheader("🔗 ORB Feature Matching")

    with st.spinner("Matching features..."):

        match_image, match_kp1, match_kp2, good_matches = (
            match_orb_features(
                image1,
                image2,
                nfeatures=nfeatures,
                ratio_threshold=ratio_threshold
            )
        )


    good_match_count = len(good_matches)


    # ========================================================
    # METRICS
    # ========================================================

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Image 1 Keypoints",
            orb_count1
        )

    with metric2:

        st.metric(
            "Image 2 Keypoints",
            orb_count2
        )

    with metric3:

        st.metric(
            "Good Matches",
            good_match_count
        )


    # ========================================================
    # MATCH RATIO
    # ========================================================

    if min(orb_count1, orb_count2) > 0:

        match_ratio = (
            good_match_count /
            min(orb_count1, orb_count2)
        ) * 100

    else:

        match_ratio = 0


    st.caption(
        f"Match ratio: {match_ratio:.2f}%"
    )


    # ========================================================
    # MATCH VISUALIZATION
    # ========================================================

    if good_match_count > 0:

        st.image(
            cv2.cvtColor(
                match_image,
                cv2.COLOR_BGR2RGB
            ),
            caption="ORB Feature Matches",
            width="stretch"
        )

    else:

        st.warning(
            "No good matches were found. "
            "Try similar images or increase the ORB feature count."
        )


    # ========================================================
    # HARRIS VS ORB
    # ========================================================

    st.divider()

    st.subheader("📊 Harris vs ORB")

    comparison_col1, comparison_col2 = st.columns(2)

    with comparison_col1:

        st.markdown("### Harris Corner Detection")

        st.write(
            f"Image 1 corners: **{harris_count1:,}**"
        )

        st.write(
            f"Image 2 corners: **{harris_count2:,}**"
        )

        st.write(
            "Harris detects corner-like structures based "
            "on changes in image intensity."
        )

    with comparison_col2:

        st.markdown("### ORB")

        st.write(
            f"Image 1 keypoints: **{orb_count1:,}**"
        )

        st.write(
            f"Image 2 keypoints: **{orb_count2:,}**"
        )

        st.write(
            f"Good matches: **{good_match_count:,}**"
        )

        st.write(
            "ORB detects keypoints and creates binary "
            "descriptors that can be matched between images."
        )


else:

    st.info(
        "👆 Upload Image 1 and Image 2 to begin."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Day 25 • Computer Vision • Harris + ORB + Brute Force Matching"
)