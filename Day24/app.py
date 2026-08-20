import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import streamlit as st

from ocr import easyocr_engine
from ocr import paddleocr_engine
from ocr import rapidocr_engine

from utils import image_processing
from utils import result_processing


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Document OCR Studio",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# OCR ENGINE REGISTRY
# ============================================================

OCR_ENGINES = {
    "EasyOCR": easyocr_engine,
    "PaddleOCR": paddleocr_engine,
    "RapidOCR": rapidocr_engine,
}


# ============================================================
# IMPORTANT:
# PaddleOCR should NOT run multiple inference jobs at once.
#
# RapidOCR is lightweight enough for parallel processing.
#
# EasyOCR can handle limited parallel workers.
# ============================================================

MAX_WORKERS_PER_ENGINE = {
    "EasyOCR": 2,
    "PaddleOCR": 1,
    "RapidOCR": 3,
}


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.8rem;
        border-radius: 18px;

        background:
            linear-gradient(
                135deg,
                #0f172a 0%,
                #1e293b 100%
            );

        border:
            1px solid
            rgba(148,163,184,0.20);
    }

    .hero-title {
        color: #f8fafc;
        font-size: 2.45rem;
        font-weight: 750;
        letter-spacing: -0.8px;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 900px;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    .engine-card {
        padding: 1rem 1.2rem;
        border-radius: 13px;
        border: 1px solid rgba(128,128,128,0.22);
        background: rgba(128,128,128,0.045);
        margin-bottom: 1rem;
    }

    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(128,128,128,0.18);
        text-align: center;
        opacity: 0.6;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL CACHE
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def get_model(engine_name):

    engine_module = OCR_ENGINES[
        engine_name
    ]

    return engine_module.load_model()


# ============================================================
# SAFE DOCUMENT PROCESSING
# ============================================================

def process_single_document(
    filename,
    file_bytes,
    model,
    engine_module,
    apply_preprocessing,
):

    try:

        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        image = (
            image_processing
            .load_image_from_bytes(
                file_bytes
            )
        )

        if image is None:

            return {
                "filename": filename,
                "error": (
                    "Could not decode "
                    "the uploaded image."
                ),
            }

        # ----------------------------------------------------
        # Preprocessing
        #
        # Keep the original image by default.
        # This is important for OCR accuracy.
        # ----------------------------------------------------

        if apply_preprocessing:

            processed = (
                image_processing
                .preprocess_image(
                    image
                )
            )

            ocr_image = (
                image_processing
                .to_three_channel(
                    processed
                )
            )

        else:

            ocr_image = image

        # ----------------------------------------------------
        # OCR TIMER
        # ----------------------------------------------------

        start = time.perf_counter()

        detections = (
            engine_module
            .run_ocr(
                model,
                ocr_image,
            )
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        extracted_text = (
            result_processing
            .build_extracted_text(
                detections
            )
        )

        confidence = (
            result_processing
            .average_confidence(
                detections
            )
        )

        annotated = (
            image_processing
            .draw_boxes(
                image,
                detections,
            )
        )

        return {
            "filename": filename,
            "error": None,
            "original_image": image,
            "annotated_image": annotated,
            "detections": detections,
            "extracted_text": extracted_text,
            "confidence": confidence,
            "elapsed_time": elapsed,
        }

    except Exception as error:

        return {
            "filename": filename,
            "error": (
                f"{type(error).__name__}: "
                f"{error}"
            ),
        }


# ============================================================
# MULTI-DOCUMENT PROCESSING
# ============================================================

def process_all_documents(
    documents,
    model,
    engine_module,
    apply_preprocessing,
    max_workers,
):

    results = []

    # --------------------------------------------------------
    # VERY IMPORTANT:
    #
    # documents already contains bytes.
    #
    # We do NOT access Streamlit UploadedFile objects
    # from worker threads.
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        future_map = {}

        for document in documents:

            future = executor.submit(
                process_single_document,
                document["filename"],
                document["bytes"],
                model,
                engine_module,
                apply_preprocessing,
            )

            future_map[
                future
            ] = document["filename"]

        for future in as_completed(
            future_map
        ):

            filename = future_map[
                future
            ]

            try:

                result = (
                    future.result()
                )

            except Exception as error:

                result = {
                    "filename": filename,
                    "error": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                }

            results.append(
                result
            )

    # Restore upload order.
    order = {
        document["filename"]: index
        for index, document
        in enumerate(documents)
    }

    results.sort(
        key=lambda item:
            order.get(
                item["filename"],
                999999,
            )
    )

    return results


# ============================================================
# RESULT RENDERER
# ============================================================

def render_result(
    result,
    engine_name,
    result_index,
):

    filename = result[
        "filename"
    ]

    if result.get("error"):

        st.error(
            f"{filename}: "
            f"{result['error']}"
        )

        return

    st.markdown(
        f"### 📄 {filename}"
    )

    image_col, text_col = (
        st.columns(
            [1.05, 0.95],
            gap="large",
        )
    )

    with image_col:

        st.caption(
            "Detection map"
        )

        annotated_rgb = (
            cv2.cvtColor(
                result[
                    "annotated_image"
                ],
                cv2.COLOR_BGR2RGB,
            )
        )

        st.image(
            annotated_rgb,
            width="stretch",
        )

    with text_col:

        st.caption(
            "Extracted text"
        )

        st.text_area(
            "OCR Result",
            value=(
                result[
                    "extracted_text"
                ]
                or "No text detected."
            ),
            height=350,
            label_visibility="collapsed",
            key=(
                f"text_"
                f"{engine_name}_"
                f"{result_index}_"
                f"{filename}"
            ),
        )

    metric1, metric2, metric3 = (
        st.columns(3)
    )

    with metric1:

        st.metric(
            "Text Regions",
            len(
                result[
                    "detections"
                ]
            ),
        )

    with metric2:

        st.metric(
            "Average Confidence",
            (
                f"{result['confidence'] * 100:.1f}%"
            ),
        )

    with metric3:

        st.metric(
            "Extraction Time",
            (
                f"{result['elapsed_time']:.2f}s"
            ),
        )

    download_name = (
        filename
        + "_"
        + engine_name.lower()
        + ".txt"
    )

    st.download_button(
        "⬇ Download Extracted Text",
        data=result[
            "extracted_text"
        ],
        file_name=download_name,
        mime="text/plain",
        width="stretch",
        key=(
            f"download_"
            f"{engine_name}_"
            f"{result_index}"
        ),
    )

    with st.expander(
        "View detection details"
    ):

        for detection in result[
            "detections"
        ]:

            st.write(
                f"**{detection['text']}** "
                f"— confidence: "
                f"{float(detection['confidence']):.3f}"
            )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Document OCR Studio</div>
        <div class="hero-subtitle">
            A multi-engine OCR workspace powered by
            EasyOCR, PaddleOCR, RapidOCR, OpenCV, and Streamlit.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## OCR Configuration"
    )

    engine_name = st.selectbox(
        "Recognition Engine",
        [
            "EasyOCR",
            "PaddleOCR",
            "RapidOCR",
        ],
    )

    st.markdown("---")

    st.markdown(
        "### Image Preprocessing"
    )

    apply_preprocessing = st.checkbox(
        "Enable preprocessing",
        value=False,
        help=(
            "CLAHE + adaptive thresholding. "
            "Leave disabled for normal printed documents."
        ),
    )

    st.markdown("---")

    st.markdown(
        "### Engine Information"
    )

    if engine_name == "EasyOCR":

        st.caption(
            "EasyOCR runs on CPU and uses "
            "limited parallel workers."
        )

    elif engine_name == "PaddleOCR":

        st.caption(
            "PaddleOCR uses the lightweight "
            "PP-OCRv5 mobile models and "
            "one inference worker for stability."
        )

    else:

        st.caption(
            "RapidOCR uses ONNX Runtime and "
            "supports parallel document processing."
        )


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">Document Input</div>',
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Upload one or more document images",
    type=[
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "webp",
        "tiff",
    ],
    accept_multiple_files=True,
    label_visibility="collapsed",
)


if not uploaded_files:

    st.info(
        "Upload one or more document images "
        "to begin OCR analysis."
    )

    st.stop()


# ============================================================
# READ FILES ON MAIN STREAMLIT THREAD
#
# This avoids passing UploadedFile objects into
# ThreadPoolExecutor workers.
# ============================================================

documents = []

for uploaded_file in uploaded_files:

    try:

        file_bytes = (
            uploaded_file.getvalue()
        )

        documents.append(
            {
                "filename": uploaded_file.name,
                "bytes": file_bytes,
            }
        )

    except Exception as error:

        st.error(
            f"Could not read "
            f"{uploaded_file.name}: "
            f"{error}"
        )


if not documents:

    st.stop()


st.success(
    f"{len(documents)} document(s) uploaded."
)


# ============================================================
# OCR CONTROLS
# ============================================================

st.markdown(
    '<div class="section-title">OCR Analysis</div>',
    unsafe_allow_html=True,
)

run_button = st.button(
    (
        f"Run {engine_name} on "
        f"All {len(documents)} Document(s)"
    ),
    type="primary",
    width="stretch",
)


# ============================================================
# RUN OCR
# ============================================================

if run_button:

    # --------------------------------------------------------
    # Model loading
    # --------------------------------------------------------

    st.info(
        f"Preparing {engine_name}..."
    )

    model_load_start = (
        time.perf_counter()
    )

    try:

        model = get_model(
            engine_name
        )

        model_load_time = (
            time.perf_counter()
            - model_load_start
        )

    except Exception as error:

        st.error(
            f"{engine_name} could not "
            f"be initialized."
        )

        st.exception(error)

        st.stop()

    st.success(
        f"{engine_name} ready "
        f"in {model_load_time:.2f} seconds."
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    worker_count = (
        MAX_WORKERS_PER_ENGINE[
            engine_name
        ]
    )

    overall_start = (
        time.perf_counter()
    )

    progress = st.progress(
        0,
        text=(
            f"Running {engine_name}..."
        ),
    )

    results = []

    try:

        # We implement progress ourselves so the UI
        # doesn't look frozen during multi-document OCR.

        with ThreadPoolExecutor(
            max_workers=worker_count
        ) as executor:

            future_map = {}

            for document in documents:

                future = executor.submit(
                    process_single_document,
                    document["filename"],
                    document["bytes"],
                    model,
                    OCR_ENGINES[
                        engine_name
                    ],
                    apply_preprocessing,
                )

                future_map[
                    future
                ] = document["filename"]

            completed = 0

            for future in as_completed(
                future_map
            ):

                filename = future_map[
                    future
                ]

                try:

                    result = (
                        future.result()
                    )

                except Exception as error:

                    result = {
                        "filename": filename,
                        "error": (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                    }

                results.append(
                    result
                )

                completed += 1

                progress.progress(
                    int(
                        completed
                        / len(documents)
                        * 100
                    ),
                    text=(
                        f"Processed "
                        f"{completed}/"
                        f"{len(documents)} "
                        f"documents"
                    ),
                )

        overall_elapsed = (
            time.perf_counter()
            - overall_start
        )

        progress.progress(
            100,
            text="OCR complete.",
        )

    except Exception as error:

        progress.empty()

        st.error(
            "OCR processing failed."
        )

        st.exception(error)

        st.stop()

    # --------------------------------------------------------
    # Restore upload order
    # --------------------------------------------------------

    order = {
        document["filename"]: index
        for index, document
        in enumerate(documents)
    }

    results.sort(
        key=lambda result:
            order.get(
                result["filename"],
                999999,
            )
    )

    # --------------------------------------------------------
    # Batch metrics
    # --------------------------------------------------------

    st.markdown("---")

    successful = [
        result
        for result in results
        if not result.get("error")
    ]

    failed = [
        result
        for result in results
        if result.get("error")
    ]

    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )

    with metric1:

        st.metric(
            "Documents",
            len(results),
        )

    with metric2:

        st.metric(
            "Successful",
            len(successful),
        )

    with metric3:

        st.metric(
            "Failed",
            len(failed),
        )

    with metric4:

        st.metric(
            "Total Batch Time",
            f"{overall_elapsed:.2f}s",
        )

    st.caption(
        f"Model initialization: "
        f"{model_load_time:.2f}s · "
        f"Inference workers: "
        f"{worker_count}"
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Recognition Results'
        '</div>',
        unsafe_allow_html=True,
    )

    for index, result in enumerate(
        results
    ):

        render_result(
            result,
            engine_name,
            index,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        Day 24 · Document OCR Studio

        <br>

        EasyOCR · PaddleOCR · RapidOCR
        · OpenCV · Streamlit

        <br><br>

        Developed by Hadeed Jalani

    </div>
    """,
    unsafe_allow_html=True,
)