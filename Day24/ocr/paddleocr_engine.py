import os
import json

# Force PaddleX to use Hugging Face instead of the failing
# AIStudio model source.
os.environ["PADDLE_PDX_MODEL_SOURCE"] = "huggingface"

# CPU / oneDNN stability settings.
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"

from paddleocr import PaddleOCR


def load_model():

    return PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="en_PP-OCRv5_mobile_rec",

        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,

        enable_mkldnn=False,

        lang="en",
    )


def run_ocr(model, image):

    result = model.predict(image)

    detections = []

    if result is None:
        return detections

    try:
        results = list(result)
    except Exception:
        results = [result]

    for result_object in results:

        data = _extract_result_data(
            result_object
        )

        if not data:
            continue

        # PaddleOCR 3.x stores the actual OCR information
        # inside the "res" dictionary.
        res = data.get("res", data)

        if not isinstance(res, dict):
            continue

        texts = res.get(
            "rec_texts",
            []
        )

        scores = res.get(
            "rec_scores",
            []
        )

        # Prefer rec_polys because they correspond directly
        # to the recognized text entries.
        polygons = res.get(
            "rec_polys",
            []
        )

        # Fallback to rectangular boxes if necessary.
        boxes = res.get(
            "rec_boxes",
            []
        )

        if texts is None:
            texts = []

        if scores is None:
            scores = []

        if polygons is None:
            polygons = []

        if boxes is None:
            boxes = []

        for index, text in enumerate(texts):

            text = str(text).strip()

            if not text:
                continue

            try:
                confidence = float(
                    scores[index]
                )
            except Exception:
                confidence = 0.0

            bbox = None

            # ------------------------------------------------
            # First choice: polygon
            # ------------------------------------------------

            if index < len(polygons):

                try:

                    polygon = polygons[index]

                    xs = [
                        float(point[0])
                        for point in polygon
                    ]

                    ys = [
                        float(point[1])
                        for point in polygon
                    ]

                    bbox = (
                        min(xs),
                        min(ys),
                        max(xs),
                        max(ys),
                    )

                except Exception:
                    bbox = None

            # ------------------------------------------------
            # Second choice: rectangular box
            # ------------------------------------------------

            if bbox is None and index < len(boxes):

                try:

                    box = boxes[index]

                    if hasattr(
                        box,
                        "tolist"
                    ):
                        box = box.tolist()

                    x1, y1, x2, y2 = box

                    bbox = (
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    )

                except Exception:
                    bbox = None

            if bbox is None:
                continue

            detections.append(
                {
                    "text": text,
                    "bbox": bbox,
                    "confidence": confidence,
                }
            )

    # --------------------------------------------------------
    # Reading order
    # --------------------------------------------------------

    detections.sort(
        key=lambda item: (
            item["bbox"][1],
            item["bbox"][0],
        )
    )

    return detections


def _extract_result_data(
    result_object
):

    """
    Extract the dictionary from PaddleOCR 3.x OCRResult.

    PaddleOCR can expose the result through:
        result.json
        result.res
        dictionary-like objects

    The actual OCR data is normally under:
        result["res"]
    """

    # --------------------------------------------------------
    # 1. OCRResult.json
    # --------------------------------------------------------

    try:

        data = result_object.json

        if callable(data):
            data = data()

        # Some versions return a JSON string.
        if isinstance(data, str):

            try:
                data = json.loads(data)
            except Exception:
                return None

        if isinstance(data, dict):
            return data

    except Exception:
        pass

    # --------------------------------------------------------
    # 2. OCRResult.res
    # --------------------------------------------------------

    try:

        data = result_object.res

        if isinstance(data, dict):

            return {
                "res": data
            }

    except Exception:
        pass

    # --------------------------------------------------------
    # 3. Dictionary result
    # --------------------------------------------------------

    if isinstance(
        result_object,
        dict
    ):

        return result_object

    # --------------------------------------------------------
    # Nothing usable
    # --------------------------------------------------------

    return None