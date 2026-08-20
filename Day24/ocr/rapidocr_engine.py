from rapidocr_onnxruntime import RapidOCR


def load_model():
    return RapidOCR()


def run_ocr(model, image):

    result, _ = model(image)

    detections = []

    if not result:
        return detections

    for item in result:

        if not item or len(item) < 3:
            continue

        bbox = item[0]
        text = str(item[1]).strip()

        try:
            confidence = float(item[2])
        except Exception:
            confidence = 0.0

        if not text:
            continue

        try:

            xs = [
                float(point[0])
                for point in bbox
            ]

            ys = [
                float(point[1])
                for point in bbox
            ]

            x1 = min(xs)
            y1 = min(ys)
            x2 = max(xs)
            y2 = max(ys)

        except Exception:
            continue

        detections.append(
            {
                "text": text,
                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2,
                ),
                "confidence": confidence,
            }
        )

    detections.sort(
        key=lambda item: (
            item["bbox"][1],
            item["bbox"][0],
        )
    )

    return detections