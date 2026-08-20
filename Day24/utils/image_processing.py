import cv2
import numpy as np


def load_image_from_bytes(file_bytes):

    array = np.frombuffer(
        file_bytes,
        dtype=np.uint8,
    )

    return cv2.imdecode(
        array,
        cv2.IMREAD_COLOR,
    )


def preprocess_image(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0,
    )

    thresholded = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return thresholded


def to_three_channel(image):

    if image is None:
        return None

    if len(image.shape) == 2:

        return cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR,
        )

    return image


def draw_boxes(
    image,
    detections,
):

    output = image.copy()

    height, width = output.shape[:2]

    for detection in detections:

        bbox = detection.get(
            "bbox"
        )

        if bbox is None:
            continue

        try:

            x1, y1, x2, y2 = bbox

            x1 = int(
                max(
                    0,
                    min(
                        width - 1,
                        x1,
                    ),
                )
            )

            y1 = int(
                max(
                    0,
                    min(
                        height - 1,
                        y1,
                    ),
                )
            )

            x2 = int(
                max(
                    0,
                    min(
                        width - 1,
                        x2,
                    ),
                )
            )

            y2 = int(
                max(
                    0,
                    min(
                        height - 1,
                        y2,
                    ),
                )
            )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

        except Exception:
            continue

    return output