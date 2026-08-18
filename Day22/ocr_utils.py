# ==========================================================
# Day 22 - OCR Utilities
# Enhanced OCR Processing Pipeline
# ==========================================================

import cv2
import numpy as np


# ==========================================================
# Basic Image Conversion
# ==========================================================

def ensure_bgr(image):
    """
    Ensure the image is a 3-channel BGR image.
    """

    if image is None:
        raise ValueError("Input image is None.")

    if len(image.shape) == 2:
        return cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    if image.shape[2] == 4:
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2BGR
        )

    return image


def bgr_to_rgb(image):
    """
    Safely convert BGR or grayscale image to RGB.
    """

    if image is None:
        return None

    if len(image.shape) == 2:

        return cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

    if image.shape[2] == 4:

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2RGB
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ==========================================================
# Grayscale
# ==========================================================

def safe_grayscale(image):
    """
    Convert an image to grayscale safely.

    Prevents the common OpenCV error where cvtColor()
    receives an image that is already single-channel.
    """

    if image is None:
        raise ValueError("Input image is None.")

    if len(image.shape) == 2:
        return image

    channels = image.shape[2]

    if channels == 1:
        return image[:, :, 0]

    if channels == 4:

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2GRAY
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


# ==========================================================
# Image Upscaling
# ==========================================================

def upscale_image(
    image,
    scale=2.0
):
    """
    Upscale image using high-quality Lanczos interpolation.

    Upscaling helps OCR when characters are small.
    """

    if image is None:
        return None

    height, width = image.shape[:2]

    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LANCZOS4
    )


# ==========================================================
# Contrast Enhancement
# ==========================================================

def enhance_contrast(gray):
    """
    Improve local contrast using CLAHE.

    CLAHE is generally safer for OCR than aggressive
    global contrast stretching.
    """

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(gray)


# ==========================================================
# Noise Reduction
# ==========================================================

def reduce_noise(gray):
    """
    Mild noise reduction.

    Keeps character edges while removing small artifacts.
    """

    return cv2.fastNlMeansDenoising(
        gray,
        None,
        h=7,
        templateWindowSize=7,
        searchWindowSize=21
    )


# ==========================================================
# Sharpening
# ==========================================================

def sharpen_image(gray):
    """
    Mild unsharp masking.

    Helps make handwritten strokes more distinct.
    """

    blurred = cv2.GaussianBlur(
        gray,
        (0, 0),
        1.2
    )

    sharpened = cv2.addWeighted(
        gray,
        1.6,
        blurred,
        -0.6,
        0
    )

    return sharpened


# ==========================================================
# Adaptive Threshold
# ==========================================================

def adaptive_threshold(gray):
    """
    Create a clean binary image.

    This is useful for high-contrast documents,
    but is NOT always ideal for handwriting.
    """

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )


# ==========================================================
# Deskew
# ==========================================================

def deskew_image(gray):
    """
    Attempt to correct small document rotation.

    Useful when the entire page is slightly tilted.
    """

    try:

        inverted = cv2.bitwise_not(gray)

        coords = np.column_stack(
            np.where(inverted > 0)
        )

        if len(coords) < 50:
            return gray

        angle = cv2.minAreaRect(
            coords
        )[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Avoid aggressive rotation
        if abs(angle) > 10:
            return gray

        height, width = gray.shape[:2]

        center = (
            width // 2,
            height // 2
        )

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        return cv2.warpAffine(
            gray,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    except Exception:
        return gray


# ==========================================================
# Main Preprocessing
# ==========================================================

def preprocess_image(
    image,
    enhancement=True
):
    """
    Main preprocessing function used by the Streamlit app.

    Pipeline:

        Original
           ↓
        Grayscale
           ↓
        Deskew
           ↓
        Upscale
           ↓
        CLAHE
           ↓
        Denoise
           ↓
        Sharpen

    The function intentionally avoids aggressive
    thresholding because handwriting can lose important
    stroke information after binarization.
    """

    image = ensure_bgr(image)

    gray = safe_grayscale(image)

    gray = deskew_image(gray)

    gray = upscale_image(
        gray,
        scale=2.0
    )

    if enhancement:

        gray = enhance_contrast(
            gray
        )

        gray = reduce_noise(
            gray
        )

        gray = sharpen_image(
            gray
        )

    return gray


# ==========================================================
# OCR Preprocessing Variants
# ==========================================================

def create_ocr_variants(
    image
):
    """
    Create several OCR-friendly versions of the image.

    Instead of assuming one preprocessing technique is
    always best, the OCR engine can compare multiple
    versions.
    """

    image = ensure_bgr(image)

    gray = safe_grayscale(
        image
    )

    gray = deskew_image(
        gray
    )

    # Variant 1
    # Minimal processing.
    original = upscale_image(
        gray,
        scale=2.0
    )

    # Variant 2
    # Contrast + mild sharpening.
    enhanced = upscale_image(
        gray,
        scale=2.0
    )

    enhanced = enhance_contrast(
        enhanced
    )

    enhanced = sharpen_image(
        enhanced
    )

    # Variant 3
    # Denoised + enhanced.
    clean = upscale_image(
        gray,
        scale=2.0
    )

    clean = enhance_contrast(
        clean
    )

    clean = reduce_noise(
        clean
    )

    # Variant 4
    # Adaptive threshold.
    threshold = upscale_image(
        gray,
        scale=2.0
    )

    threshold = enhance_contrast(
        threshold
    )

    threshold = adaptive_threshold(
        threshold
    )

    return [
        ("Original Enhanced", original),
        ("Contrast Enhanced", enhanced),
        ("Denoised Enhanced", clean),
        ("Adaptive Threshold", threshold),
    ]


# ==========================================================
# OCR Result Normalization
# ==========================================================

def normalize_text(text):
    """
    Clean common OCR formatting artifacts.
    """

    if text is None:
        return ""

    text = str(text)

    # Normalize unusual whitespace.
    text = text.replace(
        "\t",
        " "
    )

    text = " ".join(
        text.split()
    )

    return text.strip()


# ==========================================================
# Bounding Box Helpers
# ==========================================================

def get_box_geometry(
    box
):
    """
    Return useful geometry information from an EasyOCR box.
    """

    points = np.array(
        box,
        dtype=np.float32
    )

    x_min = float(
        np.min(points[:, 0])
    )

    x_max = float(
        np.max(points[:, 0])
    )

    y_min = float(
        np.min(points[:, 1])
    )

    y_max = float(
        np.max(points[:, 1])
    )

    center_x = (
        x_min + x_max
    ) / 2

    center_y = (
        y_min + y_max
    ) / 2

    height = (
        y_max - y_min
    )

    width = (
        x_max - x_min
    )

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "center_x": center_x,
        "center_y": center_y,
        "width": width,
        "height": height,
    }


# ==========================================================
# Sort OCR Results Into Reading Order
# ==========================================================

def sort_detections(
    detections
):
    """
    Sort EasyOCR detections into natural reading order.

    EasyOCR may return text regions in an order that does
    not match how a human reads the page.

    This function groups nearby words into lines and sorts
    lines from top to bottom.
    """

    if not detections:
        return []

    items = []

    for detection in detections:

        box = detection[0]
        text = normalize_text(
            detection[1]
        )
        confidence = float(
            detection[2]
        )

        if not text:
            continue

        geometry = get_box_geometry(
            box
        )

        items.append(
            {
                "box": box,
                "text": text,
                "confidence": confidence,
                **geometry,
            }
        )

    if not items:
        return []

    # Estimate typical text height.
    heights = [
        item["height"]
        for item in items
    ]

    median_height = float(
        np.median(heights)
    )

    line_threshold = max(
        median_height * 0.65,
        12
    )

    # Sort approximately top-to-bottom.
    items.sort(
        key=lambda item: (
            item["center_y"],
            item["x_min"]
        )
    )

    lines = []

    for item in items:

        assigned = False

        for line in lines:

            line_y = np.mean(
                [
                    word["center_y"]
                    for word in line
                ]
            )

            if abs(
                item["center_y"] - line_y
            ) <= line_threshold:

                line.append(
                    item
                )

                assigned = True

                break

        if not assigned:

            lines.append(
                [item]
            )

    # Sort words left-to-right.
    for line in lines:

        line.sort(
            key=lambda item: item["x_min"]
        )

    # Sort lines top-to-bottom.
    lines.sort(
        key=lambda line: np.mean(
            [
                word["center_y"]
                for word in line
            ]
        )
    )

    ordered = []

    for line in lines:

        ordered.extend(
            line
        )

    return ordered


# ==========================================================
# Reconstruct Text
# ==========================================================

def detections_to_text(
    detections
):
    """
    Convert ordered OCR detections into readable lines.
    """

    ordered = sort_detections(
        detections
    )

    if not ordered:
        return ""

    heights = [
        item["height"]
        for item in ordered
    ]

    median_height = float(
        np.median(heights)
    )

    line_threshold = max(
        median_height * 0.7,
        12
    )

    lines = []

    current_line = []

    current_y = None

    for item in ordered:

        if current_y is None:

            current_line = [
                item
            ]

            current_y = item[
                "center_y"
            ]

            continue

        if abs(
            item["center_y"] - current_y
        ) <= line_threshold:

            current_line.append(
                item
            )

            current_y = np.mean(
                [
                    word["center_y"]
                    for word in current_line
                ]
            )

        else:

            current_line.sort(
                key=lambda word:
                word["x_min"]
            )

            lines.append(
                current_line
            )

            current_line = [
                item
            ]

            current_y = item[
                "center_y"
            ]

    if current_line:

        current_line.sort(
            key=lambda word:
            word["x_min"]
        )

        lines.append(
            current_line
        )

    output_lines = []

    for line in lines:

        words = [
            word["text"]
            for word in line
        ]

        output_lines.append(
            " ".join(words)
        )

    return "\n".join(
        output_lines
    ).strip()


# ==========================================================
# OCR Quality Score
# ==========================================================

def score_ocr_result(
    detections
):
    """
    Calculate a quality score for a candidate OCR result.

    Confidence alone is not enough, because OCR can sometimes
    assign high confidence to incorrect words.

    The score considers:

    - Average confidence
    - Number of detected regions
    - Amount of extracted text
    """

    if not detections:

        return 0.0

    valid = [
        detection
        for detection in detections
        if normalize_text(
            detection[1]
        )
    ]

    if not valid:

        return 0.0

    confidences = [
        float(
            detection[2]
        )
        for detection in valid
    ]

    average_confidence = float(
        np.mean(confidences)
    )

    text = detections_to_text(
        valid
    )

    character_count = len(
        text
    )

    # Reasonable reward for actual extracted text.
    text_score = min(
        character_count / 100.0,
        1.0
    )

    # Detection score.
    detection_score = min(
        len(valid) / 15.0,
        1.0
    )

    score = (
        average_confidence * 0.65
        + text_score * 0.20
        + detection_score * 0.15
    )

    return float(score)


# ==========================================================
# OCR Extraction
# ==========================================================

def extract_text(
    reader,
    image,
    preprocess=True
):
    """
    Perform robust OCR using multiple image variants.

    EasyOCR is run on several versions of the image and
    the strongest result is selected automatically.
    """

    if image is None:

        raise ValueError(
            "Input image is empty."
        )

    if preprocess:

        variants = create_ocr_variants(
            image
        )

    else:

        image_bgr = ensure_bgr(
            image
        )

        variants = [
            (
                "Original",
                upscale_image(
                    image_bgr,
                    scale=2.0
                )
            )
        ]

    best_detections = []
    best_score = -1.0

    for variant_name, variant in variants:

        try:

            detections = reader.readtext(
                variant,
                detail=1,
                paragraph=False,
                decoder="beamsearch",
                beamWidth=5,
                batch_size=1,
                text_threshold=0.45,
                low_text=0.25,
                link_threshold=0.25,
                canvas_size=2560,
                mag_ratio=1.0,
                width_ths=0.65,
                height_ths=0.55,
                ycenter_ths=0.55,
                slope_ths=0.15,
            )

            score = score_ocr_result(
                detections
            )

            if score > best_score:

                best_score = score

                best_detections = detections

        except Exception:
            continue

    if not best_detections:

        return []

    # Normalize and sort.
    ordered = sort_detections(
        best_detections
    )

    return [
        (
            item["box"],
            item["text"],
            item["confidence"]
        )
        for item in ordered
    ]


# ==========================================================
# Extract Text
# ==========================================================

def get_text(
    detections
):
    """
    Return clean reconstructed OCR text.
    """

    if not detections:
        return ""

    return detections_to_text(
        detections
    )


# ==========================================================
# Confidence
# ==========================================================

def calculate_average_confidence(
    detections
):
    """
    Calculate average OCR confidence.
    """

    if not detections:
        return 0.0

    values = [
        float(
            detection[2]
        )
        for detection in detections
    ]

    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


# ==========================================================
# Draw Detection Boxes
# ==========================================================

def draw_detections(
    image,
    detections
):
    """
    Draw OCR bounding boxes and confidence values.
    """

    image = ensure_bgr(
        image
    ).copy()

    if not detections:
        return image

    for detection in detections:

        box = detection[0]
        text = detection[1]
        confidence = float(
            detection[2]
        )

        points = np.array(
            box,
            dtype=np.int32
        )

        cv2.polylines(
            image,
            [points],
            True,
            (40, 180, 80),
            2
        )

        x = int(
            np.min(points[:, 0])
        )

        y = int(
            np.min(points[:, 1])
        )

        label = (
            f"{confidence * 100:.0f}%"
        )

        cv2.putText(
            image,
            label,
            (x, max(y - 6, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 180, 80),
            1,
            cv2.LINE_AA
        )

    return image