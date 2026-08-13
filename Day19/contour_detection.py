# ==========================================================
# MLB Summer Internship - Day 19
# Robust Contour Detection & Shape Classification
# ==========================================================

import cv2
import numpy as np


# ==========================================================
# Preprocessing
# ==========================================================

def preprocess_image(image):
    """
    Prepare image for contour detection.

    Uses grayscale + blur + adaptive thresholding.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # OTSU threshold
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Small morphological cleanup
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    return gray, binary


# ==========================================================
# Contour Detection
# ==========================================================

def detect_contours(image, min_area_ratio=0.01):
    """
    Detect only meaningful external contours.

    Important:
    RETR_EXTERNAL prevents inner borders from being
    detected as separate objects.
    """

    gray, binary = preprocess_image(image)

    image_height, image_width = gray.shape

    image_area = image_height * image_width

    min_area = image_area * min_area_ratio

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore tiny objects
        if area < min_area:
            continue

        x, y, width, height = cv2.boundingRect(
            contour
        )

        # Ignore extremely small bounding boxes
        if width < 20 or height < 20:
            continue

        # Ignore extremely thin objects
        if width / float(height) > 10:
            continue

        if height / float(width) > 10:
            continue

        valid_contours.append(contour)

    # Largest objects first
    valid_contours = sorted(
        valid_contours,
        key=cv2.contourArea,
        reverse=True
    )

    return valid_contours, gray, binary


# ==========================================================
# Geometry Helpers
# ==========================================================

def calculate_circularity(contour):

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(
        contour,
        True
    )

    if perimeter == 0:
        return 0.0

    return float(
        (4 * np.pi * area) /
        (perimeter * perimeter)
    )


def calculate_solidity(contour):

    area = cv2.contourArea(contour)

    hull = cv2.convexHull(contour)

    hull_area = cv2.contourArea(hull)

    if hull_area == 0:
        return 0.0

    return float(
        area / hull_area
    )


def calculate_extent(contour):

    area = cv2.contourArea(contour)

    x, y, width, height = cv2.boundingRect(
        contour
    )

    rectangle_area = width * height

    if rectangle_area == 0:
        return 0.0

    return float(
        area / rectangle_area
    )


# ==========================================================
# Shape Classification
# ==========================================================

def classify_shape(contour):

    perimeter = cv2.arcLength(
        contour,
        True
    )

    area = cv2.contourArea(
        contour
    )

    if perimeter <= 0 or area <= 0:
        return "Unknown", 0.0

    approximation = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    vertices = len(
        approximation
    )

    x, y, width, height = cv2.boundingRect(
        approximation
    )

    if height == 0:
        return "Unknown", 0.0

    aspect_ratio = width / float(height)

    circularity = calculate_circularity(
        contour
    )

    solidity = calculate_solidity(
        contour
    )

    extent = calculate_extent(
        contour
    )

    # ------------------------------------------------------
    # Triangle
    # ------------------------------------------------------

    if vertices == 3:

        if solidity > 0.85:

            return "Triangle", 0.95

        return "Triangle", 0.80

    # ------------------------------------------------------
    # Four-sided shapes
    # ------------------------------------------------------

    if vertices == 4:

        # Nearly square
        if 0.90 <= aspect_ratio <= 1.10:

            return "Square", 0.95

        # Rectangle
        return "Rectangle", 0.95

    # ------------------------------------------------------
    # Circle
    # ------------------------------------------------------

    # Circularity is the most important feature here.
    #
    # Avoid classifying every rounded polygon as a circle.

    if circularity >= 0.88 and solidity >= 0.95:

        return "Circle", min(
            0.99,
            circularity
        )

    # ------------------------------------------------------
    # Rounded shapes / polygons
    # ------------------------------------------------------

    if vertices == 5:

        return "Pentagon", 0.92

    if vertices == 6:

        return "Hexagon", 0.92

    if vertices == 7:

        return "Heptagon", 0.90

    if vertices == 8:

        return "Octagon", 0.90

    if vertices == 9:

        return "Nonagon", 0.88

    # ------------------------------------------------------
    # More complex polygon
    # ------------------------------------------------------

    if vertices > 9:

        if circularity >= 0.75:

            return "Rounded Polygon", 0.75

        return "Polygon", 0.80

    return "Unknown", 0.30


# ==========================================================
# Analyze One Contour
# ==========================================================

def analyze_contour(
    contour,
    object_id
):

    area = cv2.contourArea(
        contour
    )

    perimeter = cv2.arcLength(
        contour,
        True
    )

    x, y, width, height = cv2.boundingRect(
        contour
    )

    shape, confidence = classify_shape(
        contour
    )

    circularity = calculate_circularity(
        contour
    )

    solidity = calculate_solidity(
        contour
    )

    extent = calculate_extent(
        contour
    )

    approximation = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    vertices = len(
        approximation
    )

    (
        center_x,
        center_y
    ), radius = cv2.minEnclosingCircle(
        contour
    )

    return {

        "id": object_id,

        "shape": shape,

        "confidence": round(
            confidence,
            3
        ),

        "area": round(
            float(area),
            2
        ),

        "perimeter": round(
            float(perimeter),
            2
        ),

        "x": int(x),

        "y": int(y),

        "width": int(width),

        "height": int(height),

        "circularity": round(
            circularity,
            3
        ),

        "solidity": round(
            solidity,
            3
        ),

        "extent": round(
            extent,
            3
        ),

        "vertices": vertices,

        "circle_center": (
            int(center_x),
            int(center_y)
        ),

        "circle_radius": round(
            float(radius),
            2
        )
    }


# ==========================================================
# Analyze All Contours
# ==========================================================

def analyze_contours(
    contours,
    min_confidence=0.70
):

    objects = []

    for contour in contours:

        object_id = len(objects) + 1

        data = analyze_contour(
            contour,
            object_id
        )

        # Ignore extremely uncertain detections
        if data["confidence"] < min_confidence:

            data["shape"] = "Unknown"

        objects.append(data)

    return objects


# ==========================================================
# Draw Detection Results
# ==========================================================

def draw_shape_information(
    image,
    contours,
    objects
):

    result = image.copy()

    for contour, data in zip(
        contours,
        objects
    ):

        x = data["x"]
        y = data["y"]

        width = data["width"]
        height = data["height"]

        shape = data["shape"]

        object_id = data["id"]

        confidence = data["confidence"]

        # --------------------------------------------------
        # Draw contour
        # --------------------------------------------------

        cv2.drawContours(
            result,
            [contour],
            -1,
            (0, 255, 0),
            2
        )

        # --------------------------------------------------
        # Bounding box
        # --------------------------------------------------

        cv2.rectangle(
            result,
            (x, y),
            (
                x + width,
                y + height
            ),
            (255, 0, 0),
            2
        )

        # --------------------------------------------------
        # Label
        # --------------------------------------------------

        label = (
            f"#{object_id} "
            f"{shape} "
            f"{confidence * 100:.0f}%"
        )

        label_y = max(
            y - 10,
            25
        )

        cv2.putText(
            result,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        # --------------------------------------------------
        # Measurements
        # --------------------------------------------------

        measurement = (
            f"A:{data['area']:.0f} "
            f"P:{data['perimeter']:.0f}"
        )

        measurement_y = min(
            y + height + 20,
            result.shape[0] - 10
        )

        cv2.putText(
            result,
            measurement,
            (x, measurement_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return result


# ==========================================================
# Shape Statistics
# ==========================================================

def calculate_shape_statistics(
    objects
):

    statistics = {

        "total": len(objects),

        "triangles": 0,

        "squares": 0,

        "rectangles": 0,

        "circles": 0,

        "polygons": 0,

        "unknown": 0,

    }

    for obj in objects:

        shape = obj["shape"].lower()

        if shape == "triangle":

            statistics["triangles"] += 1

        elif shape == "square":

            statistics["squares"] += 1

        elif shape == "rectangle":

            statistics["rectangles"] += 1

        elif shape == "circle":

            statistics["circles"] += 1

        elif (
            "polygon" in shape
            or shape in [
                "pentagon",
                "hexagon",
                "heptagon",
                "octagon",
                "nonagon",
                "rounded polygon"
            ]
        ):

            statistics["polygons"] += 1

        else:

            statistics["unknown"] += 1

    return statistics