# ==========================================================
# MLB Summer Internship - Day 19
# Shape Detection and Contour Analysis
# ==========================================================

import cv2
import numpy as np


# ==========================================================
# Shape Classification
# ==========================================================

def classify_shape(contour):
    """
    Classify a contour into a basic geometric shape.

    Supported shapes:
        Triangle
        Square
        Rectangle
        Circle
        Polygon
    """

    perimeter = cv2.arcLength(
        contour,
        True
    )

    if perimeter == 0:
        return "Unknown"

    approximation = cv2.approxPolyDP(
        contour,
        0.04 * perimeter,
        True
    )

    vertices = len(approximation)

    area = cv2.contourArea(contour)

    if area <= 0:
        return "Unknown"

    # ------------------------------------------------------
    # Triangle
    # ------------------------------------------------------

    if vertices == 3:
        return "Triangle"

    # ------------------------------------------------------
    # Four-sided shapes
    # ------------------------------------------------------

    if vertices == 4:

        x, y, width, height = cv2.boundingRect(
            approximation
        )

        if height == 0:
            return "Rectangle"

        aspect_ratio = width / float(height)

        if 0.90 <= aspect_ratio <= 1.10:
            return "Square"

        return "Rectangle"

    # ------------------------------------------------------
    # Circle detection using circularity
    # ------------------------------------------------------

    circularity = (
        4 * np.pi * area
    ) / (
        perimeter * perimeter
    )

    if circularity >= 0.80:
        return "Circle"

    # ------------------------------------------------------
    # Other polygons
    # ------------------------------------------------------

    if vertices > 4:
        return "Polygon"

    return "Unknown"


# ==========================================================
# Circularity
# ==========================================================

def calculate_circularity(contour):
    """
    Calculate contour circularity.

    A perfect circle approaches 1.0.
    Irregular shapes have lower values.
    """

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(
        contour,
        True
    )

    if perimeter == 0:
        return 0.0

    circularity = (
        4 * np.pi * area
    ) / (
        perimeter * perimeter
    )

    return float(circularity)


# ==========================================================
# Contour Analysis
# ==========================================================

def analyze_contour(contour, object_id):
    """
    Extract useful measurements from a contour.
    """

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(
        contour,
        True
    )

    x, y, width, height = cv2.boundingRect(
        contour
    )

    shape = classify_shape(
        contour
    )

    circularity = calculate_circularity(
        contour
    )

    # ------------------------------------------------------
    # Minimum enclosing circle
    # ------------------------------------------------------

    (center_x, center_y), radius = cv2.minEnclosingCircle(
        contour
    )

    center = (
        int(center_x),
        int(center_y)
    )

    radius = float(radius)

    return {
        "id": object_id,
        "shape": shape,
        "area": round(float(area), 2),
        "perimeter": round(float(perimeter), 2),
        "x": int(x),
        "y": int(y),
        "width": int(width),
        "height": int(height),
        "circularity": round(circularity, 3),
        "circle_center": center,
        "circle_radius": round(radius, 2),
    }


# ==========================================================
# Analyze All Contours
# ==========================================================

def analyze_contours(contours):
    """
    Analyze all detected contours and return
    a list of object information dictionaries.
    """

    objects = []

    for object_id, contour in enumerate(
        contours,
        start=1
    ):

        data = analyze_contour(
            contour,
            object_id
        )

        objects.append(data)

    return objects


# ==========================================================
# Draw Shape Information
# ==========================================================

def draw_shape_information(
    image,
    contours,
    objects
):
    """
    Draw contours, bounding boxes, labels,
    and minimum enclosing circles.
    """

    result = image.copy()

    for contour, data in zip(
        contours,
        objects
    ):

        object_id = data["id"]
        shape = data["shape"]

        x = data["x"]
        y = data["y"]
        width = data["width"]
        height = data["height"]

        center = data["circle_center"]
        radius = int(
            data["circle_radius"]
        )

        # --------------------------------------------------
        # Contour
        # --------------------------------------------------

        cv2.drawContours(
            result,
            [contour],
            -1,
            (0, 255, 0),
            2
        )

        # --------------------------------------------------
        # Bounding rectangle
        # --------------------------------------------------

        cv2.rectangle(
            result,
            (x, y),
            (x + width, y + height),
            (255, 0, 0),
            2
        )

        # --------------------------------------------------
        # Minimum enclosing circle
        # --------------------------------------------------

        if radius > 2:

            cv2.circle(
                result,
                center,
                radius,
                (255, 255, 0),
                1
            )

        # --------------------------------------------------
        # Label
        # --------------------------------------------------

        label = (
            f"#{object_id} {shape}"
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
            0.65,
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
            y + height + 22,
            result.shape[0] - 10
        )

        cv2.putText(
            result,
            measurement,
            (x, measurement_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return result


# ==========================================================
# Shape Statistics
# ==========================================================

def calculate_shape_statistics(objects):
    """
    Generate summary statistics for detected shapes.
    """

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

        elif shape == "polygon":
            statistics["polygons"] += 1

        else:
            statistics["unknown"] += 1

    return statistics