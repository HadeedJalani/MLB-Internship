# ==========================================================
# MLB Summer Internship - Day 19
# Shape Detection System
# ==========================================================

import cv2
import os
import csv

from contour_detection import detect_contours
from shape_detection import (
    analyze_contours,
    draw_shape_information,
    calculate_shape_statistics,
)


# ==========================================================
# Configuration
# ==========================================================

INPUT_FOLDER = "input_images"

OUTPUT_FOLDER = "output_images"

CONTOUR_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "contours"
)

SHAPE_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "shapes"
)

COMPARISON_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "comparisons"
)

REPORT_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "reports"
)


# ==========================================================
# Create Output Directories
# ==========================================================

for folder in [
    OUTPUT_FOLDER,
    CONTOUR_FOLDER,
    SHAPE_FOLDER,
    COMPARISON_FOLDER,
    REPORT_FOLDER,
]:
    os.makedirs(
        folder,
        exist_ok=True
    )


# ==========================================================
# Configuration Parameters
# ==========================================================

MIN_CONTOUR_AREA = 500


# ==========================================================
# Create Contour Image
# ==========================================================

def create_contour_image(
    image,
    contours
):
    """
    Draw clean contours on the original image.
    """

    result = image.copy()

    for index, contour in enumerate(
        contours,
        start=1
    ):

        cv2.drawContours(
            result,
            [contour],
            -1,
            (0, 255, 0),
            2
        )

        x, y, width, height = cv2.boundingRect(
            contour
        )

        cv2.putText(
            result,
            f"Object {index}",
            (
                x,
                max(y - 10, 25)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    return result


# ==========================================================
# Create Comparison Image
# ==========================================================

def create_comparison(
    original,
    final
):
    """
    Create a side-by-side comparison image.
    """

    target_height = 600

    def resize_image(image):

        height, width = image.shape[:2]

        scale = (
            target_height /
            float(height)
        )

        new_width = int(
            width * scale
        )

        return cv2.resize(
            image,
            (
                new_width,
                target_height
            )
        )

    original_resized = resize_image(
        original
    )

    final_resized = resize_image(
        final
    )

    # ------------------------------------------------------
    # Make both images the same width
    # ------------------------------------------------------

    max_width = max(
        original_resized.shape[1],
        final_resized.shape[1]
    )

    def pad_image(image):

        padding = max_width - image.shape[1]

        if padding <= 0:
            return image

        return cv2.copyMakeBorder(
            image,
            0,
            0,
            0,
            padding,
            cv2.BORDER_CONSTANT,
            value=(30, 30, 30)
        )

    original_resized = pad_image(
        original_resized
    )

    final_resized = pad_image(
        final_resized
    )

    # ------------------------------------------------------
    # Add titles
    # ------------------------------------------------------

    cv2.putText(
        original_resized,
        "Original",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        final_resized,
        "Detected Shapes",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return np_hstack(
        original_resized,
        final_resized
    )


# ==========================================================
# Safe Horizontal Stack
# ==========================================================

def np_hstack(
    image1,
    image2
):
    """
    Safely combine two images horizontally.
    """

    return cv2.hconcat(
        [
            image1,
            image2
        ]
    )


# ==========================================================
# Save CSV Report
# ==========================================================

def save_report(
    filename,
    objects
):

    base_name = os.path.splitext(
        filename
    )[0]

    report_path = os.path.join(
        REPORT_FOLDER,
        f"{base_name}_report.csv"
    )

    fields = [
        "id",
        "shape",
        "area",
        "perimeter",
        "x",
        "y",
        "width",
        "height",
        "circularity",
        "circle_center",
        "circle_radius",
    ]

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fields
        )

        writer.writeheader()

        for obj in objects:

            row = obj.copy()

            row["circle_center"] = str(
                row["circle_center"]
            )

            writer.writerow(
                row
            )

    return report_path


# ==========================================================
# Process One Image
# ==========================================================

def process_image(filename):

    input_path = os.path.join(
        INPUT_FOLDER,
        filename
    )

    image = cv2.imread(
        input_path
    )

    if image is None:

        print(
            f"Unable to read: {filename}"
        )

        return

    print(
        f"\nProcessing: {filename}"
    )

    # ------------------------------------------------------
    # Detect contours
    # ------------------------------------------------------

    contours, gray, threshold = detect_contours(
        image,
        min_area=MIN_CONTOUR_AREA
    )

    # ------------------------------------------------------
    # Analyze shapes
    # ------------------------------------------------------

    objects = analyze_contours(
        contours
    )

    # ------------------------------------------------------
    # Create contour result
    # ------------------------------------------------------

    contour_image = create_contour_image(
        image,
        contours
    )

    # ------------------------------------------------------
    # Create shape result
    # ------------------------------------------------------

    shape_image = draw_shape_information(
        image,
        contours,
        objects
    )

    # ------------------------------------------------------
    # Save filenames
    # ------------------------------------------------------

    base_name = os.path.splitext(
        filename
    )[0]

    contour_path = os.path.join(
        CONTOUR_FOLDER,
        f"{base_name}_contours.jpg"
    )

    shape_path = os.path.join(
        SHAPE_FOLDER,
        f"{base_name}_shapes.jpg"
    )

    comparison_path = os.path.join(
        COMPARISON_FOLDER,
        f"{base_name}_comparison.jpg"
    )

    # ------------------------------------------------------
    # Save contour image
    # ------------------------------------------------------

    cv2.imwrite(
        contour_path,
        contour_image
    )

    # ------------------------------------------------------
    # Save shape image
    # ------------------------------------------------------

    cv2.imwrite(
        shape_path,
        shape_image
    )

    # ------------------------------------------------------
    # Save comparison
    # ------------------------------------------------------

    comparison = create_comparison(
        image,
        shape_image
    )

    cv2.imwrite(
        comparison_path,
        comparison
    )

    # ------------------------------------------------------
    # Save report
    # ------------------------------------------------------

    report_path = save_report(
        filename,
        objects
    )

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    statistics = calculate_shape_statistics(
        objects
    )

    print(
        f"Objects detected: "
        f"{statistics['total']}"
    )

    print(
        f"Triangles: "
        f"{statistics['triangles']}"
    )

    print(
        f"Squares: "
        f"{statistics['squares']}"
    )

    print(
        f"Rectangles: "
        f"{statistics['rectangles']}"
    )

    print(
        f"Circles: "
        f"{statistics['circles']}"
    )

    print(
        f"Polygons: "
        f"{statistics['polygons']}"
    )

    print(
        f"Contour output: "
        f"{contour_path}"
    )

    print(
        f"Shape output: "
        f"{shape_path}"
    )

    print(
        f"Comparison: "
        f"{comparison_path}"
    )

    print(
        f"Report: "
        f"{report_path}"
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)

    print(
        "Day 19 - Shape Detection System"
    )

    print("=" * 70)

    if not os.path.exists(
        INPUT_FOLDER
    ):

        raise FileNotFoundError(
            "input_images folder was not found."
        )

    image_files = [
        file
        for file in os.listdir(
            INPUT_FOLDER
        )
        if file.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp"
            )
        )
    ]

    if not image_files:

        raise FileNotFoundError(
            "No images found inside input_images."
        )

    print(
        f"\nImages found: "
        f"{len(image_files)}"
    )

    for filename in sorted(
        image_files
    ):

        process_image(
            filename
        )

    print("\n" + "=" * 70)

    print(
        "Shape detection completed successfully."
    )

    print("=" * 70)

    print(
        f"\nContour outputs: "
        f"{CONTOUR_FOLDER}"
    )

    print(
        f"Shape outputs: "
        f"{SHAPE_FOLDER}"
    )

    print(
        f"Comparisons: "
        f"{COMPARISON_FOLDER}"
    )

    print(
        f"Reports: "
        f"{REPORT_FOLDER}"
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()