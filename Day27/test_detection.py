import os
import cv2
from ultralytics import YOLO


MODEL_PATH = "yolo11n.pt"
INPUT_DIR = "sample_images"
OUTPUT_DIR = "outputs/images"

CONFIDENCE = 0.25


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading YOLO model...")

    model = YOLO(MODEL_PATH)

    image_files = [
        file
        for file in os.listdir(INPUT_DIR)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        )
    ]

    if not image_files:
        print(
            f"No images found in: {INPUT_DIR}"
        )
        return

    print(
        f"\nFound {len(image_files)} image(s)."
    )

    print("=" * 60)

    for image_file in image_files:

        input_path = os.path.join(
            INPUT_DIR,
            image_file,
        )

        print(
            f"\nProcessing: {image_file}"
        )

        image = cv2.imread(input_path)

        if image is None:
            print(
                "  ERROR: Could not read image."
            )
            continue

        results = model.predict(
            source=image,
            conf=CONFIDENCE,
            device="cpu",
            verbose=False,
        )

        result = results[0]

        annotated_image = result.plot(
            conf=True,
            labels=True,
            boxes=True,
        )

        output_name = (
            os.path.splitext(image_file)[0]
            + "_detected.jpg"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            output_name,
        )

        cv2.imwrite(
            output_path,
            annotated_image,
        )

        # ---------------------------------------------
        # Detection information
        # ---------------------------------------------

        if result.boxes is None:

            print("  Objects detected: 0")

            continue

        boxes = result.boxes

        print(
            f"  Objects detected: {len(boxes)}"
        )

        for index, box in enumerate(
            boxes,
            start=1,
        ):

            class_id = int(
                box.cls[0]
            )

            class_name = (
                result.names[class_id]
            )

            confidence = float(
                box.conf[0]
            )

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            print(
                f"  {index}. "
                f"{class_name} "
                f"({confidence:.2%}) "
                f"bbox="
                f"({int(x1)}, {int(y1)}, "
                f"{int(x2)}, {int(y2)})"
            )

        print(
            f"  Saved: {output_path}"
        )

    print("\n" + "=" * 60)

    print(
        "\nYOLO image detection testing completed."
    )


if __name__ == "__main__":
    main()