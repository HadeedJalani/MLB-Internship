import cv2
import os


INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images/enhancement"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def enhance_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print(f"Unable to read: {image_path}")
        return

    filename = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    # Grayscale
    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            f"{filename}_grayscale.jpg"
        ),
        grayscale
    )

    # Gaussian Blur
    gaussian = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            f"{filename}_gaussian.jpg"
        ),
        gaussian
    )

    # Median Blur
    median = cv2.medianBlur(
        image,
        5
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            f"{filename}_median.jpg"
        ),
        median
    )

    # Bilateral Filter
    bilateral = cv2.bilateralFilter(
        image,
        9,
        75,
        75
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            f"{filename}_bilateral.jpg"
        ),
        bilateral
    )

    # Brightness
    brightness = cv2.convertScaleAbs(
        image,
        alpha=1.0,
        beta=40
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            f"{filename}_brightness.jpg"
        ),
        brightness
    )

    # Contrast
    contrast = cv2.convertScaleAbs(
        image,
        alpha=1.4,
        beta=0
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            f"{filename}_contrast.jpg"
        ),
        contrast
    )

    # Sharpening
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        3
    )

    sharpened = cv2.addWeighted(
        image,
        1.5,
        blurred,
        -0.5,
        0
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            f"{filename}_sharpened.jpg"
        ),
        sharpened
    )


def main():

    print("=" * 60)
    print("Day 17 - Image Enhancement")
    print("=" * 60)

    image_files = [
        file for file in os.listdir(INPUT_FOLDER)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    if not image_files:
        print("No input images found.")
        return

    for image_file in image_files:

        print(f"Enhancing: {image_file}")

        enhance_image(
            os.path.join(
                INPUT_FOLDER,
                image_file
            )
        )

    print("\nEnhancement processing completed.")
    print(f"Outputs saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()