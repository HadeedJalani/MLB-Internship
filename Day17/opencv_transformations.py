import cv2
import numpy as np
import os


INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images/transformations"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def rotate_image(image, angle):
    height, width = image.shape[:2]

    center = (width // 2, height // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE
    )


def translate_image(image, x_shift, y_shift):
    height, width = image.shape[:2]

    matrix = np.float32([
        [1, 0, x_shift],
        [0, 1, y_shift]
    ])

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE
    )


def scale_image(image, scale):
    height, width = image.shape[:2]

    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


def affine_transform(image):
    height, width = image.shape[:2]

    source_points = np.float32([
        [0, 0],
        [width - 1, 0],
        [0, height - 1]
    ])

    destination_points = np.float32([
        [0, 0],
        [width - 100, 50],
        [100, height - 50]
    ])

    matrix = cv2.getAffineTransform(
        source_points,
        destination_points
    )

    return cv2.warpAffine(
        image,
        matrix,
        (width, height)
    )


def perspective_transform(image):

    height, width = image.shape[:2]

    source_points = np.float32([
        [50, 50],
        [width - 50, 30],
        [width - 30, height - 50],
        [30, height - 30]
    ])

    destination_points = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ])

    matrix = cv2.getPerspectiveTransform(
        source_points,
        destination_points
    )

    return cv2.warpPerspective(
        image,
        matrix,
        (width, height)
    )


def process_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print(f"Unable to read: {image_path}")
        return

    filename = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    # Translation
    translated = translate_image(
        image,
        80,
        50
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            f"{filename}_translated.jpg"
        ),
        translated
    )

    # Rotation
    for angle in [30, 90, 180]:

        rotated = rotate_image(
            image,
            angle
        )

        cv2.imwrite(
            os.path.join(
                OUTPUT_FOLDER,
                f"{filename}_rotate_{angle}.jpg"
            ),
            rotated
        )

    # Scaling
    for scale in [0.5, 1.5]:

        scaled = scale_image(
            image,
            scale
        )

        cv2.imwrite(
            os.path.join(
                OUTPUT_FOLDER,
                f"{filename}_scale_{scale}.jpg"
            ),
            scaled
        )

    # Affine
    affine = affine_transform(image)

    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            f"{filename}_affine.jpg"
        ),
        affine
    )

    # Perspective
    perspective = perspective_transform(image)

    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            f"{filename}_perspective.jpg"
        ),
        perspective
    )


def main():

    print("=" * 60)
    print("Day 17 - OpenCV Image Transformations")
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

        image_path = os.path.join(
            INPUT_FOLDER,
            image_file
        )

        print(f"Processing: {image_file}")

        process_image(image_path)

    print("\nTransformation processing completed.")
    print(f"Outputs saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()