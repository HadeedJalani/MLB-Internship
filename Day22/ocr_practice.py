import os
import cv2
import easyocr


# ==========================================================
# Configuration
# ==========================================================

INPUT_DIR = "input_images"
OUTPUT_DIR = "extracted_text"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# Initialize EasyOCR
# ==========================================================

print("Initializing EasyOCR...")

reader = easyocr.Reader(
    ["en"],
    gpu=False
)


# ==========================================================
# Image Preprocessing
# ==========================================================

def preprocess_image(image):
    """
    Apply basic preprocessing to improve OCR readability.
    """

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    enhanced = cv2.equalizeHist(
        grayscale
    )

    denoised = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0
    )

    return denoised


# ==========================================================
# OCR Processing
# ==========================================================

def extract_text(image_path, preprocess=True):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    if preprocess:
        image_for_ocr = preprocess_image(image)
    else:
        image_for_ocr = image

    results = reader.readtext(
        image_for_ocr
    )

    extracted_lines = []

    for detection in results:

        text = detection[1]
        confidence = detection[2]

        extracted_lines.append(
            text
        )

        print(
            f"{text} "
            f"(confidence: {confidence:.2f})"
        )

    return extracted_lines


# ==========================================================
# Process Dataset
# ==========================================================

def main():

    image_files = [
        filename
        for filename in os.listdir(INPUT_DIR)
        if filename.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            )
        )
    ]

    if not image_files:

        print(
            f"No images found in '{INPUT_DIR}'."
        )

        return

    print(
        f"\nFound {len(image_files)} images."
    )

    print(
        "=" * 60
    )

    for index, filename in enumerate(
        image_files,
        start=1
    ):

        image_path = os.path.join(
            INPUT_DIR,
            filename
        )

        print(
            f"\n[{index}/{len(image_files)}] "
            f"Processing: {filename}"
        )

        print("-" * 60)

        try:

            extracted_text = extract_text(
                image_path,
                preprocess=True
            )

            output_filename = (
                os.path.splitext(filename)[0]
                + ".txt"
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                output_filename
            )

            with open(
                output_path,
                "w",
                encoding="utf-8"
            ) as text_file:

                text_file.write(
                    "\n".join(extracted_text)
                )

            print(
                f"\nSaved: {output_path}"
            )

        except Exception as error:

            print(
                f"Failed to process "
                f"{filename}: {error}"
            )

    print(
        "\n" + "=" * 60
    )

    print(
        "OCR processing completed."
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()