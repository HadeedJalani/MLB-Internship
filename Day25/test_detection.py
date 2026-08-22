import cv2

from feature_detection import (
    detect_harris_corners,
    detect_orb_keypoints,
)


IMAGE_PATH = "sample_images/test1.jpg"


def main():

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print(f"Could not load image: {IMAGE_PATH}")
        return

    # -------------------------
    # Harris
    # -------------------------

    harris_image, corner_count = detect_harris_corners(
        image
    )

    # -------------------------
    # ORB
    # -------------------------

    orb_image, keypoints, descriptors = detect_orb_keypoints(
        image
    )

    # -------------------------
    # Save outputs
    # -------------------------

    cv2.imwrite(
        "outputs/harris_corners.jpg",
        harris_image
    )

    cv2.imwrite(
        "outputs/orb_keypoints.jpg",
        orb_image
    )

    print("\n===== FEATURE DETECTION =====")
    print(f"Harris corners : {corner_count}")
    print(f"ORB keypoints  : {len(keypoints)}")

    print("\nOutputs:")
    print("outputs/harris_corners.jpg")
    print("outputs/orb_keypoints.jpg")


if __name__ == "__main__":
    main()