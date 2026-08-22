import cv2

from feature_matching import match_features, get_match_statistics


IMAGE_1 = "sample_images/test1.jpg"
IMAGE_2 = "sample_images/test2.jpg"


def main():
    image1 = cv2.imread(IMAGE_1)
    image2 = cv2.imread(IMAGE_2)

    if image1 is None:
        print(f"Could not load: {IMAGE_1}")
        return

    if image2 is None:
        print(f"Could not load: {IMAGE_2}")
        return

    (
        matched_image,
        keypoints1,
        keypoints2,
        good_matches,
    ) = match_features(
        image1,
        image2,
        nfeatures=1000,
        ratio_threshold=0.75,
    )

    if matched_image is None:
        print("No descriptors were found.")
        return

    stats = get_match_statistics(
        keypoints1,
        keypoints2,
        good_matches,
    )

    print("\n===== ORB FEATURE MATCHING =====")
    print(f"Keypoints in image 1 : {stats['keypoints_image1']}")
    print(f"Keypoints in image 2 : {stats['keypoints_image2']}")
    print(f"Good matches         : {stats['good_matches']}")
    print(f"Match ratio          : {stats['match_ratio']:.2f}%")

    output_path = "outputs/test_matches.jpg"

    cv2.imwrite(
        output_path,
        matched_image,
    )

    print(f"\nResult saved to: {output_path}")


if __name__ == "__main__":
    main()