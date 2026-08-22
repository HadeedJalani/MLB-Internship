import cv2
import numpy as np


def detect_harris_corners(image, threshold=0.01):
    """
    Detect corners using Harris Corner Detection.

    Returns:
        output_image: image with detected corners
        corner_count: number of detected corners
    """

    if image is None:
        raise ValueError("Input image is None.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_float = np.float32(gray)

    harris = cv2.cornerHarris(
        gray_float,
        blockSize=2,
        ksize=3,
        k=0.04
    )

    # Improve visibility of corners
    harris = cv2.dilate(harris, None)

    max_response = harris.max()

    output = image.copy()

    if max_response <= 0:
        return output, 0

    threshold_value = threshold * max_response

    mask = harris > threshold_value

    # Draw Harris corners in red
    output[mask] = [0, 0, 255]

    corner_count = int(np.count_nonzero(mask))

    return output, corner_count


def detect_orb_keypoints(image, nfeatures=1000):
    """
    Detect ORB keypoints and descriptors.

    Returns:
        output_image
        keypoints
        descriptors
    """

    if image is None:
        raise ValueError("Input image is None.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(
        nfeatures=int(nfeatures)
    )

    keypoints, descriptors = orb.detectAndCompute(
        gray,
        None
    )

    output = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return output, keypoints, descriptors