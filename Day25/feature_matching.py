import cv2


def match_orb_features(
    image1,
    image2,
    nfeatures=1000,
    ratio_threshold=0.75
):
    """
    Detect ORB features in two images and match them
    using BFMatcher + Lowe's ratio test.

    Returns:
        result_image
        keypoints1
        keypoints2
        good_matches
    """

    if image1 is None or image2 is None:
        raise ValueError("Both images are required.")

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # ORB detector
    orb = cv2.ORB_create(
        nfeatures=int(nfeatures)
    )

    keypoints1, descriptors1 = orb.detectAndCompute(
        gray1,
        None
    )

    keypoints2, descriptors2 = orb.detectAndCompute(
        gray2,
        None
    )

    # No features detected
    if descriptors1 is None or descriptors2 is None:
        return (
            image1.copy(),
            keypoints1 or [],
            keypoints2 or [],
            []
        )

    # ORB uses binary descriptors -> Hamming distance
    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False
    )

    knn_matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    good_matches = []

    for pair in knn_matches:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    # Sort strongest matches first
    good_matches = sorted(
        good_matches,
        key=lambda match: match.distance
    )

    # Display up to 100 matches
    matches_to_draw = good_matches[:100]

    result_image = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return (
        result_image,
        keypoints1,
        keypoints2,
        good_matches
    )