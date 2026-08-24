import cv2
import numpy as np


def read_image(image_source):
    """
    Read an image from either a file path or a Streamlit-uploaded object.
    Returns a BGR OpenCV image.
    """

    if isinstance(image_source, str):
        image = cv2.imread(image_source)

        if image is None:
            raise ValueError(f"Could not read image: {image_source}")

        return image

    image_bytes = image_source.read()

    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode the uploaded image.")

    return image


def to_grayscale(image):
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# =========================================================
# BASIC THRESHOLDING
# =========================================================

def binary_threshold(image, threshold=127):
    """
    Standard fixed binary thresholding.
    """

    gray = to_grayscale(image)

    _, result = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    return result


def adaptive_threshold(image):
    """
    Adaptive Gaussian thresholding.

    Useful when the image has uneven lighting or shadows.
    """

    gray = to_grayscale(image)

    result = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return result


def otsu_threshold(image):
    """
    Automatic Otsu thresholding.
    """

    gray = to_grayscale(image)

    _, result = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return result


# =========================================================
# FOREGROUND / BACKGROUND SEGMENTATION
# =========================================================

def foreground_segmentation(image):
    """
    Simple foreground/background segmentation using
    Otsu thresholding and morphological cleanup.

    Returns a binary foreground mask.
    """

    gray = to_grayscale(image)

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return mask


def apply_mask(image, mask):
    """
    Apply a binary mask to the original image.
    Background becomes black.
    """

    if len(mask.shape) != 2:
        raise ValueError(
            "Mask must be a grayscale/binary image."
        )

    if mask.shape[:2] != image.shape[:2]:
        raise ValueError(
            "Image and mask dimensions do not match."
        )

    return cv2.bitwise_and(
        image,
        image,
        mask=mask
    )


def segment_foreground(image):
    """
    Complete foreground segmentation pipeline.

    Returns:
        mask
        segmented image
    """

    mask = foreground_segmentation(image)
    segmented = apply_mask(
        image,
        mask
    )

    return mask, segmented


# =========================================================
# WATERSHED SEGMENTATION
# =========================================================

def watershed_segmentation(image):
    """
    Watershed-based image segmentation.

    The watershed algorithm treats the image as a
    topographic surface and separates connected regions.

    Returns a colored segmentation result.
    """

    gray = to_grayscale(image)

    # Smooth image
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Binary threshold
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove small noise
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    opening = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    # Sure background
    sure_bg = cv2.dilate(
        opening,
        kernel,
        iterations=3
    )

    # Distance transform
    dist_transform = cv2.distanceTransform(
        opening,
        cv2.DIST_L2,
        5
    )

    # Sure foreground
    _, sure_fg = cv2.threshold(
        dist_transform,
        0.5 * dist_transform.max(),
        255,
        0
    )

    sure_fg = np.uint8(sure_fg)

    # Unknown region
    unknown = cv2.subtract(
        sure_bg,
        sure_fg
    )

    # Marker labels
    _, markers = cv2.connectedComponents(
        sure_fg
    )

    markers = markers + 1

    markers[unknown == 255] = 0

    # Apply watershed
    image_copy = image.copy()

    markers = cv2.watershed(
        image_copy,
        markers
    )

    # Create result
    result = image.copy()

    # Watershed boundaries
    result[markers == -1] = [0, 0, 255]

    return result


# =========================================================
# BACKGROUND REMOVAL USING GRABCUT
# =========================================================

def background_removal(image):
    """
    Remove the background using OpenCV GrabCut.

    The image border is treated as background while
    the central region is treated as probable foreground.

    Returns the foreground with the background set to black.
    """

    height, width = image.shape[:2]

    if height < 20 or width < 20:
        raise ValueError(
            "Image is too small for background removal."
        )

    # GrabCut mask
    mask = np.zeros(
        (height, width),
        np.uint8
    )

    # Keep a small margin around the object
    margin_x = max(
        5,
        int(width * 0.05)
    )

    margin_y = max(
        5,
        int(height * 0.05)
    )

    rect = (
        margin_x,
        margin_y,
        max(1, width - 2 * margin_x),
        max(1, height - 2 * margin_y)
    )

    bgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    fgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    cv2.grabCut(
        image,
        mask,
        rect,
        bgd_model,
        fgd_model,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    # Foreground pixels
    foreground_mask = np.where(
        (mask == cv2.GC_FGD) |
        (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype("uint8")

    # Clean mask
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    foreground_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    foreground_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    result = cv2.bitwise_and(
        image,
        image,
        mask=foreground_mask
    )

    return result, foreground_mask


# =========================================================
# COMPLETE FOREGROUND PIPELINE
# =========================================================

def process_image(image, method):
    """
    Main interface for the Streamlit application.
    """

    if image is None:
        raise ValueError(
            "No image was provided."
        )

    method = method.strip().lower()

    if method == "binary thresholding":
        return binary_threshold(image)

    if method == "adaptive thresholding":
        return adaptive_threshold(image)

    if method == "otsu thresholding":
        return otsu_threshold(image)

    if method == "foreground segmentation":
        _, segmented = segment_foreground(image)
        return segmented

    if method == "watershed segmentation":
        return watershed_segmentation(image)

    if method == "background removal":
        segmented, _ = background_removal(image)
        return segmented

    raise ValueError(
        f"Unknown segmentation method: {method}"
    )


# =========================================================
# SAVE IMAGE
# =========================================================

def save_image(image, output_path):
    """
    Save an OpenCV image to disk.
    """

    success = cv2.imwrite(
        output_path,
        image
    )

    if not success:
        raise IOError(
            f"Could not save image to: {output_path}"
        )

    return output_path