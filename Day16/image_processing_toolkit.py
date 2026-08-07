# ==========================================================
# MLB Summer Internship - Day 16
# Image Processing Toolkit using OpenCV
# Menu Driven Application
# ==========================================================

import cv2
import os

# ==========================================================
# Paths
# ==========================================================

IMAGE_PATH = "input_images/landscape.jpg"

OUTPUT_FOLDER = "output_images/toolkit_outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# Load Image
# ==========================================================

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"\nImage not found:\n{IMAGE_PATH}")

image = cv2.imread(IMAGE_PATH)

current_image = image.copy()

print("=" * 70)
print("IMAGE PROCESSING TOOLKIT")
print("=" * 70)

# ==========================================================
# Functions
# ==========================================================

def show_image(img, title="Image"):

    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(img, filename):

    path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(path, img)
    print(f"\n✓ Saved : {path}")


def grayscale():

    global current_image

    current_image = cv2.cvtColor(
        current_image,
        cv2.COLOR_BGR2GRAY
    )

    show_image(current_image, "Grayscale")

    save_image(current_image, "grayscale.jpg")


def resize():

    global current_image

    width = int(input("Enter Width : "))
    height = int(input("Enter Height : "))

    current_image = cv2.resize(
        current_image,
        (width, height)
    )

    show_image(current_image, "Resized")

    save_image(current_image, "resized.jpg")


def rotate():

    global current_image

    print("\n1. Rotate 90°")
    print("2. Rotate 180°")
    print("3. Rotate 270°")

    choice = input("Choice : ")

    if choice == "1":

        current_image = cv2.rotate(
            current_image,
            cv2.ROTATE_90_CLOCKWISE
        )

    elif choice == "2":

        current_image = cv2.rotate(
            current_image,
            cv2.ROTATE_180
        )

    elif choice == "3":

        current_image = cv2.rotate(
            current_image,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )

    else:

        print("Invalid Choice")
        return

    show_image(current_image, "Rotated")

    save_image(current_image, "rotated.jpg")


def flip():

    global current_image

    print("\n1. Horizontal")
    print("2. Vertical")
    print("3. Both")

    choice = input("Choice : ")

    if choice == "1":

        current_image = cv2.flip(current_image, 1)

    elif choice == "2":

        current_image = cv2.flip(current_image, 0)

    elif choice == "3":

        current_image = cv2.flip(current_image, -1)

    else:

        print("Invalid Choice")
        return

    show_image(current_image, "Flipped")

    save_image(current_image, "flipped.jpg")


def crop():

    global current_image

    h, w = current_image.shape[:2]

    current_image = current_image[
        h//4:h*3//4,
        w//4:w*3//4
    ]

    show_image(current_image, "Cropped")

    save_image(current_image, "cropped.jpg")


def draw_shapes():

    global current_image

    img = current_image.copy()

    h, w = img.shape[:2]

    cv2.rectangle(
        img,
        (50,50),
        (350,250),
        (0,255,0),
        3
    )

    cv2.circle(
        img,
        (w//2,h//2),
        120,
        (255,0,0),
        3
    )

    cv2.line(
        img,
        (0,0),
        (w,h),
        (0,0,255),
        4
    )

    pts = [
        (500,80),
        (600,180),
        (550,280),
        (450,250)
    ]

    import numpy as np

    pts = np.array(pts)

    cv2.polylines(
        img,
        [pts],
        True,
        (255,255,0),
        3
    )

    current_image = img

    show_image(current_image, "Shapes")

    save_image(current_image, "drawing_shapes.jpg")


def add_text():

    global current_image

    text = input("Enter Text : ")

    cv2.putText(
        current_image,
        text,
        (50,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    show_image(current_image, "Text")

    save_image(current_image, "text_added.jpg")


def brightness():

    global current_image

    value = int(input("Brightness (-100 to 100): "))

    current_image = cv2.convertScaleAbs(
        current_image,
        alpha=1,
        beta=value
    )

    show_image(current_image)

    save_image(current_image, "brightness.jpg")


def compare_rgb():

    rgb = cv2.cvtColor(
        current_image,
        cv2.COLOR_BGR2RGB
    )

    cv2.imshow("Original BGR", current_image)
    cv2.imshow("RGB Version", rgb)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==========================================================
# Menu
# ==========================================================

while True:

    print("\n")

    print("="*60)

    print("1. Load Original Image")

    print("2. Convert to Grayscale")

    print("3. Resize")

    print("4. Rotate")

    print("5. Flip")

    print("6. Crop")

    print("7. Draw Shapes")

    print("8. Add Text")

    print("9. Adjust Brightness")

    print("10. Compare RGB & BGR")

    print("11. Save Current Image")

    print("0. Exit")

    print("="*60)

    choice = input("Enter Choice : ")

    if choice == "1":

        current_image = image.copy()

        show_image(current_image)

    elif choice == "2":

        grayscale()

    elif choice == "3":

        resize()

    elif choice == "4":

        rotate()

    elif choice == "5":

        flip()

    elif choice == "6":

        crop()

    elif choice == "7":

        draw_shapes()

    elif choice == "8":

        add_text()

    elif choice == "9":

        brightness()

    elif choice == "10":

        compare_rgb()

    elif choice == "11":

        save_image(
            current_image,
            "final_output.jpg"
        )

    elif choice == "0":

        print("\nToolkit Closed Successfully!")

        break

    else:

        print("\nInvalid Choice!")