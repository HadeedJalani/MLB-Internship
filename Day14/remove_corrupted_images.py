from PIL import Image
import os

folders = [
    "dataset/train/cats",
    "dataset/train/dogs",
    "dataset/validation/cats",
    "dataset/validation/dogs",
    "dataset/test/cats",
    "dataset/test/dogs",
]

removed = 0

for folder in folders:

    print(f"\nChecking {folder}")

    for filename in os.listdir(folder):

        path = os.path.join(folder, filename)

        try:
            img = Image.open(path)
            img.verify()

        except Exception:
            print("Removing:", path)
            os.remove(path)
            removed += 1

print("\n===================================")
print(f"Finished. Removed {removed} corrupted images.")
print("===================================")