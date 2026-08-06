from PIL import Image
import os

DATASET = "dataset"

bad = []

for split in ["train", "validation", "test"]:
    for cls in ["cats", "dogs"]:

        folder = os.path.join(DATASET, split, cls)

        for file in os.listdir(folder):

            path = os.path.join(folder, file)

            try:
                img = Image.open(path)
                img.verify()

            except Exception:
                bad.append(path)

print("\n=========================")
print("BAD IMAGES FOUND:", len(bad))
print("=========================")

for b in bad:
    print(b)