import tensorflow as tf
import numpy as np
from PIL import Image

# ======================================
# Load Model
# ======================================

model = tf.keras.models.load_model("best_model.keras")
# ======================================
# Image Path
# ======================================

IMAGE_PATH = "dog_train_00024.jpg"

# Example:
# IMAGE_PATH = "dataset/test/dogs/dog_00123.jpg"

# ======================================
# Load Image
# ======================================

img = Image.open(IMAGE_PATH).convert("RGB")

print("\nOriginal Image Size:", img.size)

img = img.resize((224, 224))

img = np.array(img).astype(np.float32)

# MobileNetV2 preprocessing
img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

img = np.expand_dims(img, axis=0)

# ======================================
# Prediction
# ======================================

prediction = model.predict(img, verbose=0)

cat_prob = float(prediction[0][0])
dog_prob = float(prediction[0][1])

print("\n==============================")
print("RAW OUTPUT")
print("==============================")

print(prediction)

print("\nCat Probability :", cat_prob)
print("Dog Probability :", dog_prob)

predicted_class = np.argmax(prediction)

class_names = ["Cat", "Dog"]

print("\n==============================")
print("FINAL PREDICTION")
print("==============================")

print("Predicted Class :", class_names[predicted_class])
print("Confidence      :", prediction[0][predicted_class] * 100, "%")