import tensorflow as tf
import numpy as np

print("Loading model...")

model = tf.keras.models.load_model("best_model.keras")

print("Loading dataset...")

dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

class_names = dataset.class_names

print("Classes:", class_names)

# Ignore corrupted images
dataset = dataset.apply(tf.data.experimental.ignore_errors())

correct = 0
total = 0

for images, labels in dataset:

    predictions = model.predict(images, verbose=0)

    predicted = np.argmax(predictions, axis=1)

    correct += np.sum(predicted == labels.numpy())

    total += len(labels)

print("\n==============================")
print("Evaluation Finished")
print("==============================")

print(f"Correct Predictions : {correct}")
print(f"Total Images        : {total}")
print(f"Accuracy            : {correct / total:.4f}")