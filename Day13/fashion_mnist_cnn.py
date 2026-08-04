# ==========================================================
# MLBench Summer Internship - Day 13
# Fashion MNIST Image Classifier using CNN
# ==========================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 70)
print("FASHION MNIST CNN CLASSIFIER")
print("=" * 70)

print("\nLoading Dataset...\n")

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("Training Images :", x_train.shape)
print("Training Labels :", y_train.shape)
print("Testing Images  :", x_test.shape)
print("Testing Labels  :", y_test.shape)

# ==========================================================
# Class Names
# ==========================================================

class_names = [

    "T-shirt/Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"

]

# ==========================================================
# Display Sample Images
# ==========================================================

print("\nDisplaying Sample Images...\n")

plt.figure(figsize=(12,6))

for i in range(10):

    plt.subplot(2,5,i+1)

    plt.imshow(x_train[i], cmap="gray")

    plt.title(class_names[y_train[i]], fontsize=9)

    plt.axis("off")

plt.tight_layout()

plt.show()

# ==========================================================
# Normalize Dataset
# ==========================================================

print("\nNormalizing Images...\n")

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# CNN requires 4D tensors

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("Training Shape :", x_train.shape)
print("Testing Shape  :", x_test.shape)

# ==========================================================
# Build CNN
# ==========================================================

print("\nBuilding CNN...\n")

model = Sequential([

    Input(shape=(28,28,1)),

    Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation="relu"
    ),

    MaxPooling2D(pool_size=(2,2)),

    Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation="relu"
    ),

    MaxPooling2D(pool_size=(2,2)),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        10,
        activation="softmax"
    )

])

# ==========================================================
# Compile Model
# ==========================================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

# ==========================================================
# Model Summary
# ==========================================================

print("=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

model.summary()

# ==========================================================
# Train Model
# ==========================================================

print("\nTraining CNN...\n")

history = model.fit(

    x_train,

    y_train,

    validation_split=0.2,

    epochs=10,

    batch_size=64,

    verbose=1

)
# ==========================================================
# Evaluate Model
# ==========================================================

print("\nEvaluating Model...\n")

test_loss, test_accuracy = model.evaluate(

    x_test,

    y_test,

    verbose=0

)

print("=" * 70)
print(f"Testing Accuracy : {test_accuracy*100:.2f}%")
print(f"Testing Loss     : {test_loss:.4f}")
print("=" * 70)

# ==========================================================
# Save Model
# ==========================================================

print("\nSaving Trained Model...\n")

model.save("cnn_model.keras")

print("Model saved successfully as cnn_model.keras")

# ==========================================================
# Plot Accuracy Curve
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(

    history.history["accuracy"],

    label="Training Accuracy",

    linewidth=2

)

plt.plot(

    history.history["val_accuracy"],

    label="Validation Accuracy",

    linewidth=2

)

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig(

    "accuracy_curve.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

# ==========================================================
# Plot Loss Curve
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(

    history.history["loss"],

    label="Training Loss",

    linewidth=2

)

plt.plot(

    history.history["val_loss"],

    label="Validation Loss",

    linewidth=2

)

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig(

    "loss_curve.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

print("\nAccuracy and Loss graphs saved successfully.")

# ==========================================================
# Make Predictions
# ==========================================================

print("\nMaking Predictions...\n")

predictions = model.predict(x_test, verbose=0)

predicted_labels = np.argmax(predictions, axis=1)

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_test, predicted_labels)

plt.figure(figsize=(8, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    cmap="Blues",
    xticks_rotation=45,
    values_format="d"
)

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Confusion Matrix saved successfully.")

# ==========================================================
# Correct Predictions
# ==========================================================

correct_indices = np.where(predicted_labels == y_test)[0]

print(f"\nCorrect Predictions: {len(correct_indices)}")

plt.figure(figsize=(15,6))

num_correct = min(10, len(correct_indices))

for i in range(num_correct):

    index = correct_indices[i]

    plt.subplot(2,5,i+1)

    plt.imshow(x_test[index].squeeze(), cmap="gray")

    plt.title(
        f"P: {class_names[predicted_labels[index]]}\n"
        f"A: {class_names[y_test[index]]}",
        fontsize=8
    )

    plt.axis("off")

plt.tight_layout()

plt.savefig("correct_predictions.png", dpi=300)

plt.show()

plt.close()

# ==========================================================
# Incorrect Predictions
# ==========================================================

incorrect_indices = np.where(predicted_labels != y_test)[0]

print(f"Incorrect Predictions: {len(incorrect_indices)}")

plt.figure(figsize=(15,6))

num_incorrect = min(10, len(incorrect_indices))

for i in range(num_incorrect):

    index = incorrect_indices[i]

    plt.subplot(2,5,i+1)

    plt.imshow(x_test[index].squeeze(), cmap="gray")

    plt.title(
        f"P: {class_names[predicted_labels[index]]}\n"
        f"A: {class_names[y_test[index]]}",
        fontsize=8,
        color="red"
    )

    plt.axis("off")

plt.tight_layout()

plt.savefig("incorrect_predictions.png", dpi=300)

plt.show()

plt.close()
print("\n" + "=" * 70)
print("CNN PROJECT COMPLETED SUCCESSFULLY")
...
