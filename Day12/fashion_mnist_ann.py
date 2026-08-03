# ==========================================================
# MLBench Summer Internship - Day 12
# Mini Project - Fashion MNIST ANN
# ==========================================================

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.utils import to_categorical

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

print("=" * 70)
print("LOADING FASHION MNIST DATASET")
print("=" * 70)

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print("Dataset Loaded Successfully!")

# --------------------------------------------------
# Dataset Information
# --------------------------------------------------

print("\nTraining Images :", X_train.shape)
print("Training Labels :", y_train.shape)

print("Testing Images  :", X_test.shape)
print("Testing Labels  :", y_test.shape)

# --------------------------------------------------
# Normalize Images
# --------------------------------------------------

print("\nNormalizing Images...")

X_train = X_train / 255.0
X_test = X_test / 255.0

# --------------------------------------------------
# One-Hot Encoding
# --------------------------------------------------

y_train_cat = to_categorical(y_train)

y_test_cat = to_categorical(y_test)

# --------------------------------------------------
# Build ANN
# --------------------------------------------------

print("\nBuilding Artificial Neural Network...")

model = Sequential([

    Input(shape=(28, 28)),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dense(
        64,
        activation="relu"
    ),

    Dense(
        10,
        activation="softmax"
    )

])

# --------------------------------------------------
# Compile Model
# --------------------------------------------------

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

print("\nModel Summary\n")

model.summary()

# --------------------------------------------------
# Train Model
# --------------------------------------------------

print("\nTraining Model...\n")

history = model.fit(

    X_train,

    y_train_cat,

    epochs=10,

    batch_size=32,

    validation_split=0.2,

    verbose=1

)

# --------------------------------------------------
# Evaluate Model
# --------------------------------------------------

print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(

    X_test,

    y_test_cat,

    verbose=0

)

print("=" * 70)

print(f"Testing Accuracy : {accuracy * 100:.2f}%")

print(f"Testing Loss     : {loss:.4f}")

print("=" * 70)

# --------------------------------------------------
# Save Model
# --------------------------------------------------

model.save("fashion_ann_model.keras")

print("\nModel saved successfully!")

# --------------------------------------------------
# Accuracy Graph
# --------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")

plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig("training_accuracy.png")

plt.show()

# --------------------------------------------------
# Loss Graph
# --------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig("training_loss.png")

plt.show()

print("\nGraphs saved successfully!")
# --------------------------------------------------
# Sample Predictions
# --------------------------------------------------

print("\nMaking Sample Predictions...")

class_names = [

    "T-shirt / Top",
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

predictions = model.predict(X_test)

predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(15, 8))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(X_test[i], cmap="gray")

    actual = class_names[y_test[i]]

    predicted = class_names[predicted_labels[i]]

    color = "green" if actual == predicted else "red"

    plt.title(
        f"Actual:\n{actual}\n\nPredicted:\n{predicted}",
        fontsize=9,
        color=color
    )

    plt.axis("off")

plt.tight_layout()

plt.savefig("sample_predictions.png")

plt.show()

print("\nSample predictions saved as sample_predictions.png")