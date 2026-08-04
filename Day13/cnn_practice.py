# ==========================================================
# MLBench Summer Internship - Day 13
# Practice 1 & 2
# Convolutional Neural Networks (CNN)
# ==========================================================

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense
)

print("=" * 70)
print("CONVOLUTIONAL NEURAL NETWORK PRACTICE")
print("=" * 70)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

print("\nLoading Fashion MNIST Dataset...\n")

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("Training Images :", x_train.shape)
print("Training Labels :", y_train.shape)
print("Testing Images  :", x_test.shape)
print("Testing Labels  :", y_test.shape)

# ----------------------------------------------------------
# Class Names
# ----------------------------------------------------------

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

# ----------------------------------------------------------
# Display Sample Images
# ----------------------------------------------------------

print("\nDisplaying Sample Images...\n")

plt.figure(figsize=(12,5))

for i in range(10):

    plt.subplot(2,5,i+1)

    plt.imshow(x_train[i], cmap="gray")

    plt.title(class_names[y_train[i]], fontsize=9)

    plt.axis("off")

plt.tight_layout()

plt.show()

# ----------------------------------------------------------
# Normalize Images
# ----------------------------------------------------------

x_train = x_train / 255.0
x_test = x_test / 255.0

# CNN expects 4D tensors
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

print("Images normalized successfully.")

# ----------------------------------------------------------
# Build CNN
# ----------------------------------------------------------

print("\nBuilding CNN Model...\n")

model = Sequential([

    Input(shape=(28,28,1)),

    Conv2D(

        filters=32,
        kernel_size=(3,3),
        activation="relu"

    ),

    MaxPooling2D(

        pool_size=(2,2)

    ),

    Flatten(),

    Dense(

        units=128,
        activation="relu"

    ),

    Dense(

        units=10,
        activation="softmax"

    )

])

# ----------------------------------------------------------
# Compile
# ----------------------------------------------------------

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------

print("=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

model.summary()

# ----------------------------------------------------------
# Train
# ----------------------------------------------------------

print("\nTraining CNN...\n")

history = model.fit(

    x_train,

    y_train,

    epochs=5,

    batch_size=32,

    validation_split=0.2,

    verbose=1

)

# ----------------------------------------------------------
# Evaluate
# ----------------------------------------------------------

print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(

    x_test,

    y_test,

    verbose=0

)

print("=" * 70)
print(f"Testing Accuracy : {accuracy*100:.2f}%")
print(f"Testing Loss     : {loss:.4f}")
print("=" * 70)

print("\nPractice Completed Successfully.")