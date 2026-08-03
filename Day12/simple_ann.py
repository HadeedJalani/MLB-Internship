# ==========================================================
# MLBench Summer Internship - Day 12
# Practice 2 - Building Your First Artificial Neural Network
# ==========================================================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

# --------------------------------------------------
# Build Neural Network
# --------------------------------------------------

model = Sequential([

    Input(shape=(4,)),

    Dense(
        units=16,
        activation="relu"
    ),

    Dense(
        units=8,
        activation="relu"
    ),

    Dense(
        units=3,
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

# --------------------------------------------------
# Display Model Summary
# --------------------------------------------------

print("=" * 70)
print("ARTIFICIAL NEURAL NETWORK")
print("=" * 70)

print("\nModel Summary\n")

model.summary()

# --------------------------------------------------
# Explain Each Layer
# --------------------------------------------------

print("\n" + "=" * 70)
print("LAYER EXPLANATION")
print("=" * 70)

print("""
Input Layer
-----------
Shape        : (4,)
Purpose      : Receives four input features and passes them to the network.

Hidden Layer 1
--------------
Neurons      : 16
Activation   : ReLU
Purpose      : Learns complex patterns from the input data.

Hidden Layer 2
--------------
Neurons      : 8
Activation   : ReLU
Purpose      : Extracts higher-level features from the previous layer.

Output Layer
------------
Neurons      : 3
Activation   : Softmax
Purpose      : Produces probabilities for three output classes.

Optimizer
---------
Adam
Purpose      : Updates the model weights to minimize the loss.

Loss Function
-------------
Categorical Crossentropy
Purpose      : Measures the difference between predicted and actual class labels.

Evaluation Metric
-----------------
Accuracy
Purpose      : Measures the percentage of correctly classified samples.
""")

print("=" * 70)
print("Neural Network created successfully!")
print("=" * 70)