 # ==========================================================
# MLBench Summer Internship - Day 12
# Practice 3 - Activation Functions
# ==========================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

# --------------------------------------------------
# Function to Build Model
# --------------------------------------------------

def build_model(activation_function):

    model = Sequential([

        Input(shape=(4,)),

        Dense(
            units=16,
            activation=activation_function
        ),

        Dense(
            units=8,
            activation=activation_function
        ),

        Dense(
            units=3,
            activation="softmax"
        )

    ])

    model.compile(

        optimizer="adam",

        loss="categorical_crossentropy",

        metrics=["accuracy"]

    )

    return model


# --------------------------------------------------
# Activation Functions to Compare
# --------------------------------------------------

activation_functions = [

    "relu",
    "sigmoid",
    "tanh"

]

# --------------------------------------------------
# Display Model Summaries
# --------------------------------------------------

for activation in activation_functions:

    print("\n" + "=" * 70)
    print(f"MODEL USING {activation.upper()} ACTIVATION")
    print("=" * 70)

    model = build_model(activation)

    model.summary()

# --------------------------------------------------
# Explanation
# --------------------------------------------------

print("\n" + "=" * 70)
print("ACTIVATION FUNCTION COMPARISON")
print("=" * 70)

print("""

1. ReLU (Rectified Linear Unit)
--------------------------------
Formula:
f(x) = max(0, x)

Advantages
- Fast training
- Computationally efficient
- Most widely used in hidden layers
- Reduces the vanishing gradient problem

Common Uses
- Hidden layers in Deep Learning
- CNNs
- Image Classification

----------------------------------------------------

2. Sigmoid
--------------------------------
Formula:
f(x) = 1 / (1 + e^-x)

Output Range
0 to 1

Advantages
- Produces probability-like outputs

Disadvantages
- Suffers from vanishing gradients
- Slower convergence

Common Uses
- Binary Classification Output Layer

----------------------------------------------------

3. Tanh
--------------------------------
Formula:
f(x) = (e^x - e^-x) / (e^x + e^-x)

Output Range
-1 to 1

Advantages
- Zero-centered output
- Better than Sigmoid for hidden layers

Disadvantages
- Can still suffer from vanishing gradients

Common Uses
- Hidden layers in some neural networks
- Recurrent Neural Networks (RNNs)

----------------------------------------------------

Softmax
--------------------------------
Used only in the output layer.

Purpose
- Converts outputs into probabilities.
- The probabilities always sum to 100%.

Used For
- Multi-class classification.

""")

print("=" * 70)
print("Practice completed successfully!")
print("=" * 70)