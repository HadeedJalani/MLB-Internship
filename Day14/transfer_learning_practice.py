# ==========================================================
# MLBench Summer Internship - Day 14
# Transfer Learning Practice
# MobileNetV2 Feature Extraction
# ==========================================================

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.models import Model

print("=" * 70)
print("Transfer Learning Practice - MobileNetV2")
print("=" * 70)

# ==========================================================
# Configuration
# ==========================================================

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 2

# ==========================================================
# Load Pretrained MobileNetV2
# ==========================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

print("✓ MobileNetV2 Loaded Successfully")

# ==========================================================
# Freeze Base Model
# ==========================================================

print("\nFreezing Base Model Layers...")

base_model.trainable = False

print("✓ Base Model Frozen")

# ==========================================================
# Build Custom Classification Head
# ==========================================================

print("\nBuilding Classification Head...")

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    256,
    activation="relu"
)(x)

x = Dropout(0.3)(x)

output = Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

print("✓ Custom Head Added")

# ==========================================================
# Compile Model
# ==========================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("✓ Model Compiled")

# ==========================================================
# Model Summary
# ==========================================================

print("\n" + "=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

model.summary()

# ==========================================================
# Architecture Information
# ==========================================================

print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

print(f"\nInput Shape : {model.input_shape}")
print(f"Output Shape: {model.output_shape}")

print(f"\nTotal Layers: {len(model.layers)}")

# ==========================================================
# Trainable Parameters
# ==========================================================

trainable_params = sum(
    tf.keras.backend.count_params(w)
    for w in model.trainable_weights
)

non_trainable_params = sum(
    tf.keras.backend.count_params(w)
    for w in model.non_trainable_weights
)

print(f"\nTrainable Parameters     : {trainable_params:,}")
print(f"Frozen Parameters        : {non_trainable_params:,}")
print(f"Total Parameters         : {trainable_params + non_trainable_params:,}")

# ==========================================================
# Display Layer Information
# ==========================================================

print("\n" + "=" * 70)
print("FIRST 20 LAYERS")
print("=" * 70)

for i, layer in enumerate(model.layers[:20]):
    print(
        f"{i+1:02d}. "
        f"{layer.name:35}"
        f"Trainable: {layer.trainable}"
    )

# ==========================================================
# Last Layers
# ==========================================================

print("\n" + "=" * 70)
print("LAST 10 LAYERS")
print("=" * 70)

for layer in model.layers[-10:]:
    print(
        f"{layer.name:35}"
        f"Trainable: {layer.trainable}"
    )

# ==========================================================
# Explain Transfer Learning
# ==========================================================

print("\n" + "=" * 70)
print("TRANSFER LEARNING SUMMARY")
print("=" * 70)

print("""
Transfer Learning:
------------------
Instead of training a CNN from scratch,
we use a model already trained on over
1 million ImageNet images.

Advantages:
✓ Faster Training
✓ Higher Accuracy
✓ Less Data Required
✓ Better Generalization

Workflow:
Input Image
      ↓
MobileNetV2 (Frozen Feature Extractor)
      ↓
Global Average Pooling
      ↓
Dense Layer (ReLU)
      ↓
Dropout
      ↓
Output Layer (Cat / Dog)

During initial training:
✔ MobileNetV2 remains frozen.
✔ Only the custom classification head learns.

Later:
✔ Some MobileNetV2 layers can be unfrozen
  for fine-tuning.
""")

print("=" * 70)
print("Transfer Learning Practice Completed Successfully!")
print("=" * 70)