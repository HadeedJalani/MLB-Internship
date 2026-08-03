# ==========================================================
# MLBench Summer Internship - Day 12
# TensorFlow Installation Verification
# ==========================================================

import tensorflow as tf
from tensorflow import keras

print("=" * 70)
print("TENSORFLOW INSTALLATION")
print("=" * 70)

print(f"\nTensorFlow Version : {tf.__version__}")

print(f"Keras Version      : {keras.__version__}")

print(f"\nGPU Available      : {len(tf.config.list_physical_devices('GPU')) > 0}")

print("\nAvailable Devices")

for device in tf.config.list_physical_devices():
    print(device)

print("\nTensorFlow imported successfully.")
print("Keras imported successfully.")

print("\nInstallation completed successfully.")