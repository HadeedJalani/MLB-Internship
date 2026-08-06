# ==========================================================
# MLBench Summer Internship - Day 14
# Cats vs Dogs Image Classifier using Transfer Learning
# MobileNetV2 + TensorFlow/Keras
# ==========================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.utils import image_dataset_from_directory

# ==========================================================
# Reproducibility
# ==========================================================

SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ==========================================================
# Configuration
# ==========================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 0.0001

AUTOTUNE = tf.data.AUTOTUNE

# ==========================================================
# Dataset Paths
# ==========================================================

TRAIN_DIR = "dataset/train"

VALIDATION_DIR = "dataset/validation"

TEST_DIR = "dataset/test"

# ==========================================================
# Check Dataset
# ==========================================================

print("=" * 70)
print("Checking Dataset...")
print("=" * 70)

for path in [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nFolder not found:\n{path}"
        )

print("✓ Dataset Found")

# ==========================================================
# Load Dataset
# ==========================================================

print("\nLoading Dataset...\n")

train_dataset = image_dataset_from_directory(

    TRAIN_DIR,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=True,

    seed=SEED

)

validation_dataset = image_dataset_from_directory(

    VALIDATION_DIR,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False

)

test_dataset = image_dataset_from_directory(

    TEST_DIR,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False

)

print("\n✓ Dataset Loaded Successfully")

# ==========================================================
# Class Names
# ==========================================================

class_names = train_dataset.class_names

print("\nClasses Found:")

for i, cls in enumerate(class_names):

    print(f"{i} -> {cls}")

# ==========================================================
# Dataset Optimization
# ==========================================================

# Ignore corrupted images if encountered
train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
validation_dataset = validation_dataset.apply(tf.data.experimental.ignore_errors())
test_dataset = test_dataset.apply(tf.data.experimental.ignore_errors())

# Optimize dataset pipeline
train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)

# ==========================================================
# Display Dataset Information
# ==========================================================

print("\n" + "=" * 70)

print("DATASET INFORMATION")

print("=" * 70)

print(f"Training Classes : {class_names}")

print(f"Image Size       : {IMAGE_SIZE}")

print(f"Batch Size       : {BATCH_SIZE}")

# ==========================================================
# Visualize Sample Images
# ==========================================================

print("\nDisplaying Sample Images...")

plt.figure(figsize=(12, 12))

for images, labels in train_dataset.take(1):

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(class_names[labels[i]])

        plt.axis("off")

plt.tight_layout()

plt.savefig(
    "sample_images.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("✓ Sample Images Saved")

print("\nDataset Preparation Completed Successfully.")

# ==========================================================
# Data Augmentation
# ==========================================================

print("\n" + "=" * 70)
print("Creating Data Augmentation Pipeline...")
print("=" * 70)

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomRotation(0.2),

    tf.keras.layers.RandomZoom(0.2),

    tf.keras.layers.RandomContrast(0.2),

    tf.keras.layers.RandomBrightness(0.2)

], name="Data_Augmentation")

print("✓ Data Augmentation Ready")

# ==========================================================
# MobileNetV2 Preprocessing
# ==========================================================

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# ==========================================================
# Load MobileNetV2
# ==========================================================

print("\nLoading MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(

    input_shape=(224, 224, 3),

    include_top=False,

    weights="imagenet"

)

print("✓ Pre-trained Weights Loaded")

# ==========================================================
# Freeze Base Model
# ==========================================================

base_model.trainable = False

print("✓ Base Model Frozen")

# ==========================================================
# Build Transfer Learning Model
# ==========================================================

print("\nBuilding Model...")

inputs = tf.keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.30)(x)

x = tf.keras.layers.Dense(

    256,

    activation="relu"

)(x)

x = tf.keras.layers.Dropout(0.20)(x)

outputs = tf.keras.layers.Dense(

    2,

    activation="softmax"

)(x)

model = tf.keras.Model(

    inputs,

    outputs,

    name="Cats_vs_Dogs_MobileNetV2"

)

print("✓ Model Built Successfully")

# ==========================================================
# Compile Model
# ==========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=LEARNING_RATE

    ),

    loss="sparse_categorical_crossentropy",

    metrics=[

        "accuracy"

    ]

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
# Parameter Statistics
# ==========================================================

trainable_params = np.sum(
    [tf.keras.backend.count_params(w)
     for w in model.trainable_weights]
)

non_trainable_params = np.sum(
    [tf.keras.backend.count_params(w)
     for w in model.non_trainable_weights]
)

print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

print(f"Trainable Parameters     : {trainable_params:,}")

print(f"Frozen Parameters        : {non_trainable_params:,}")

print(f"Total Parameters         : {trainable_params + non_trainable_params:,}")

print(f"\nInput Shape              : {model.input_shape}")

print(f"Output Shape             : {model.output_shape}")

# ==========================================================
# Architecture Diagram
# ==========================================================


# ==========================================================
# Callbacks
# ==========================================================

print("\n" + "=" * 70)
print("Creating Callbacks...")
print("=" * 70)

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True,

    verbose=1

)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(

    "best_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.2,

    patience=2,

    min_lr=1e-6,

    verbose=1

)

callbacks = [

    early_stopping,

    model_checkpoint,

    reduce_lr

]

# ==========================================================
# Initial Training
# ==========================================================
import os

MODEL_PATH = "cats_vs_dogs_model.keras"

FINE_TUNE_EPOCHS = 5

total_epochs = EPOCHS + FINE_TUNE_EPOCHS

history = None

history_fine = None

if os.path.exists(MODEL_PATH):

    print("\nModel already exists.")
    print("Loading Saved Model...")

    model = tf.keras.models.load_model(MODEL_PATH)

else:

    print("\nNo Saved Model Found.")
    print("Training New Model...")

    print("\n" + "=" * 70)
    print("INITIAL TRAINING (Feature Extraction)")
    print("=" * 70)

    history = model.fit(

        train_dataset,

        validation_data=validation_dataset,

        epochs=EPOCHS,

        callbacks=callbacks

    )

    print("\n✓ Initial Training Completed")

    # ==========================================================
    # Fine-Tuning Setup & Recompile
    # ==========================================================

    print("\n" + "=" * 70)
    print("Starting Fine-Tuning...")
    print("=" * 70)

    base_model.trainable = True

    fine_tune_at = 120

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    print(f"Frozen Layers : {fine_tune_at}")
    print(f"Trainable Layers : {len(base_model.layers)-fine_tune_at}")

    # Recompile with smaller learning rate prior to fine-tuning
    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=LEARNING_RATE / 10

        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    print("✓ Model Recompiled")

    # ==========================================================
    # Fine-Tuning Training
    # ==========================================================

    history_fine = model.fit(

        train_dataset,

        validation_data=validation_dataset,

        epochs=total_epochs,

        initial_epoch=history.epoch[-1] + 1,

        callbacks=callbacks

    )

    print("\n✓ Fine-Tuning Completed")

    model.save(MODEL_PATH)

    print("\n✓ Model Saved Successfully")

# ==========================================================
# Merge Histories
# ==========================================================

if history is not None and history_fine is not None:

    accuracy = (
        history.history["accuracy"] +
        history_fine.history["accuracy"]
    )

    val_accuracy = (
        history.history["val_accuracy"] +
        history_fine.history["val_accuracy"]
    )

    loss = (
        history.history["loss"] +
        history_fine.history["loss"]
    )

    val_loss = (
        history.history["val_loss"] +
        history_fine.history["val_loss"]
    )

    print("\nTraining History Combined Successfully")

else:

    accuracy = []
    val_accuracy = []
    loss = []
    val_loss = []

    print("\nTraining history skipped because the saved model was loaded.")

# ==========================================================
# Model Evaluation
# ==========================================================

#print("\n" + "=" * 70)
#print("Evaluating Model...")
#print("=" * 70)

#test_loss, test_accuracy = model.evaluate(test_dataset)

#print(f"\nTest Accuracy : {test_accuracy:.4f}")
#print(f"Test Loss     : {test_loss:.4f}")

# ==========================================================
# Model Evaluation
# ==========================================================

print("\n" + "=" * 70)
print("Evaluating Model...")
print("=" * 70)

print("\n" + "=" * 60)
print("MODEL USED FOR EVALUATION")
print("=" * 60)

print("Model Name:", model.name)

test_loss, test_accuracy = model.evaluate(test_dataset)

print(f"\nTest Accuracy : {test_accuracy:.4f}")
print(f"Test Loss     : {test_loss:.4f}")

print("\nSaving evaluated model...")

model.save("final_test_model.keras")

print("✓ Model saved as final_test_model.keras")

# ==========================================================
# Accuracy Curve
# ==========================================================

if len(accuracy) > 0:

    plt.figure(figsize=(10,6))

    plt.plot(accuracy, label="Training Accuracy", linewidth=2)

    plt.plot(val_accuracy, label="Validation Accuracy", linewidth=2)

    plt.title("Training vs Validation Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.savefig("accuracy_curve.png", dpi=300)

    plt.show()

# ==========================================================
# Loss Curve
# ==========================================================

if len(loss) > 0:

    plt.figure(figsize=(10,6))

    plt.plot(loss, label="Training Loss", linewidth=2)

    plt.plot(val_loss, label="Validation Loss", linewidth=2)

    plt.title("Training vs Validation Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.savefig("loss_curve.png", dpi=300)

    plt.show()

    print("✓ Training Curves Saved")

# ==========================================================
# Predictions
# ==========================================================

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(images, verbose=0)

    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())

    y_pred.extend(predicted_labels)

y_true = np.array(y_true)

y_pred = np.array(y_pred)

# ==========================================================
# Classification Report
# ==========================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(

    classification_report(

        y_true,

        y_pred,

        target_names=class_names

    )

)

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 6))

plt.imshow(cm, interpolation="nearest", cmap="Blues")

plt.title("Confusion Matrix", fontsize=16)

plt.colorbar()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names, rotation=45)

plt.yticks(tick_marks, class_names)

# Write values inside each cell
threshold = cm.max() / 2

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black",
            fontsize=12,
            fontweight="bold"
        )

plt.ylabel("Actual Label", fontsize=12)

plt.xlabel("Predicted Label", fontsize=12)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("✓ Confusion Matrix Saved")

# ==========================================================
# Store Test Images
# ==========================================================

all_images = []
all_labels = []

for images, labels in test_dataset:

    all_images.append(images.numpy())

    all_labels.append(labels.numpy())

all_images = np.concatenate(all_images)

all_labels = np.concatenate(all_labels)

predictions = model.predict(all_images, verbose=0)

predicted_classes = np.argmax(predictions, axis=1)

# ==========================================================
# Correct Predictions
# ==========================================================

correct_indices = np.where(predicted_classes == all_labels)[0][:10]

plt.figure(figsize=(15,6))

for i, idx in enumerate(correct_indices):

    plt.subplot(2,5,i+1)

    plt.imshow(all_images[idx].astype("uint8"))

    plt.title(

        f"P:{class_names[predicted_classes[idx]]}\nA:{class_names[all_labels[idx]]}"

    )

    plt.axis("off")

plt.tight_layout()

plt.savefig(

    "correct_predictions.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

print("✓ Correct Predictions Saved")

# ==========================================================
# Incorrect Predictions
# ==========================================================

incorrect_indices = np.where(predicted_classes != all_labels)[0][:10]

plt.figure(figsize=(15,6))

for i, idx in enumerate(incorrect_indices):

    plt.subplot(2,5,i+1)

    plt.imshow(all_images[idx].astype("uint8"))

    plt.title(

        f"P:{class_names[predicted_classes[idx]]}\nA:{class_names[all_labels[idx]]}",

        color="red"

    )

    plt.axis("off")

plt.tight_layout()

plt.savefig(

    "incorrect_predictions.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

print("✓ Incorrect Predictions Saved")