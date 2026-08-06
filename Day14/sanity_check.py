import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("final_test_model.keras")

dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(224,224),
    batch_size=1,
    shuffle=False
)

class_names = dataset.class_names

for images, labels in dataset.take(20):

    prediction = model.predict(images, verbose=0)

    pred = np.argmax(prediction)

    print(
        "Actual:",
        class_names[int(labels[0])],
        "| Predicted:",
        class_names[pred],
        "| Raw:",
        prediction
    )