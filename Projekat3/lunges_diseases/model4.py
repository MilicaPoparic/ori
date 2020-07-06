
"""
    Skripta za poretanje na google colab
"""

!pip install tf - nightly
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import os
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os


def read_test_data():
    result = {}
    with open("chest-xray-dataset-test/chest_xray_test_dataset.csv", 'r') as file:
        csv_file = csv.DictReader(file)
        for row in csv_file:
            key = row.pop('X_ray_image_name')
            result[key] = row
    classify_test_data(result)


def classify_test_data(result):
    bacteria = []
    normal = []
    virus = []
    if not os.path.exists('test'):
        os.makedirs('test')
    if not os.path.exists('test/normal'):
        os.makedirs('test/normal')
    if not os.path.exists('test/virus'):
        os.makedirs('test/virus')
    if not os.path.exists('test/bacteria'):
        os.makedirs('test/bacteria')

    for file in os.listdir('chest-xray-dataset-test/test'):
        if not os.path.isfile('chest-xray-dataset-test/test/' + file):
            continue
        # proveri kakva su pluca
        if not result.keys().__contains__(file):
            continue

        if result[file].get('Label') == "Normal":
            normal.append(file)
        elif result[file].get('Label_1_Virus_category') == "bacteria":
            bacteria.append(file)
        elif result[file].get('Label_1_Virus_category') == "Virus":
            virus.append(file)

    for f in normal:
        source_folder = "chest-xray-dataset-test/test/" + f
        dest_folder = "test/normal"
        shutil.move(source_folder, dest_folder)
    for f in bacteria:
        source_folder = "chest-xray-dataset-test/test/" + f
        dest_folder = "test/bacteria"
        shutil.move(source_folder, dest_folder)
    for f in virus:
        source_folder = "chest-xray-dataset-test/test/" + f
        dest_folder = "test/virus"
        shutil.move(source_folder, dest_folder)


read_test_data()
result = {}
with open("chest_xray_data_set/metadata/chest_xray_metadata.csv", 'r') as file:
    csv_file = csv.DictReader(file)
    for row in csv_file:
        key = row.pop('X_ray_image_name')
        result[key] = row

    # u listama se nalaze imena fajlova
bacteria = []
normal = []
virus = []
if not os.path.exists('classified'):
    os.makedirs('classified')
if not os.path.exists('classified/normal'):
    os.makedirs('classified/normal')
if not os.path.exists('classified/virus'):
    os.makedirs('classified/virus')
if not os.path.exists('classified/bacteria'):
    os.makedirs('classified/bacteria')

for file in os.listdir('chest_xray_data_set'):
    if not os.path.isfile('chest_xray_data_set/' + file):
        continue
    if not result.keys().__contains__(file):
        continue

    if result[file].get('Label') == "Normal":
        normal.append(file)
    elif result[file].get('Label_1_Virus_category') == "bacteria":
        bacteria.append(file)
    elif result[file].get('Label_1_Virus_category') == "Virus":
        virus.append(file)

for f in normal:
    source_folder = "chest_xray_data_set/" + f
    dest_folder = "classified/normal"
    shutil.move(source_folder, dest_folder)
for f in bacteria:
    source_folder = "chest_xray_data_set/" + f
    dest_folder = "classified/bacteria"
    shutil.move(source_folder, dest_folder)
for f in virus:
    source_folder = "chest_xray_data_set/" + f
    dest_folder = "classified/virus"
    shutil.move(source_folder, dest_folder)

image_size = (180, 180)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "classified",
    validation_split=0.2,
    subset="training",
    seed=700,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "classified",
    validation_split=0.2,
    subset="validation",
    seed=100,
    image_size=image_size,
    batch_size=batch_size,
)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        # labels[i] = tf.reshape(labels, [3, 1])
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")


data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)


plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units=num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=3)
keras.utils.plot_model(model, show_shapes=True)

epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)

img = keras.preprocessing.image.load_img(
    "test/virus/person1622_virus_2810.jpeg", target_size=image_size
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = predictions[0]

print(
        "This image is %.2f percent virus, %.2f percent bacteria and %.2f percent normal."
        % (100 * (1 - score[2]), 100 * (1 - score[0]), 100 * (1 - score[1]))
)

test_generator = tf.keras.preprocessing.image_dataset_from_directory(
    "test",
    image_size=image_size,
    batch_size=batch_size,
)

print("Loss of the model is - ", model.evaluate(test_generator)[0] * 100, "%")
print("Accuracy of the model is - ", model.evaluate(test_generator)[1] * 100, "%")

