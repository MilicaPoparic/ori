
"""
    Skripta za poretanje na google colab
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import os
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
        # print(key)
        result[key] = row
    # print(result['IM-0128-0001.jpeg'].get('Label')) ovo je da li je pneumonia ili normal

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

shape = (180, 180)
height, width = shape

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    rescale=1. / 255,
    horizontal_flip=True,
    validation_split=0.2)

training_set = train_datagen.flow_from_directory(
    'classified',
    target_size=(height, width),
    classes=('bacteria', 'normal', 'virus'),
    color_mode='rgb',
    batch_size=30,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    'classified',
    target_size=(height, width),
    classes=('bacteria', 'normal', 'virus'),
    color_mode='rgb',
    batch_size=30,
    class_mode='categorical',
    subset='validation')

conv_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(180, 180, 3))
x = tf.keras.layers.Flatten()(conv_model.output)
x = tf.keras.layers.Dense(100, activation='relu')(x)
x = tf.keras.layers.Dense(100, activation='relu')(x)
x = tf.keras.layers.Dense(100, activation='relu')(x)

predictions = tf.keras.layers.Dense(3, activation='softmax')(x)

full_model = tf.keras.models.Model(inputs=conv_model.input, outputs=predictions)
# full_model.summary()

for layer in conv_model.layers:
    layer.trainable = False


full_model.compile(loss='categorical_crossentropy',
                   optimizer=tf.keras.optimizers.Adamx(1e-5),
                   metrics=['acc'])
history = full_model.fit_generator(
    training_set,
    validation_data=validation_generator,
    workers=10,
    epochs=30
)

test_generator = train_datagen.flow_from_directory(
    'test',
    target_size=(height, width),
    batch_size=30,
    color_mode='rgb',
    class_mode='categorical')

print("Loss of the model is - ", full_model.evaluate(test_generator)[0] * 100, "%")
print("Accuracy of the model is - ", full_model.evaluate(test_generator)[1] * 100, "%")




