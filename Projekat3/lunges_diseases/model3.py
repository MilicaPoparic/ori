
"""
    Skripta za poretanje na google colab
"""

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import os
import csv
import shutil


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

labels = ['bacteria', 'normal', 'virus']
img_size = 150

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

train = train_datagen.flow_from_directory(
    'classified',
    target_size=(150, 150),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    subset='training')

val = train_datagen.flow_from_directory(
    'classified',
    target_size=(150, 150),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    subset='validation')

model = tf.keras.models.Sequential()
model.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.9))
model.add(Dense(units=3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(train, epochs=30, validation_data=val)

test_generator = train_datagen.flow_from_directory(
    'test',
    target_size=(150, 150),
    batch_size=30,
    color_mode='rgb',
    class_mode='categorical')

print("Loss of the model is - ", model.evaluate(test_generator)[0] * 100, "%")
print("Accuracy of the model is - ", model.evaluate(test_generator)[1] * 100, "%")