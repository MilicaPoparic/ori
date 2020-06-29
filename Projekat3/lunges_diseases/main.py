import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import shutil
import os
import csv
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pydot
import pydotplus
from tensorflow.keras.layers import Dense, Flatten, Conv2D



def read_data():
    result = {}
    with open("chest_xray_data_set/metadata/chest_xray_metadata.csv", 'r') as file:
        csv_file = csv.DictReader(file)
        for row in csv_file:
            key = row.pop('X_ray_image_name')
            # print(key)
            result[key] = row
    # print(result['IM-0128-0001.jpeg'].get('Label')) ovo je da li je pneumonia ili normal
    return result


def classify_data(result):
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

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 3:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


if __name__ == "__main__":

    #print("CITAM PODATKE")
    #result = read_data()
    #print("PREMENSTAM SLIKE")
    #classify_data(result)

    #print("OBRADJENO")
    # Any results you write to the current directory are saved as output.

    image_size = (180, 180)
    batch_size = 32

    print("TRENIRANJE")
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "classified",
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    print("VALIDACIJA")
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "classified",
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    # print("VIZUELIZACIJA")
    # #visualization
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(int(labels[i]))
    #         plt.axis("off")
    # plt.show()

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
    augmented_train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y))
    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)

    model = make_model(input_shape=image_size + (3,), num_classes=2)
    keras.utils.plot_model(model, show_shapes=True)

    epochs = 50

    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
    )
    img = keras.preprocessing.image.load_img(
        "classified/bacteria/person1_bacteria_1.jpeg", target_size=image_size
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = predictions[0]
    print(
        "This image is %.2f percent bacteria and %.2f percent other."
        % (100 * (1 - score), 100 * score)
    )





    # training_dir = "classified"
    # training_generator = ImageDataGenerator(rescale=1 / 255, featurewise_center=False,
    #                                         # set input mean to 0 over the dataset
    #                                         samplewise_center=False,  # set each sample mean to 0
    #                                         featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #                                         samplewise_std_normalization=False,  # divide each input by its std
    #                                         zca_whitening=False,  # apply ZCA whitening
    #                                         rotation_range=30,
    #                                         # randomly rotate images in the range (degrees, 0 to 180)
    #                                         zoom_range=0.2,  # Randomly zoom image
    #                                         width_shift_range=0.1,
    #                                         # randomly shift images horizontally (fraction of total width)
    #                                         height_shift_range=0.1,
    #                                         # randomly shift images vertically (fraction of total height)
    #                                         horizontal_flip=False,  # randomly flip images
    #                                         vertical_flip=False)
    # train_generator = training_generator.flow_from_directory(training_dir, target_size=(200, 200), batch_size=4,
    #                                                          class_mode='binary')
    #
    # validation_dir = "classified"
    # validation_generator = ImageDataGenerator(rescale=1 / 255)
    # val_generator = validation_generator.flow_from_directory(validation_dir, target_size=(200, 200), batch_size=4,
    #                                                          class_mode='binary')
    #
    # test_dir = "classified"
    # test_generator = ImageDataGenerator(rescale=1 / 255)
    # test_generator = test_generator.flow_from_directory(test_dir, target_size=(200, 200), batch_size=16,
    #                                                     class_mode='binary')
    #
    # # Developing
    # # a
    # # Convolutional
    # # Neural
    # # Network
    # # for the classification task
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Conv2D(32, (3, 3), input_shape=(200, 200, 3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D(2, 2),
    #
    #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D(2, 2),
    #     tf.keras.layers.Dropout(0.2),
    #
    #     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D(2, 2),
    #     tf.keras.layers.Dropout(0.2),
    #
    #     tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D(2, 2),
    #
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(256, activation='relu'),
    #     tf.keras.layers.Dense(1, activation='sigmoid')
    #
    # ])
    #
    # # Using
    # # the
    # # Adam
    # # optmizer
    # # with the learning rate of 0.001
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
    # history = model.fit_generator(train_generator, validation_data=val_generator, epochs=30, verbose=1)
    #
    # # Plotting
    # # the
    # # training and validation
    # # accuracy
    # # with respect to the number of epochs
    # # % matplotlib
    # # inline
    #
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    #
    # epochs = range(len(acc))
    #
    # plt.plot(epochs, acc, 'r', label='Training accuracy')
    # plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    # plt.title('Training and validation accuracy')
    # plt.legend(loc=0)
    # plt.figure()
    #
    # # Now, we will test our model on the test data
    # print("Loss of the model is - ", model.evaluate(test_generator)[0] * 100, "%")
    # print("Accuracy of the model is - ", model.evaluate(test_generator)[1] * 100, "%")
    #
    # # Saving the model weights for future use
    # model.save_weights("model.h5")
    # print("Saved model to disk")
    #
    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)
