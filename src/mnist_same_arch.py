import os
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
import numpy as np


"""
Here we test our fooling images with more models and check if the models are equally fooled by them.
"""
model_folder = os.path.join(os.getcwd(), 'saved_models')
data_folder = os.path.join(os.getcwd(), 'saved_datasets')

model_1_path = "mnist_cnn.h5"
model_1_path = os.path.join(model_folder, model_1_path)
model_2_path = "mnist_cnn_big.h5"
model_2_path = os.path.join(model_folder, model_2_path)
data_path = "mnist.npz"
data_path = os.path.join(data_folder, data_path)

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

if not os.path.isdir(data_folder):
    os.mkdir(data_folder)

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data(path=data_path)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train[x_train < 127] = 0
x_train[x_train >= 127] = 1
x_test[x_test < 127] = 0
x_test[x_test >= 127] = 1


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_orig_train = np.copy(y_train)
y_orig_test = np.copy(y_test)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

if not os.path.isdir(model_folder):
    os.mkdir(model_folder)

try:
    model_1 = load_model(model_1_path)
except OSError:
    model_1 = Sequential()
    model_1.add(
        Conv2D(
            32, kernel_size=(3, 3),
            activation='relu',
            input_shape=input_shape
        )
    )
    model_1.add(Conv2D(64, (3, 3), activation='relu'))
    model_1.add(MaxPooling2D(pool_size=(2, 2)))
    model_1.add(Dropout(0.25))
    model_1.add(Flatten())
    model_1.add(Dense(128, activation='relu'))
    model_1.add(Dropout(0.5))
    model_1.add(Dense(num_classes, activation='softmax'))

    model_1.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy']
    )

    model_1.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test)
    )
    score = model_1.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model_1.save(model_1_path)


try:
    model_2 = load_model(model_2_path)
except OSError:
    model_2 = Sequential()
    model_2.add(
        Conv2D(
            64, kernel_size=(3, 3),
            activation='relu',
            input_shape=input_shape
        )
    )
    model_2.add(Conv2D(128, (3, 3), activation='relu'))
    model_2.add(MaxPooling2D(pool_size=(2, 2)))
    model_2.add(Dropout(0.25))
    model_2.add(Flatten())
    model_2.add(Dense(256, activation='relu'))
    model_2.add(Dropout(0.5))
    model_2.add(Dense(num_classes, activation='softmax'))

    model_2.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy']
    )

    model_2.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test)
    )
    score = model_2.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model_2.save(model_2_path)


fooling_images = []


fooling_path = os.path.join(os.path.dirname(os.getcwd()), 'MNIST_Fooling')
for root, dirs, filenames in os.walk(fooling_path):
    for f in filenames:
        temp = np.load(os.path.join(fooling_path, f))
        temp = temp.reshape((10, 28, 28, 1))

        # Add it to the new array
        for x in temp:
            fooling_images.append(x)

fooling_images = np.array(fooling_images)

pred1 = model_1.predict(fooling_images)
pred2 = model_2.predict(fooling_images)

errors_1 = 0

for i, x in enumerate(pred1):
    if x[i % 10] < 0.9:
        errors_1 += 1

errors_2 = 0

for i, x in enumerate(pred2):
    if x[i % 10] < 0.9:
        errors_2 += 1

print("Error by the original model:", errors_1)
print("Error by the bigger model:", errors_2)
