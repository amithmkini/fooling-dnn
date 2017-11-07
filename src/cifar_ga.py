import os
import keras
import time
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras.models import load_model
import numpy as np


def mutation_manager(data, indices):
    cols = data.shape[1]

    for x in indices:
        row_no = int(x / cols)
        col_no = x % cols
        data[row_no, col_no] = np.random.random()

    return data


# Not using Roulette select as it reduces the performance
def roulette_select(population, fitnesses, num):
    total_fitness = float(sum(fitnesses))
    rel_fitness = [f/total_fitness for f in fitnesses]
    # Generate probability intervals for each individual
    probs = [sum(rel_fitness[:i+1]) for i in range(len(rel_fitness))]
    # Draw new population
    new_population = []
    for n in range(num):
        r = np.random.random()
        for (i, individual) in enumerate(population):
            if r <= probs[i]:
                new_population.append((individual, fitnesses[i]))
                break
    return new_population


def crossover_manager(genome1, genome2):
    random_index_to_cut = int(np.random.random() * len(genome1))
    new1 = list(genome1[:random_index_to_cut]) + list(genome2[random_index_to_cut:])
    new2 = list(genome2[:random_index_to_cut]) + list(genome1[random_index_to_cut:])

    return np.asarray(new1), np.asarray(new2)


def genetic_op(data, required_val, mutation_rate, crossover_rate):
    # Crossover operation
    no_of_crossovers = int(data.shape[0] * crossover_rate)
    indices_to_crossover = np.random.choice(data.shape[0], no_of_crossovers, replace=False)
    data_to_be_crossover = data[indices_to_crossover]

    for x in range(0, len(indices_to_crossover) - 1, 2):
        genome1 = data_to_be_crossover[x]
        genome2 = data_to_be_crossover[x + 1]
        ind1 = indices_to_crossover[x]
        ind2 = indices_to_crossover[x + 1]
        new1, new2 = crossover_manager(genome1, genome2)
        data[ind1] = new1
        data[ind2] = new2

    # Now for the mutation operation
    no_of_mutations = int(data.shape[0] * data.shape[1] * mutation_rate)
    indices_to_mutate = np.random.choice(data.shape[0] * data.shape[1], no_of_mutations, replace=False)
    data = mutation_manager(data, indices_to_mutate)

    # Finally fitness
    y = data.reshape(data.shape[0], 32, 32, 3)
    r = model.predict(y)
    fitness = r[:, required_val]

    return data, fitness


def add_noise(img, _):
    random_noise = np.random.random((32, 32, 3)) * np.random.random((32, 32, 3)) * \
                   np.random.random((32, 32, 3)) * np.random.random((32, 32, 3))
    return np.clip(img + random_noise, 0, 1)


def generate_noisy_images(new_num):
    crossover_rate = 0.05
    mutation_rate = 0.1
    population_size = 100
    generations = 200

    promotion_rate = 0.1
    multiplier = int(1 / promotion_rate)

    fit_num = []

    # Now for each number we have to create noisy initial population.
    try:
        for numb in range(10):
            required_val = numb
            print("Now evaluating for number", required_val)
            initial_popl = new_num[numb].reshape(
                new_num[numb].shape[0],
                new_num[numb].shape[1] * new_num[numb].shape[2] * new_num[numb].shape[3]
            )
            initial_popl = np.tile(initial_popl, (population_size // new_num[numb].shape[0], 1))
            prev_popl = [[0, 0]]

            for gen in range(generations):
                change = False
                while not change:
                    data, fitness = genetic_op(initial_popl, required_val, crossover_rate, mutation_rate)
                    data_with_fitness = zip(data, fitness)
                    data_with_fitness = sorted(data_with_fitness, key=lambda x: x[1], reverse=True)

                    # Now select the promoted data for the next round
                    promotion = data_with_fitness[:int(population_size * promotion_rate)]
                    # promotion = roulette_select(list(data), fitness, int(population_size * promotion_rate))

                    if prev_popl[0][1] > promotion[0][1]:
                        initial_popl = np.asarray([x[0] for x in prev_popl])
                    else:
                        only_data = [x[0] for x in promotion]
                        initial_popl = np.asarray(only_data * multiplier)
                        prev_popl = promotion * multiplier
                        change = True

                print("Max fitness for gen {} is {}".format(gen, promotion[0][1]))
                if promotion[0][1] > .95 or (gen > 50 and promotion[0][1] > .80):
                    print("Adding the image to the new array...")
                    fit_num.append(promotion[0][0])
                    break
        return fit_num

    except KeyboardInterrupt:
        print("GA stopped!")
        return fit_num


batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = False
num_predictions = 20

model_folder = os.path.join(os.getcwd(), 'saved_models')
data_folder = os.path.join(os.getcwd(), 'saved_datasets')
model_path = "keras_cifar10_trained_model.h5"
model_path = os.path.join(model_folder, model_path)

if not os.path.isdir(data_folder):
    os.mkdir(data_folder)
if not os.path.isdir(model_folder):
    os.mkdir(model_folder)


# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_orig_train = np.copy(y_train)
y_orig_test = np.copy(y_test)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


try:
    model = load_model(model_path)
except OSError or ValueError:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            workers=4)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    model.save(model_path)

"""
Now to generate the fooling images, we can start from the scratch.
But doing so will take a lot of time. So we decided to start with
a population of noisy MNIST dataset
"""
# First we will split the data
num = []

for x in range(10):
    indices = np.argwhere(y_orig_train == x)[:, 0].reshape((-1,))
    num.append(x_train[indices])

num = np.asarray(num)

# Now that the data is sorted, we take the first 20 of each of the
# dataset and add noise to each of them.

chosen_ones = []
for j in range(10):
    numbers = num[j]
    r = model.predict(numbers)
    new = [numbers[i] for i in range(len(r)) if r[i][j] > .8]
    new = np.array(new)
    chosen_ones.append(new)

chosen_ones = np.array(chosen_ones)

new_num = []

for x in range(10):
    numbers = chosen_ones[x][:20]
    final_image = np.apply_over_axes(add_noise, numbers, [0])
    new_num.append(final_image)
new_num = np.asarray(new_num)

final = [x[:20] for x in new_num]
new_num = np.array(final)


all_num = []
for x in range(10):
    fit_num = generate_noisy_images(new_num)
    path = os.path.join(data_folder, str(int(time.time())) + ".npy")
    np.save(path, np.array(fit_num))