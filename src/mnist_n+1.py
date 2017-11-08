"""
So, to prevent adversarial images from getting high accuracies, we add a new class
with all the fooling images. Then we try again to generate fooling images from them.

"""
import os
import keras
import time
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np


def mutation_manager(data, indices):
    cols = data.shape[1]

    for x in indices:
        row_no = int(x / cols)
        col_no = x % cols

        if data[row_no, col_no] == 1:
            data[row_no, col_no] = 0
        else:
            data[row_no, col_no] = 1
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


def genetic_op(data, required_val, mutation_rate, crossover_rate, model):
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
    y = data.reshape(data.shape[0], 28, 28, 1)
    r = model.predict(y)
    fitness = r[:, required_val]

    return data, fitness


def add_noise(img, _):
    random_noise = np.random.randint(0, 2, (28, 28, 1)) * np.random.randint(0, 2, (28, 28, 1)) \
                   * np.random.randint(0, 2, (28, 28, 1))
    return np.clip(img + random_noise, 0, 1)


def generate_noisy_images(new_num, model):
    crossover_rate = 0.3
    mutation_rate = 0.2
    population_size = 50
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
                new_num[numb].shape[1] * new_num[numb].shape[2]
            )
            initial_popl = np.tile(initial_popl, (population_size // new_num[numb].shape[0], 1))
            prev_popl = [[0, 0]]

            for gen in range(generations):
                change = False
                while not change:
                    data, fitness = genetic_op(initial_popl, required_val, crossover_rate, mutation_rate, model)
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
                if promotion[0][1] > .99:
                    print("Adding the image to the new array...")
                    fit_num.append(promotion[0][0])
                    break
        return fit_num

    except KeyboardInterrupt:
        print("GA stopped!")
        return None


model_folder = os.path.join(os.getcwd(), 'saved_models')
data_folder = os.path.join(os.getcwd(), 'saved_datasets')

model_path = "mnist_cnn_n1.h5"
model_path = os.path.join(model_folder, model_path)
data_path = "mnist.npz"
data_path = os.path.join(data_folder, data_path)


batch_size = 128
num_classes = 11
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

# Converting the image to B/W
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
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# We need to add a 11th class to the output

new_y_train = np.zeros((y_train.shape[0], 11))
new_y_test = np.zeros((y_test.shape[0], 11))

new_y_train[:, :-1] = y_train
new_y_test[:, :-1] = y_test


if not os.path.isdir(model_folder):
    os.mkdir(model_folder)

try:
    model = load_model(model_path)
except OSError or ValueError:
    # Now add the (n+1)th y_test
    fooling_path = os.path.join(os.path.dirname(os.getcwd()), 'MNIST_Fooling')
    for root, dirs, filenames in os.walk(fooling_path):
        for f in filenames:
            temp = np.load(os.path.join(fooling_path, f))
            temp_train = temp.reshape((10, 28, 28, 1))
            temp_y_train = np.zeros((10, 11))
            temp_y_train[:, -1] = 1

            # Add it to the new array

            x_train = np.append(x_train, temp_train, axis=0)
            new_y_train = np.append(new_y_train, temp_y_train, axis=0)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(11, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, new_y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, new_y_test))
    score = model.evaluate(x_test, new_y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save(model_path)

"""
Now to generate the fooling images, we can start from the scratch.
But doing so will take a lot of time. So we decided to start with
a population of noisy MNIST dataset
"""
# First we will split the data
num = []

for x in range(10):
    indices = np.argwhere(y_orig_train == x).reshape((-1,))
    num.append(x_train[indices])

num = np.asarray(num)

# Now that the data is sorted, we take the first 20 of each of the
# dataset and add noise to each of them.

new_num = []

for x in range(10):
    numbers = num[x][:20]
    final_image = np.apply_over_axes(add_noise, numbers, [0])
    new_num.append(final_image)

new_num = np.asarray(new_num)

all_num = []
for x in range(10):
    fit_num = generate_noisy_images(new_num, model)
    path = os.path.join(data_folder, str(int(time.time())) + ".npy")
    np.save(path, np.array(fit_num))
