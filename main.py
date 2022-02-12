import timeit

import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils.data_transformer import calculate_by_layers
from utils.model import create_model_from_arrays

model = tf.keras.models.load_model('data/models/epochs/ep008-loss0.023-accuracy0.992_20211127-220304.h5')
model = create_model_from_arrays(model.weights)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

accuracy = 0
print("Start work")
start = timeit.default_timer()
res = calculate_by_layers(model, x_test[0])
stop = timeit.default_timer()
print("Time: " + str(stop - start))
print(res)
