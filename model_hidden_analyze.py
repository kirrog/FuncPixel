import time

import numpy as np
from tensorflow import keras
import tensorflow as tf

from data_work.data_loader import load_x_model_vectors, load_grad_local_maximums_and_vals
from utils.model_redact import create_hidden_output_layer, model_redact_with_local_max


# def create_hidden_output_layer(model):
#     extractor = keras.Model(inputs=model.inputs,
#                             outputs=[layer.output for layer in model.layers])
#     return extractor


def change_weights_of_hidden_layer(ouput, weights_matrix, weights_shift):
    avg = np.average(ouput)
    for i in range(ouput.shape[1]):
        if ouput[0, i] > avg:
            weights_shift[i] = 0
            weights_matrix[:, i] = 0


data_x = load_x_model_vectors()
data_m_and_v = load_grad_local_maximums_and_vals()

x_pict = np.reshape(data, newshape=(data.shape[0], data.shape[1], 28, 28, 1))

tf.config.set_visible_devices([], 'GPU')
model = tf.keras.models.load_model('data/models/epochs/ep008-loss0.023-accuracy0.992_20211127-220304.h5')
model.summary()

# test = np.zeros((1, 28, 28, 1))
# res = model(test)
# print(res)
# start = time.time()
# model_redact_with_local_max(model, x_0_pict[:10000], x_0_pict[10000:])
# stop = time.time()
# print(str(stop - start))
# res = model(test)
# print(res)

# res = model_hidden(test)
# print(res[-1])
# a = model.get_weights()
# print(len(res))
# for i in range(2, len(res) - 1):
#     print(res[i].shape)
#     change_weights_of_hidden_layer(res[i], a[(i - 2) * 2], a[(i - 2) * 2 + 1])
# model.set_weights(a)
# res = model_hidden(test)
# print(res[-1])
# model.weights[6][]
# res = model(test)
# print(res)
