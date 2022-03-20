import numpy as np
from tensorflow import keras
import tensorflow as tf


def create_hidden_output_layer(model):
    extractor = keras.Model(inputs=model.inputs,
                            outputs=[layer.output for layer in model.layers])
    return extractor


def change_weights_of_hidden_layer(ouput, weights_matrix, weights_shift):
    avg = np.average(ouput)
    to_del = []
    to_save = []
    for i in range(ouput.shape[1]):
        if ouput[0, i] > avg:
            weights_shift[i] = 0
            weights_matrix[:, i] = 0  # check if its right


tf.config.set_visible_devices([], 'GPU')
model = tf.keras.models.load_model('data/models/epochs/ep008-loss0.023-accuracy0.992_20211127-220304.h5')
model.summary()
model_hidden = create_hidden_output_layer(model)

test = np.zeros((1, 28, 28, 1))
res = model_hidden(test)
print(res[-1])
a = model.get_weights()
for i in range(2, len(res) - 1):
    print(res[i].shape)
    change_weights_of_hidden_layer(res[i], a[(i - 2) * 2], a[(i - 2) * 2 + 1])
model.set_weights(a)
res = model_hidden(test)
print(res[-1])
# model.weights[6][]
# res = model(test)
# print(res)
