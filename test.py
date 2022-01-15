import tensorflow as tf
from tensorflow import keras

from utils.data_transformer import calculate_by_layers
from utils.hidden_layer import HiddenLayerDense

model = tf.keras.models.load_model('data/models/epochs/ep008-loss0.023-accuracy0.992_20211127-220304.h5')
model = HiddenLayerDense('sigm', model.weights[0].numpy(), model.weights[1].numpy())

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

accuracy = 0

res = calculate_by_layers(model, x_test[0])
print(res)
