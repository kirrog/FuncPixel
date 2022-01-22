import tensorflow as tf
from tensorflow import keras

from utils.data_transformer import calculate_by_layers
from utils.hidden_layer import HiddenLayerDense

model = tf.keras.models.load_model('data/models/epochs/ep008-loss0.023-accuracy0.992_20211127-220304.h5')
h_next = None
h_prev = None
for i in range(2): #int(len(model.weights) / 2)
    h_next = HiddenLayerDense('sigm', model.weights[i * 2].numpy(), model.weights[i * 2 + 1].numpy())
    print("Layer created")
    if h_prev:
        h_next.link_layer_as_prev(h_prev)
        print("Layer linked")
    h_prev = h_next
model = h_next

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

accuracy = 0
print("Start calc")
res = calculate_by_layers(model, x_test[0])
print(res)
