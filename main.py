import tensorflow as tf

from utils.hidden_layer import HiddenLayerDense

model = tf.keras.models.load_model('data/models/epochs/ep008-loss0.023-accuracy0.992_20211127-220304.h5')
h_prev = None
h_next = None
for i in range(int(len(model.weights)/2)):
    h_next = HiddenLayerDense('sigm', model.weights[i*2].numpy(), model.weights[i*2+1].numpy())
    print("Layer created")
    if h_prev:
        h_next.link_layer_as_prev(h_prev)
        print("Layer linked")
    h_prev = h_next

print(h_next)
