from tensorflow import keras
import tensorflow as tf


def create_hidden_output_layer(model):
    extractor = keras.Model(inputs=model.inputs,
                            outputs=[layer.output for layer in model.layers])
    return extractor

model = tf.keras.models.load_model('data/models/epochs/ep008-loss0.023-accuracy0.992_20211127-220304.h5')
model.summary()
model_hidden = create_hidden_output_layer(model)