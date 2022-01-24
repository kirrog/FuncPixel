import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('data/models/epochs/ep008-loss0.023-accuracy0.992_20211127-220304.h5')
model.summary()

x = tf.Variable(np.zeros((1, 28, 28, 1)))

with tf.GradientTape() as tape:
    y = model(x)
    y = y * tf.Variable((0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

dy_dx = tape.gradient(y, x)

print(dy_dx.numpy().max())
