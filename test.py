import numpy as np
import tensorflow as tf

from finding.find import find_supr

model = tf.keras.models.load_model('data/models/epochs/ep008-loss0.023-accuracy0.992_20211127-220304.h5')
model.summary()

x = tf.Variable(np.zeros((1, 28, 28, 1)))

r_fir, r_res, p_curr = find_supr(model, x, 0)

print(r_fir)
print(r_res)
