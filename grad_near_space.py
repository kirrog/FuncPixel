import sys
import timeit

import numpy as np
import tensorflow as tf

from finding.find import num_of_classes
from data_work.data_loader import pict_size, load_x_model_vectors

learning_rate = 0.1


def calc_from_point_nearest_grad(model, point, class_type):
    res = np.zeros((2, 2))
    p_p = np.copy(point)
    p_m = np.copy(point)
    p_p -= learning_rate
    p_m += learning_rate
    with  tf.GradientTape() as tape:
        p_p_v = tf.Variable(np.reshape(p_p, newshape=(1, pict_size, pict_size, 1)), shape=(1, pict_size, pict_size, 1))
        loss = lambda: -model(p_p_v)[0, class_type]
        p_res = loss()
        res[0, 0] = np.sum(np.absolute((tape.gradient(p_res, p_p_v).numpy())))
        res[1, 0] = -p_res.numpy()
    with  tf.GradientTape() as tape:
        p_m_v = tf.Variable(np.reshape(p_m, newshape=(1, pict_size, pict_size, 1)), shape=(1, pict_size, pict_size, 1))
        loss = lambda: -model(p_m_v)[0, class_type]
        m_res = loss()
        res[0, 1] = np.sum(np.absolute((tape.gradient(m_res, p_m_v).numpy())))
        res[1, 1] = -m_res.numpy()
    return res


def search_grad_nearest_space_points(model, x_model_vectors, batch_size=256):
    size = int(x_model_vectors.shape[0] / batch_size)
    print("Work: " + str(size))
    print("Batch: " + str(batch_size))
    time_start_full = timeit.default_timer()
    for i in range(size):
        res = np.zeros((batch_size, num_of_classes, 2, 2))
        for k in range(batch_size):
            time_start = timeit.default_timer()
            for j in range(x_model_vectors.shape[1]):
                res[k, j] = calc_from_point_nearest_grad(model, x_model_vectors[i * batch_size + k, j], j)
            time_stop = timeit.default_timer()
            sys.stdout.write(
                "\rBatch: {i:04d} Point: {p:06d} time all: {t_a:.05f} time: {t:.05f}".format(i=i, p=k,
                                                                                             t_a=(
                                                                                                     time_stop - time_start_full),
                                                                                             t=(
                                                                                                     time_stop - time_start)))
        np.save("/media/kirrog/data/data/points_nearest_grad_calc/{i:04d}.npy".format(i=i), res)


def grad_example(model):
    with  tf.GradientTape() as tape:
        p_p_v = tf.Variable(np.reshape(np.zeros(784), newshape=(1, 28, 28, 1)), shape=(1, 28, 28, 1))
        loss = lambda: -model(p_p_v)[0, 0]
        res = tape.gradient(loss(), p_p_v).numpy()
        print(res.shape)
        print(res[0, 0])


model = tf.keras.models.load_model('data/models/epochs/ep008-loss0.023-accuracy0.992_20211127-220304.h5')
model.summary()

data = load_x_model_vectors()
print("Data loaded")

search_grad_nearest_space_points(model, data, 256)
