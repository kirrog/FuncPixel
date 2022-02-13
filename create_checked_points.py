import sys

import numpy as np
import tensorflow as tf
import timeit

from finding.find import calc_from_point

model = tf.keras.models.load_model('data/models/epochs/ep008-loss0.023-accuracy0.992_20211127-220304.h5')
model.summary()
#
# x = tf.Variable(np.zeros((1, 28, 28, 1)))
# start = timeit.default_timer()
# res_in_point, res_from_point, result_points = calc_from_point(model, x)
# stop = timeit.default_timer()
# print(res_in_point.shape)
# print(res_from_point.shape)
# print(result_points.shape)
# print("Time: " + str(stop - start))
from finding.generator import create_generator_points

ouput_path = "/media/kirrog/data/data/"
dim, side_size = 7, 28
points_shape = (1, side_size, side_size, 1)
g, num_of_points = create_generator_points(7, 28)

num_of_classes = 10
iters = 128
i = 0
start_all = timeit.default_timer()
for i in range(int(num_of_points // iters)):
    start = timeit.default_timer()
    test = []
    for j in range(iters):
        test.append(next(g))
    res_in_point = np.zeros((iters, num_of_classes))
    res_from_point = np.zeros((iters, num_of_classes))
    result_points = np.zeros((iters, 10, side_size, side_size))
    for point, number in test:
        number = number % iters
        x = tf.Variable(point)
        x = tf.reshape(x, points_shape)
        res_in_point[number], res_from_point[number], result_points[number] = calc_from_point(model, x)
    np.save(ouput_path + "point_answ/{i:05d}.npy".format(i=i), res_in_point)
    np.save(ouput_path + "point_grad_answ/{i:05d}.npy".format(i=i), res_from_point)
    np.save(ouput_path + "points_grad/{i:05d}.npy".format(i=i), result_points)
    stop = timeit.default_timer()
    sys.stdout.write("\rTime spended: {t:05f} Worked: {i:05d}".format(i=i * iters, t=stop - start))
stop_all = timeit.default_timer()
print("Time all: " + str(stop_all - start_all))
