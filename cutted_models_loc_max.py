import sys

import numpy as np
import tensorflow as tf
import timeit
from pathlib import Path

from finding.find import find_supr
from finding.generator import create_generator_points

ouput_path = "/media/kirrog/data/data/"
dim, side_size = 7, 28
points_shape = (1, side_size, side_size, 1)
num_of_classes = 10
iters = 128
learning_rate = 0.1


def calc_from_point_pos(model, point, pos):
    res_in_point = model(point).numpy()
    res_from_point, result_points = find_supr(model, point, pos)
    return res_in_point, res_from_point, result_points


for k in range(num_of_classes):
    g, num_of_points = create_generator_points(7, 28)
    model = tf.keras.models.load_model('data/models/cutted/{k:02d}.h5'.format(k=k))
    print("Model {k:02d} loaded".format(k=k))
    start_all = timeit.default_timer()
    for i in range(int(num_of_points // iters)):
        start = timeit.default_timer()
        test = []
        for j in range(iters):
            test.append(next(g))
        res_in_point = np.zeros((iters, num_of_classes))
        res_from_point = np.zeros((iters))
        result_points = np.zeros((iters, side_size, side_size))
        for point, number in test:
            number = number % iters
            x = tf.Variable(point)
            x = tf.reshape(x, points_shape)
            res_in_point[number], res_from_point[number], result_points[number] = calc_from_point_pos(model, x, k)
        class_dir = "class_{k:02d}/".format(k=k)
        Path(ouput_path + class_dir + "point_answ/").mkdir(parents=True, exist_ok=True)
        Path(ouput_path + class_dir + "point_grad_answ/").mkdir(parents=True, exist_ok=True)
        Path(ouput_path + class_dir + "points_grad/").mkdir(parents=True, exist_ok=True)
        np.save(ouput_path + class_dir + "point_answ/{i:05d}.npy".format(i=i), res_in_point)
        np.save(ouput_path + class_dir + "point_grad_answ/{i:05d}.npy".format(i=i), res_from_point)
        np.save(ouput_path + class_dir + "points_grad/{i:05d}.npy".format(i=i), result_points)
        stop = timeit.default_timer()
        sys.stdout.write("\rTime spended: {t:05f} Worked: {i:05d}".format(i=i * iters, t=stop - start))
    print()
    stop_all = timeit.default_timer()
    print("Time all: " + str(stop_all - start_all))
