import numpy
import numpy as np
import tensorflow as tf

learning_rate = 0.1
num_of_classes = 10
point_size = 28


def find_supr(model, point, pos):
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    p_curr = tf.Variable(point)
    p_prev = tf.Variable(point)
    loss = lambda: -model(p_curr)[0, pos]
    for i in range(int(2 / learning_rate)):
        opt.minimize(loss, [p_curr])
        p_diff = p_prev - p_curr
        if p_diff.numpy().max() < 0.05:
            break
        p_prev = p_curr
    r_res = -loss()
    return r_res, np.reshape(p_curr.numpy(), (point_size, point_size))


def calc_from_point(model, point):
    res_in_point = model(point).numpy()
    res_from_point = np.zeros(num_of_classes)
    result_points = np.zeros((num_of_classes, point_size, point_size))
    for i in range(num_of_classes):
        res_from_point[i], result_points[i] = find_supr(model, point, i)
    return res_in_point, res_from_point, result_points
