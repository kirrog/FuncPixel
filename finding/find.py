import numpy
import tensorflow as tf

coef = 1.0


def calc_model_point_grad(model, point, pos):
    with tf.GradientTape() as tape:
        y_r = model(point)
        x_n = numpy.zeros(10, dtype=numpy.float32)
        x_n[pos] = 1.0
        x = tf.Variable(x_n)
        y = y_r * x
    dy_dx = tape.gradient(y, point)
    return dy_dx, y_r


def make_move(bef, aft):  # maximize probability of class, so use +
    res = bef + aft * coef
    res[res > 1.0] = 1.0
    res[res < 0.0] = 0.0
    return res


def find_supr(model, point, pos):
    p = point
    d, r = calc_model_point_grad(model, p, pos)
    r_p = None
    for i in range(1000):
        d, r_p = calc_model_point_grad(model, p, pos)
        if d.numpy().max() < 0.05:
            break
        p = make_move(p, d)
    return p, r_p, r
