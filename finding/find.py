import tensorflow as tf

coef = 1.0


def calc_model_point_grad(model, point, pos):
    with tf.GradientTape() as tape:
        y = model(point)
        x = tf.Variable((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        x[pos] = 1.0
        y = y * x
    dy_dx = tape.gradient(y, point)
    return dy_dx, y


def make_move(bef, aft):  # maximize probability of class, so use +
    res = bef + aft * coef
    res[res > 1.0] = 1.0
    res[res < 0.0] = 0.0
    return res


def find_supr(model, point, pos):
    p = point
    d, r = calc_model_point_grad(model, p, pos)
    for i in range(1000):
        d, r_p = calc_model_point_grad(model, p, pos)
        p = make_move(p, d)
    return p, d, r
