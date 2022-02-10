import numpy
import tensorflow as tf

learning_rate = 0.1


def find_supr(model, point, pos):
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    p_curr = tf.Variable(point)
    p_prev = tf.Variable(point)
    loss = lambda: -model(p_curr)[0, pos]
    r_fir = -loss()
    for i in range(int(2 / learning_rate)):
        opt.minimize(loss, [p_curr])
        p_diff = p_prev - p_curr
        if p_diff.numpy().max() < 0.05:
            break
        p_prev = p_curr
    r_res = -loss()
    return r_fir, r_res, p_curr
