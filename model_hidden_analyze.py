import sys
import time

import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras.datasets import mnist

from data_work.data_loader import load_x_model_vectors, load_grad_local_maximums_and_vals


class picture_item:
    def __init__(self, pict, grad_and_val, model_output):
        self.picture = pict
        self.value = (grad_and_val[1, 0] + grad_and_val[1, 1]) / 2
        self.grad = (abs(grad_and_val[0, 0]) + abs(grad_and_val[0, 1])) / 2
        for i in model_output:
            i[i >= 0.8] = 1.0
            i[i < 0.8] = 0
        self.output_values = model_output


def comparator_values(x_1: picture_item, x_2: picture_item) -> int:
    if x_1.value > x_2.value:
        return 1
    elif x_1.value == x_2.value:
        return 0
    else:
        return -1


def comparator_grads(x_1: picture_item, x_2: picture_item) -> int:
    if x_1.grad > x_2.grad:
        return 1
    elif x_1.grad == x_2.grad:
        return 0
    else:
        return -1


def create_hidden_output_layer(model):
    extractor = keras.Model(inputs=model.inputs,
                            outputs=[layer.output for layer in model.layers])
    return extractor


def null_weights_of_hidden_layer(output, weights_matrix, weights_shift):
    for i in range(output.shape[0]):
        if output[i] == 1.0:
            weights_shift[i] = 0
            weights_matrix[:, i] = 0


def model_redact_with_local_max(model, trues, falses):
    size = len(trues[0].output_values)
    trues_size = len(trues)
    falses_size = len(falses)
    a = model.get_weights()
    null_number = 0
    for i in range(size):
        agreg_true = np.zeros(trues[0].output_values[i].shape)
        for j in range(len(trues)):
            agreg_true += trues[j].output_values[i]

        agreg_false = np.zeros(trues[0].output_values[i].shape)
        for j in range(len(falses)):
            agreg_false += falses[j].output_values[i]

        for j in range(agreg_true.shape[0]):
            if agreg_true[j] > 0.0 and agreg_false[j] > 0.0 and (agreg_false[j] / falses_size) > (
                    agreg_true[j] / trues_size):
                agreg_false[j] = 1.0
            elif agreg_true[j] < 0.0 and agreg_false[j] < 0.0 and (agreg_false[j] / falses_size) < (
                    agreg_true[j] / trues_size):
                agreg_false[j] = 1.0
            elif abs(agreg_true[j] / trues_size) < 0.2 and abs(agreg_false[j] / falses_size) > 0.8:
                agreg_false[j] = 1.0
            else:
                agreg_false[j] = 0.0
        null_number += np.sum(agreg_false)
        null_weights_of_hidden_layer(agreg_false, a[i * 2], a[i * 2 + 1])
    model.set_weights(a)
    return int(null_number)


def test_model(model, data_x, data_y):
    res = model.predict(data_x)
    acc = 0
    size_data = res.shape[0]
    for i in range(size_data):
        if res[i, data_y[i]] == res[i].max():
            acc += 1
    return (acc / size_data)


def binary_search_by_mnist_validation(mnist_data_x, mnist_data_y, limit, pictures_data, model,
                                      model_weights_save):
    size = len(pictures_data)
    step = 1
    position = int(size / pow(2, step))
    prev_pose = 0
    while prev_pose != position:
        null_number = model_redact_with_local_max(model, pictures_data[:position], pictures_data[position:])
        res = test_model(model, mnist_data_x, mnist_data_y)
        print(
            "Search: accuracy: {acc:.06f} val: {v:.06f} grad: {g:.06f} position: {pos:06d} step: {st:02d} nulls: {nul:06d}".format(
                acc=res, v=pictures_data[position].value, g=pictures_data[position].grad, pos=position, st=step,
                nul=null_number))
        restore = []
        for i in model_weights_save:
            restore.append(i.copy())
        model.set_weights(restore)
        step += 1
        prev_pose = position
        if res > limit:
            position -= int(size / pow(2, step))
        elif res == limit:
            break
        else:
            position += int(size / pow(2, step))


data_x = load_x_model_vectors()
data_m_and_v = load_grad_local_maximums_and_vals()
(mnist_train_x, mnist_train_y), (mnist_test_x, mnist_test_y) = mnist.load_data()

print('X_train: ' + str(mnist_train_x.shape))
print('Y_train: ' + str(mnist_train_y.shape))
print('X_test:  ' + str(mnist_test_x.shape))
print('Y_test:  ' + str(mnist_test_y.shape))

x_pict = np.reshape(data_x, newshape=(data_x.shape[0], data_x.shape[1], 28, 28, 1))
assert data_x.shape[0] == data_m_and_v.shape[0]
assert data_x.shape[1] == data_m_and_v.shape[1]

model = tf.keras.models.load_model('data/models/epochs/ep008-loss0.023-accuracy0.992_20211127-220304.h5')

default_accuracy = test_model(model, mnist_test_x, mnist_test_y)
print(default_accuracy)

hidden_model = create_hidden_output_layer(model)

hidden_data = []
for i in range(x_pict.shape[1]):
    hidden_data.append(hidden_model.predict(x_pict[:, i])[2:])
    print("Predicted class {i:02d}".format(i=i))

pictures = []
for i in range(data_m_and_v.shape[1]):
    curr_list = []
    pictures.append(curr_list)
    for j in range(data_m_and_v.shape[0]):
        d = data_m_and_v[j, i]
        pict = data_x[j, i]
        out = []
        for k in hidden_data[i]:
            out.append(k[j])
        item = picture_item(pict, d, out)
        curr_list.append(item)
    sys.stdout.write("\rPicture class {i:02d} created".format(i=i))

save_model_weights = []
for i in model.get_weights():
    save_model_weights.append(i.copy())
print("Model save created")

for i in range(len(pictures)):
    print("Class: {i:02d}".format(i=i))
    print("Value search")
    pictures[i].sort(key=lambda pict: pict.value, reverse=True)
    # binary search throw grad
    binary_search_by_mnist_validation(mnist_test_x, mnist_test_y, default_accuracy - 0.05, pictures[i], model,
                                      save_model_weights)
    print("Grad search")
    pictures[i].sort(key=lambda pict: pict.grad)
    binary_search_by_mnist_validation(mnist_test_x, mnist_test_y, default_accuracy - 0.05, pictures[i], model,
                                      save_model_weights)
    # binary search throw vals
print("Completed")

# test = np.zeros((1, 28, 28, 1))
# res = model(test)
# print(res)
# start = time.time()
# model_redact_with_local_max(model, x_0_pict[:10000], x_0_pict[10000:])
# stop = time.time()
# print(str(stop - start))
# res = model(test)
# print(res)

# res = model_hidden(test)
# print(res[-1])
# a = model.get_weights()
# print(len(res))
# for i in range(2, len(res) - 1):
#     print(res[i].shape)
#     change_weights_of_hidden_layer(res[i], a[(i - 2) * 2], a[(i - 2) * 2 + 1])
# model.set_weights(a)
# res = model_hidden(test)
# print(res[-1])
# model.weights[6][]
# res = model(test)
# print(res)
