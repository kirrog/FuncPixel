import math
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
    trues_coef = math.log2(len(trues)) / len(trues)
    falses_coef = math.log2(len(falses)) / len(falses)
    a = model.get_weights()
    null_number = []
    for i in range(size):
        agreg_true = np.zeros(trues[0].output_values[i].shape)
        for j in range(len(trues)):
            agreg_true += trues[j].output_values[i]

        agreg_false = np.zeros(trues[0].output_values[i].shape)
        for j in range(len(falses)):
            agreg_false += falses[j].output_values[i]

        for j in range(agreg_true.shape[0]):
            agreg_false_val = agreg_false[j] * falses_coef
            agreg_true_val = agreg_true[j] * trues_coef
            if agreg_true[j] > 0.0 and agreg_false[j] > 0.0 and (agreg_false_val - agreg_true_val) > (
                    agreg_false_val + agreg_true_val) / 2:
                agreg_false[j] = 1.0
            elif agreg_true[j] < 0.0 and agreg_false[j] < 0.0 and (agreg_false_val - agreg_true_val) < (
                    agreg_false_val + agreg_true_val) / 2:
                agreg_false[j] = 1.0
            else:
                agreg_false[j] = 0.0
        null_number.append(int(np.sum(agreg_false)))
        null_weights_of_hidden_layer(agreg_false, a[i * 2], a[i * 2 + 1])
    model.set_weights(a)
    return null_number


def full_test_model(model, data_x, data_y):
    res = model.predict(data_x)
    class_result = [0] * (data_y.max() + 1)
    class_size = [0] * (data_y.max() + 1)
    class_acc = []
    acc = 0
    size_data = res.shape[0]
    for i in range(size_data):
        if res[i, data_y[i]] == res[i].max():
            acc += 1
            class_result[data_y[i]] += 1
        class_size[data_y[i]] += 1
    for i in range(len(class_result)):
        class_acc.append(class_result[i] / class_size[i])
    return (acc / size_data), class_acc


def apply_limits_to_class(mnist_data_x, mnist_data_y, pictures_data_true, pictures_data_false, model):
    null_number = model_redact_with_local_max(model, pictures_data_true, pictures_data_false)
    res, class_res = full_test_model(model, mnist_data_x, mnist_data_y)

    string = "Search: accuracy: {acc:.06f} classes_acc: ".format(acc=res)
    for i in range(len(class_res)):
        string += "{cl:02d} : {nul:.06} ".format(cl=i, nul=class_res[i])
    string += " nulls: "
    for i in null_number:
        string += " {n:04d}".format(n=i)
    print(string)


data_x = load_x_model_vectors()
data_m_and_v = load_grad_local_maximums_and_vals()
(mnist_train_x, mnist_train_y), (mnist_test_x, mnist_test_y) = mnist.load_data()

x_pict = np.reshape(data_x, newshape=(data_x.shape[0], data_x.shape[1], 28, 28, 1))
assert data_x.shape[0] == data_m_and_v.shape[0]
assert data_x.shape[1] == data_m_and_v.shape[1]

model = tf.keras.models.load_model('data/models/epochs/ep008-loss0.023-accuracy0.992_20211127-220304.h5')

hidden_model = create_hidden_output_layer(model)

hidden_data = []
for i in range(x_pict.shape[1]):
    hidden_data.append(hidden_model.predict(x_pict[:, i])[2:])
    sys.stdout.write("\rPredicted class {i:02d}".format(i=i))
print()

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
print()

save_model_weights = []
for i in model.get_weights():
    save_model_weights.append(i.copy())
print("Model save created")

positions = [65532, 36864, 65472, 65534, 49152, 65534, 49152, 49152, 65535, 32768]

assert len(positions) == len(pictures)

model_save_path = "./data/models/sup_cut.h5"
pictures_all_true = []
pictures_all_false = []

for i in range(len(pictures)):
    print("Class: {i:02d} loaded".format(i=i))
    pictures[i].sort(key=lambda pict: pict.grad)
    pictures_all_true.extend(pictures[i][:positions[i]])
    pictures_all_false.extend(pictures[i][positions[i]:])

apply_limits_to_class(mnist_test_x, mnist_test_y, pictures_all_true, pictures_all_false, model)

model.save(model_save_path)
