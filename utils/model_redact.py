import numpy as np
from tensorflow import keras
import tensorflow as tf


def create_hidden_output_layer(model):
    extractor = keras.Model(inputs=model.inputs, outputs=([layer.output for layer in model.layers][2:5]))
    return extractor


def null_weights_of_hidden_layer(output, weights_matrix, weights_shift):
    for i in range(output.shape[0]):
        if output[i] == 1.0:
            weights_shift[i] = 0
            weights_matrix[:, i] = 0


def transform_table_trues(table):
    table[table >= 0.8] = 1.0
    table[table < 0.8] = 0.0
    size = table.shape[0]
    res = np.sum(table, axis=0)
    res /= size
    res[res >= 0.8] = 1.0
    res[res < 0.8] = 0.0
    return res


def model_redact_with_local_max(model, trues, falses):
    work_model = create_hidden_output_layer(model)
    res_true = [transform_table_trues(x) for x in work_model.predict(trues)]
    res_false = [transform_table_trues(x) for x in work_model.predict(falses)]
    for i in range(len(res_true)):
        for j in range(res_false[i].shape[0]):
            if res_false[i][j] == 1.0 and res_true[i][j] == 1.0:
                res_false[i][j] = 0.0
    a = model.get_weights()
    for i in range(int(len(a) / 2) - 1):
        null_weights_of_hidden_layer(res_false[i], a[i * 2], a[i * 2 + 1])
    model.set_weights(a)
