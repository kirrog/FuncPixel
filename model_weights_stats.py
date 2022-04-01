import numpy as np
import tensorflow as tf

levels_of_weights = 10


def calc_recursively_stat_of_tensor(tensor, stats):
    if len(tensor.shape) == 1:
        for i in range(tensor.shape[0]):
            stats[int(abs(tensor[i] * (levels_of_weights - 1)))] += 1
    else:
        for i in range(tensor.shape[0]):
            calc_recursively_stat_of_tensor(tensor[i], stats)


def calc_stat_of_tensor(tensor):
    stats = np.zeros(levels_of_weights)
    calc_recursively_stat_of_tensor(tensor, stats)
    return stats


def stat_model(model):
    stats = np.zeros(levels_of_weights)
    weights = model.get_weights()
    max_weight = 0
    for lay in weights:
        max_weight = max(max_weight, lay.max(), abs(lay.min()))
    for lay in weights:
        stats += calc_stat_of_tensor(lay / max_weight)
    return stats / np.sum(stats)


models_names = ["data/models/cutted/sup_cut.h5",
                "data/models/dropout/ep022-loss0.025-accuracy0.992_20220401-145621.h5",
                "data/models/cutted_dropout/ep016-loss0.009-accuracy0.997_20220401-144634.h5",
                "data/models/cutted_unreg/ep016-loss0.008-accuracy0.998_20220401-142033.h5"]

for i in range(len(models_names)):
    model = tf.keras.models.load_model(models_names[i])
    stats = stat_model(model)
    string = ""
    for j in range(stats.shape[0]):
        string += "{fl:.04}".format(fl=stats[j])
        if j + 1 != stats.shape[0]:
            string += "\t"
    print("Model: " + models_names[i].split("/")[-2] + " " + string)
