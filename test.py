import glob

import faiss
import numpy as np
from natsort import natsorted
from tensorflow import keras

model_vectors_path = "/media/kirrog/data/data/"
path_point_pred = "point_answ/"
path_grad_pred = "point_grad_answ/"
path_model_vectors = "points_grad/"
batch_size = 128
pict_size = 28
classes_number = 10
dim = pict_size * pict_size
topn = 90
k = 1000


def load_vectors_from_model(dir_path, batch_size):
    y_point_pred_paths = natsorted(glob.glob(dir_path + path_point_pred + '*'))
    y_grad_pred_paths = natsorted(glob.glob(dir_path + path_grad_pred + '*'))
    x_model_vectors_paths = natsorted(glob.glob(dir_path + path_model_vectors + '*'))
    size = len(y_point_pred_paths)
    print(size)
    assert len(y_point_pred_paths) == len(y_grad_pred_paths) and len(y_grad_pred_paths) == len(x_model_vectors_paths)
    y_point_pred = np.zeros((len(y_point_pred_paths) * batch_size, classes_number), dtype=np.float32)
    y_grad_pred = np.zeros((len(y_grad_pred_paths) * batch_size, classes_number), dtype=np.float32)
    x_model_vectors = np.zeros((len(x_model_vectors_paths) * batch_size, classes_number, pict_size * pict_size),
                               dtype=np.float32)
    for i in range(size):
        y_point_pred_iter = np.load(y_point_pred_paths[i])
        y_grad_pred_iter = np.load(y_grad_pred_paths[i])
        x_model_vectors_iter = np.reshape(np.load(x_model_vectors_paths[i]),
                                          (batch_size, classes_number, pict_size * pict_size))
        y_point_pred[i * batch_size:(i + 1) * batch_size] = y_point_pred_iter
        y_grad_pred[i * batch_size:(i + 1) * batch_size] = y_grad_pred_iter
        x_model_vectors[i * batch_size:(i + 1) * batch_size] = x_model_vectors_iter
    return y_point_pred, y_grad_pred, x_model_vectors


def transform_to_classes_knn_answers(dist, indexes, train_classes, max_dist=None):
    if max_dist is not None:
        delete_id = []
        for i in range(len(dist)):
            if dist[i] < max_dist:
                delete_id.append(i)
        dif = 0
        for i in delete_id:
            dist.pop(i - dif)
            indexes.pop(i - dif)
            dif += 1
    r = np.zeros((classes_number))
    for i in indexes:
        r[int(train_classes[i])] += 1
    answ = -1
    m = r.max()
    flag = False
    for i in range(classes_number):
        if r[i] == m:
            if flag:
                return -1
            else:
                flag = True
                answ = i
    return answ, dist[0]


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# x_vectors = np.array(np.reshape(x_train, (x_train.shape[0], pict_size * pict_size)), dtype=np.float32)
y_point_pred, y_grad_pred, x_model_vectors_raw = load_vectors_from_model(model_vectors_path, batch_size)
# x_model_vectors = np.reshape(x_model_vectors_raw,
#                              (x_model_vectors_raw.shape[0] * x_model_vectors_raw.shape[1],
#                               pict_size * pict_size))
x_model_vectors = np.array(np.reshape(x_model_vectors_raw[:, 9],
                             (x_model_vectors_raw.shape[0],
                              pict_size * pict_size)), dtype=np.float32)[:4500]
print(x_model_vectors.shape)
use = "gpu"
# use = "cpu"
if use == "cpu":
    index = faiss.IndexFlatL2(dim)
else:
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = 0
    flat_config = [cfg]
    resources = [faiss.StandardGpuResources()]
    index = faiss.GpuIndexFlatL2(resources[0], dim, flat_config[0])
print("Index created")
index.add(x_model_vectors)
print("Index trained")
# vectors = np.reshape(x_model_vectors, (x_model_vectors.shape[0] * x_model_vectors.shape[1], x_model_vectors.shape[2]))
y_grad_pred_line = np.reshape(y_grad_pred, (y_grad_pred.shape[0] * y_grad_pred.shape[1]))
# vectors = vectors[:10000]
D, I = index.search(x_model_vectors, topn)
print(D[0])
# exit()
print("Searched")
right = []
wrong = []
right_classes_stat = np.zeros(classes_number)
wrong_classes_stat = np.zeros(classes_number)
checked = []
num_of_classters = 0
needed_to_increase_number = 0
for i in range(D.shape[0]):
    if i in checked:
        checked.remove(i)
        continue
    distances = D[i]
    indexes = I[i]
    check = False
    for j in range(1, D.shape[1]):
        if distances[j] < dim * (0.1):
            if not check:
                check = True
                num_of_classters += 1
            checked.append(indexes[j])
            if j == D.shape[1] - 1:
                needed_to_increase_number += 1
print(num_of_classters)
print(needed_to_increase_number)
