import numpy as np
import glob
from natsort import natsorted

dir_path = "/media/kirrog/data/data/"
path_point_pred = "point_answ/"
path_grad_pred = "point_grad_answ/"
path_model_vectors = "points_grad/"
batch_size = 128
pict_size = 28
classes_number = 10
dim = pict_size * pict_size


def load_vectors_from_model():
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