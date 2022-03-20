import numpy as np
import glob
from natsort import natsorted

dir_path = "/media/kirrog/data/data/"
path_point_pred = "point_answ/"
path_grad_pred = "point_grad_answ/"
path_model_vectors = "points_grad/"
path_grad_local_minims = "points_nearest_grad_calc/"
batch_size = 128
pict_size = 28
classes_number = 10
dim = pict_size * pict_size


def load_data_from_disk(files_dirs, shape, batch_reshaper, batch_size):
    paths = natsorted(glob.glob(dir_path + files_dirs + '*'))
    size = len(paths)
    print(size)
    data_shape = [size * batch_size]
    data_shape.extend(shape)
    data = np.zeros(data_shape, dtype=np.float32)
    for i in range(size):
        batch_data = batch_reshaper(np.load(paths[i]))
        data[i * batch_size:(i + 1) * batch_size] = batch_data
    print(data.shape)
    return data


def load_y_point_pred():
    return load_data_from_disk(path_point_pred, [classes_number], lambda d: d, 128)


def load_y_grad_pred():
    return load_data_from_disk(path_grad_pred, [classes_number], lambda d: d, 128)


def load_x_model_vectors():
    return load_data_from_disk(path_model_vectors, [classes_number, pict_size * pict_size],
                               lambda d: np.reshape(d, (batch_size, classes_number, pict_size * pict_size)), 128)


def load_grad_local_minims():
    return load_data_from_disk(path_grad_local_minims, [classes_number, 2, 2], lambda d: d, 256)[:, :, 0, :]


def load_grad_local_minims_and_vals():
    return load_data_from_disk(path_grad_local_minims, [classes_number, 2, 2], lambda d: d, 256)


def load_local_vals():
    return load_data_from_disk(path_grad_local_minims, [classes_number, 2, 2], lambda d: d, 256)[:, :, 1, :]
