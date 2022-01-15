import numpy as np

from func.Arg import Argument


def from_numpy_to_args_list(data: np.array):
    res = []
    data = data.flat
    for i in range(len(data)):
        res.append(Argument(i, data[i]))
    return res
