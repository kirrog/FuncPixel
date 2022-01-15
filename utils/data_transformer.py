from typing import List

import numpy as np

from func.Arg import Argument
from utils.hidden_layer import HiddenLayerDense


def from_numpy_to_args_list(data: np.array) -> List[Argument]:
    res = []
    data = data.flat
    for i in range(len(data)):
        res.append(Argument(i, data[i]))
    return res


def from_args_list_to_numpy(data: List[Argument]) -> np.array:
    res = np.zero((len(data)))
    for i in data:
        res[i.position] = i.value
    return res


def calculate_by_layers(model: HiddenLayerDense, data: np.array) -> np.array:
    input_data = from_numpy_to_args_list(data)
    res = model.calculate(input_data)
    return res
