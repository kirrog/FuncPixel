import numpy

from func.Function import Function


def check_nn_format(data: str) -> bool:
    pass


def save_function(func: Function) -> str:
    pass


def load_function(data: str) -> Function:
    pass


def load_nn(filename: str) -> str:
    pass


def create_functions_from_nn(data: str) -> Function:
    pass


def print_2d_numpy_array(data: numpy.array, res: str):
    neirons, weights = data.shape
    res.join("[")
    start_2d = True
    for i in range(neirons):
        start = True
        if not start_2d:
            res.join(";")
        else:
            start_2d = False
        res.join("[")
        for j in range(weights):
            if not start:
                res.join(";")
            else:
                start = False
            res.join(str(data[i, j]))
        res.join("]")
    res.join("]")


def print_1d_numpy_array(data: numpy.array, res: str):
    start = True
    res.join("[")
    for j in data:
        if not start:
            res.join(";")
        else:
            start = False
        res.join(str(j))
    res.join("]")


def print_layer(data2d: numpy.array, data1d: numpy.array, func: str) -> str:
    res = ""
    res.join("{")
    res.join("(")
    res.join(func)
    res.join(")")
    res.join("(")
    print_2d_numpy_array(data2d, res)
    res.join(")")
    res.join("(")
    print_1d_numpy_array(data1d, res)
    res.join(")")
    res.join("}")
    return res
