import numpy

from func.collectors.Plus import Plus
from func.functions.Const import Const
from func.functions.Polynome import Polynome
from utils.function_constructor import construct_function


class HiddenLayerDense:

    def __init__(self, type: str, weights: numpy.array, bias_weights: numpy.array):
        self.type = type
        self.weights = weights
        self.bias_weights = bias_weights
        self.function_output = []
        self.function_input = []
        for func in range(len(bias_weights)):
            list_function = []
            list_function.append(Const(bias_weights[func]))
            for i in range(weights[func]):
                list_function.append(Polynome(i, weights[func, i], None, 1))
            p = Plus(list_function)
            self.function_input.append(list_function)
            self.function_output.append(construct_function(type, p, func))

    def link_layer_as_prev(self, prev_hidden_layer):
        prev = prev_hidden_layer.function_output  # calc number of functions out
        curr = self.function_input  # calc number of arguments in
        if len(prev) != len(curr):
            raise Exception('wrong_join_size', "Prev: " + str(len(prev)) + "Curr: " + str(len(curr)))
        for fs in curr:
            for func in fs:
                func.func = prev[func.pos]
        # for i in range(prev.size):
        # change variables to functions, which have the same numeber of elements
