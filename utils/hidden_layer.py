from typing import List
import numpy
from func.Arg import Argument
from func.CashFunc import CashFunc
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
        for func in range(bias_weights.shape[0]):
            list_function = []
            list_function.append(Const(bias_weights[func]))
            for i in range(weights.shape[0]):
                list_function.append(Polynome(i, weights[i, func], None, 1))
            p = Plus(list_function)
            self.function_input.append(list_function)
            self.function_output.append(construct_function(type, p, func))

    def link_layer_as_prev(self, prev_hidden_layer):
        prev = prev_hidden_layer.function_output  # calc number of functions out
        curr = self.function_input  # calc number of arguments in
        if len(prev) != (len(curr[0]) - 1):
            raise Exception('wrong_join_size', "Prev: " + str(len(prev)) + " Curr: " + str(len(curr[0]) - 1))
        j = 0
        for fs in curr:
            for func in fs:
                func.func = CashFunc(prev[func.pos])
            j += 1

    def calculate(self, data: List[Argument]) -> List[Argument]:
        res = []
        for f in range(len(self.function_output)):
            res.append(Argument(f, self.function_output[f].calculate(data)))
        return res
