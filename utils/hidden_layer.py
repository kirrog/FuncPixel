import numpy


class HiddenLayer:

    def __init__(self, type: str, weights: numpy.array, bias_weights: numpy.array):
        self.type = type
        self.weights = weights
        self.bias_weights = bias_weights
        self.function_output = None
        # build from it list of functions

    def link_layer_as_prev(self, prev_hidden_layer):
        prev = prev_hidden_layer.function_output # calc number of functions out
        curr = self.function_output # calc number of arguments in
        if (prev.size != curr.size):
            raise Exception('wrong_join_size', "Prev: " + str(prev.size) + "Curr: " + str(curr.size))
        # for i in range(prev.size):
        # change variables to functions, which have the same numeber of elements
