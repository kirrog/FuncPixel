import math

from func.Function import Function
from func.collectors.Multiplication import Multiplication
from func.collectors.Plus import Plus
from func.functions.BaseFunction import BaseFunction
from func.functions.Const import Const


class Sigmoid(BaseFunction):

    def __init__(self, pos: int, outCoef: int, func: Function):
        super().__init__(pos, func, outCoef)

    def calculate_function(self, arg: float):
        return (math.tanh(arg / 2) + 1.0) / 2

    def func_differential(self, func: Function, outCoef: int):
        if func is not None:
            p = [Const(1), Sigmoid(None, -outCoef, func)]
            f = Plus(p)
            m = [Sigmoid(None, outCoef, func), f]
            return Multiplication(m)
        else:
            p = [Const(1), Sigmoid(self.pos, -outCoef, None)]
            f = Plus(p)
            m = [Sigmoid(self.pos, outCoef, None), f]
            return Multiplication(m)

    def func_name(self):
        return "sigm"
