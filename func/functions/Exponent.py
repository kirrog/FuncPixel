import math

from func.Function import Function
from func.functions.BaseFunction import BaseFunction


class Exponent(BaseFunction):

    def __init__(self, pos: int, outCoef: int, func: Function):
        super().__init__(pos, func, outCoef)

    def calculate_function(self, arg: float):
        return math.pow(math.e, arg)

    def func_differential(self, func: Function, outCoef: int):
        if func is not None:
            return Exponent(None, outCoef, func)
        else:
            return Exponent(self.pos, outCoef, None)
