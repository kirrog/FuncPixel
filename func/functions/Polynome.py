import math

from func.Function import Function
from func.functions.BaseFunction import BaseFunction


class Polynome(BaseFunction):

    def __init__(self, pos: int, outCoef: int, func: Function, pow: int):
        super().__init__(pos, func, outCoef)
        self.pow = pow

    def calculate_function(self, arg: float):
        return math.pow(arg, self.pow)

    def func_differential(self, func: Function, outCoef: int):
        if func is not None:
            return Polynome(None, outCoef * self.pow, func, self.pow - 1)
        else:
            return Polynome(self.pos, outCoef * self.pow, None, self.pow - 1)
