import math

from func.Function import Function
from func.functions.BaseFunction import BaseFunction
import func.functions.Cosine as Cos


class Sinus(BaseFunction):

    def __init__(self, pos: int, outCoef: int, func: Function):
        super().__init__(pos, func, outCoef)

    def calculate_function(self, arg: float):
        return math.sin(arg)

    def func_differential(self, func: Function, outCoef: int):
        if func is not None:
            return Cos.Cosine(None, outCoef, func)
        else:
            return Cos.Cosine(self.pos, outCoef, None)

    def func_name(self):
        return "sin"