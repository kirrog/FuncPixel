import math

from func.Function import Function
from func.functions.BaseFunction import BaseFunction
import func.functions.Sinus as Sin


class Cosine(BaseFunction):

    def __init__(self, pos: int, outCoef: int, func: Function):
        super().__init__(pos, func, outCoef)

    def calculate_function(self, arg: float):
        return math.cos(arg)

    def func_differential(self, func: Function, outCoef: int):
        if func is not None:
            return Sin.Sinus(None, -outCoef, func)
        else:
            return Sin.Sinus(self.pos, -outCoef, None)
