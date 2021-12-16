import math

from func.Function import Function
from func.functions.BaseFunction import BaseFunction
import func.functions.Polynome as Pol


class Logarithm(BaseFunction):

    def __init__(self, pos: int, outCoef: int, func: Function):
        super().__init__(pos, func, outCoef)

    def calculate_function(self, arg: float):
        if arg > 0 :
            return math.log(abs(arg))
        else:
            return 0

    def func_differential(self, func: Function, outCoef: int):
        if func is not None:
            return Pol.Polynome(None, outCoef, func, pos=-1)
        else:
            return Pol.Polynome(self.pos, -outCoef, None, pos=-1)
