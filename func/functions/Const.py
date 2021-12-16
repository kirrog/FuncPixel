from func.Function import Function
from func.functions.BaseFunction import BaseFunction


class Const(BaseFunction):

    def __init__(self, const: float):
        super().__init__(pos=0, func=None, outCoef=0)
        self.const = const

    def calculate_function(self, arg: float):
        return self.const

    def differential(self, arg_num):
        return Const(0)

    def func_differential(self, func: Function, outCoef: int) -> Function:
        return Const(0)