from typing import List

from func.Function import Function
from func.collectors.CollectorFunction import CollectorFunction
from func.collectors.Multiplication import Multiplication
from func.functions.Exponent import Exponent
from func.functions.Logarithm import Logarithm


class Power(CollectorFunction):

    def __init__(self, funcs: List[Function]):
        super().__init__(funcs)

    def calculate_collected(self, arg: List[float]) -> float:
        if len(arg) > 0:
            res = arg[0]
            for i in arg[1:]:
                if i == 0:
                    return 1
                else:
                    res **= i
            return res
        else:
            return 0

    def differential(self, arg_num: int):
        function = self.functions[0]
        for i in range(1, len(self.functions)):
            function = self.diffur(function, self.functions[i], arg_num)
        return function

    def diffur(self, one: Function, two: Function, arg_num:int) -> Function:
        power = []
        power.append(Logarithm(None, 1, one))
        power.append(two)
        m1 = Multiplication(power)
        exp = Exponent(m1)
        dif = []
        dif.append(exp)
        dif.append(m1.differential(arg_num))
        m2 = Multiplication(dif)
        return m2
