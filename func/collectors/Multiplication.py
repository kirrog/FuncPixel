from abc import abstractmethod
from typing import List

from func.Arg import Argument
from func.Function import Function
from func.collectors.CollectorFunction import CollectorFunction


class Multiplication(CollectorFunction):

    def __init__(self, funcs: List[Function]):
        super().__init__(funcs)

    def calculate_collected(self, arg: List[float]) -> float:
        res = 1
        if len(arg) > 0:
            for i in arg:
                if i == 0:
                    return 0
                else:
                    res *= i
        else:
            res = 0
        return res
