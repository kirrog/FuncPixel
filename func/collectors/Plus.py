from typing import List

from func.Function import Function
from func.collectors.CollectorFunction import CollectorFunction


class Plus(CollectorFunction):

    def __init__(self, funcs: List[Function]):
        super().__init__(funcs)

    def calculate_collected(self, arg: List[float]) -> float:
        res = 0
        for i in arg:
            res += i
        return res
