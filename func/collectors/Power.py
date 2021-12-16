from typing import List

from func.Function import Function
from func.collectors.CollectorFunction import CollectorFunction


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

