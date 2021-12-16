from abc import abstractmethod
from typing import List

from func.Arg import Argument
from func.Function import Function


class CollectorFunction(Function):
    def __init__(self, funcs: List[Function]):
        super().__init__()
        self.functions = funcs
        self.listPosit = set()
        for func in funcs:
            if func.setPosit is not None:
                for el in func.setPosit:
                    self.listPosit.add(el)
            else:
                if func.__class__.__bases__[0].__name__ != "CollectorFunction":
                    self.listPosit.add(func.pos)
                else:
                    self.listPosit.update(func.listPosit)

    def calculate(self, arg: List[Argument]) -> float:
        l = []
        for f in self.functions:
            l.append(f.calculate(arg))
        return self.calculate_collected(l)

    @abstractmethod
    def calculate_collected(self, arg: List[float]) -> float:
        pass

    def differential(self, arg_num: int):
        pass
        # res = []
        # for i in range(len(self.functions)):
        #     r = self.functions.copy()
        #     r[i] = r[i].differential(arg_num)
        #
        #     res.append(Multiplication(r))
        # return Plus(res)