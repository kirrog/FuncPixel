from abc import abstractmethod, ABC
from typing import List
from func.Arg import Argument
from func.Function import Function
import func.functions.Const as Con


class BaseFunction(Function):

    def __init__(self, pos: int, func: Function, outCoef: float):
        super().__init__()
        self.func = func
        self.outCoef = outCoef
        if func is not None:
            if func.setPosit is not None:
                self.listPosit = func.setPosit.copy()
            else:
                self.listPosit = set()
            if func.__class__.__bases__[0].__name__ == "BaseFunction":
                self.listPosit.add(func.pos)
        self.pos = pos

    def calculate(self, arg: List[Argument]) -> float:
        if self.func is not None:
            return self.calculate_function(self.func.calculate(arg)) * self.outCoef
        else:
            for a in arg:
                if a.position == self.pos:
                    return self.calculate_function(a.value) * self.outCoef
            return 0.0

    @abstractmethod
    def calculate_function(self, arg: float) -> float:
        pass

    def differential(self, arg_num: int) -> Function:
        if self.func is not None:
            if arg_num in self.func.setPosit:
                pob = self.func.differential(arg_num)
                return self.func_differential(pob, self.outCoef)
            else:
                return Con.Const(0)
        else:
            if self.pos == arg_num:
                return self.func_differential(None, self.outCoef)

    @abstractmethod
    def func_differential(self, func: Function, outCoef: int) -> Function:
        pass
