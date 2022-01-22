from typing import List

from func.Arg import Argument
from func.Function import Function


def equality(arg1: List[Argument], arg2: List[Argument]):
    s1 = len(arg1)
    s2 = len(arg2)
    if s1 != s2:
        return False
    for i in range(s1):
        if arg1[i].value != arg2[i].value or arg1[i].position != arg2[i].position:
            return False
    return True


class CashFunc(Function):

    def __init__(self, func: Function):
        super().__init__()
        self.func = func
        self.cashVal = None
        self.cashArg = None
        self.setPosit = None

    def calculate(self, arg: List[Argument]) -> float:
        if self.cashArg is not None and equality(self.cashArg, arg):
            return self.cashVal
        else:
            self.cashVal = self.func.calculate(arg)
            self.cashArg = arg
            return self.cashVal

    def differential(self, arg_num: int):
        return CashFunc(self.func.differential(arg_num))

    def func_to_str(self):
        return self.func.func_to_str()
