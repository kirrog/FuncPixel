import math

from func.Arg import Argument
from func.collectors.Multiplication import Multiplication
from func.collectors.Plus import Plus
from func.functions.Const import Const
from func.functions.Exponent import Exponent
from func.functions.Polynome import Polynome


def calc_func(arg: float):
    # return pow(math.e, arg)
    # return pow(math.e, -arg)
    # return pow(math.e, arg) - pow(math.e, -arg)
    # return pow(math.e, arg) + pow(math.e, -arg)
    # return 1 / (pow(math.e, arg) + pow(math.e, -arg))
    return (pow(math.e, arg) - pow(math.e, -arg)) / (pow(math.e, arg) + pow(math.e, -arg))


def get_func():
    pol1 = Polynome(1, 1, None, 1)
    pol2 = Polynome(1, -1, None, 1)
    f1 = Exponent(1, 1, pol1)
    f2 = Exponent(1, -1, pol2)
    f3 = Exponent(1, 1, pol2)
    p1 = Plus([f1, f2])
    p2 = Plus([f1, f3])
    f4 = Polynome(1, 1, p2, -1)
    m1 = Multiplication([p1, f4])
    return m1


def test_calc():
    functi_test = get_func()
    for i in range(200):
        arg = Argument(1, i * 0.1 - 10)
        a = functi_test.calculate([arg])
        b = calc_func(arg.value)
        if a != b:
            print("Arg: " + str(arg.value) + " A: " + str(a) + " B: " + str(b))


