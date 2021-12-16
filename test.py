import math

from func.Arg import Argument
from func.collectors.Multiplication import Multiplication
from func.collectors.Plus import Plus
from func.collectors.Power import Power
from func.functions.Const import Const
from func.functions.Cosine import Cosine
from func.functions.Exponent import Exponent
from func.functions.Logarithm import Logarithm
from func.functions.Polynome import Polynome
from func.functions.Sinus import Sinus


def calc_func(arg: float):
    if arg > 0:
        return (42 * math.exp(arg)) + (math.cos(arg) * math.log(abs(arg))) + (pow(arg * arg, math.sin(arg)))
    else:
        return (42 * math.exp(arg)) + (pow(arg * arg, math.sin(arg)))


def get_func():
    f1 = Exponent(1, 1, None)
    f2 = Const(42)
    f3 = Cosine(1, 1, None)
    f4 = Logarithm(1, 1, None)
    f5 = Polynome(1, 1, None, 2)
    f6 = Sinus(1, 1, None)
    m1 = Multiplication([f1, f2])
    m2 = Multiplication([f3, f4])
    pow1 = Power([f5, f6])
    return Plus([m1, m2, pow1])


functi_test = get_func()

for i in range(100):
    arg = Argument(1, i - 50)
    a = functi_test.calculate([arg])
    b = calc_func(arg.value)
    if a != b:
        print("Arg: " + str(arg.value) + " A: " + str(a) + " B: " + str(b))
