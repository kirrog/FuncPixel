from func.Function import Function
from func.collectors.Plus import Plus
from func.functions.Const import Const
from func.functions.Cosine import Cosine
from func.functions.Exponent import Exponent
from func.functions.Logarithm import Logarithm
from func.functions.Polynome import Polynome
from func.functions.Sinus import Sinus


def construct_function(name_func: str, collect: Function, pos: int):
    if name_func == 'cos':
        return Cosine(pos, 1, collect)
    elif name_func == 'sin':
        return Sinus(pos, 1, collect)
    elif name_func == 'exp':
        return Exponent(pos, 1, collect)
    elif name_func == 'log':
        return Logarithm(pos, 1, collect)
    elif name_func == 'pol':
        return Polynome(pos, 1, collect, 1)
    elif name_func == 'sigm':
        f = Polynome(pos, -1, None, 1)
        c = [Const(1), Exponent(pos, 1, f)]
        m = Plus(c)
        return Polynome(pos, 1, m, -1)
