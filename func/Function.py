from abc import abstractmethod, ABC
from typing import List

from func.Arg import Argument


class Function(ABC):

    def __init__(self):
        self.setPosit = None

    @abstractmethod
    def calculate(self, arg: List[Argument]) -> float:
        pass

    @abstractmethod
    def differential(self, arg_num: int):
        pass
