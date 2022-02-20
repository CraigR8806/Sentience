from abc import abstractmethod
from sentience.training.InputOutput import InputOutput


class Output(InputOutput):


    def __init__(self, name:str):
        super().__init__(name)

    @abstractmethod
    def addValueToTargetOutput(self, value, output):
        pass