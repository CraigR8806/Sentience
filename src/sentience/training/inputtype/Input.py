from abc import abstractmethod
from sentience.training.InputOutput import InputOutput


class Input(InputOutput):


    def __init(self, name:str):
        super().__init__(name)


    @abstractmethod
    def addValueToInput(self, value, input):
       pass



