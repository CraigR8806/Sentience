from abc import abstractmethod
from sentience.training.InputTarget import InputTarget


class Input(InputTarget):


    def __init(self, name:str):
        super().__init__(name)


    @abstractmethod
    def addValueToInput(self, value, input):
       pass



