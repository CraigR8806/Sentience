from abc import ABC, abstractmethod


class Input(ABC):




   @abstractmethod
   def addValueToInput(self, value, input):
       pass

