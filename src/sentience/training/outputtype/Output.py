from abc import ABC, abstractmethod


class Output(ABC):

    @abstractmethod
    def addValueToTargetOutput(self, value, output):
        pass