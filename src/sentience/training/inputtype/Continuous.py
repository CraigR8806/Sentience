from sentience.training.inputtype.Input import Input
import numpy as np



class ContinuousInput(Input):


    def __init__(self, name:str, minValue:np.float32, maxValue:np.float32):
        super().__init__(name)
        self.minValue=minValue
        self.maxValue=maxValue

    def normalizeValue(self, value:np.float32) -> np.float23:
        return (value - self.minValue)/(self.maxValue - self.minValue)


    def addValueToInput(self, value:np.float32, input:list) -> np.float32:
        list.append(self.normalizedValue(value))