from sentience.netsupport.training.inputtype.Input import Input
import numpy as np



class ContinuousInput(Input):


    def __init__(self, minValue:np.float32, maxValue:np.float32):
        self.minValue=minValue
        self.maxValue=maxValue

    def normalizeValue(self, value:np.float32) -> np.float23:
        return (value - self.minValue)/(self.maxValue - self.minValue)


    def addValueToInput(self, value:np.float32, input:list) -> np.float32:
        list.append(self.normalizedValue(value))