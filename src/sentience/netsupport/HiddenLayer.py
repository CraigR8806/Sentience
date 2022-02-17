from parallellinear.datatypes.Matrix import Matrix
from parallellinear.datatypes.Vector import Vector
from src.sentience.netsupport.Layer import Layer

class HiddenLayer(Layer):
    
    
    weights=None
    biases=None

    def __init__(self, data:Vector, weights:Matrix, biases:Vector):
        super().__init__(data)
        self.weights = weights
        self.biases = biases

    @classmethod
    def randomHiddenLayer(cls, numberOfNodes, previousLayersNumberOfNodes):
        return cls(data=Vector(numberOfNodes, random=True), 
            weights=Matrix(previousLayersNumberOfNodes, numberOfNodes, 
            random=True), biases=Vector(numberOfNodes, random=True))



    def getWeights(self):
        return self.weights

    def getBiases(self):
        return self.biases

    def setWeightAtRowColumn(self, row, column, value):
        self.weights.setAtPos(row, column, value)

    def setBiasAtIndex(self, index, value):
        self.biases.setAtPos(index, value)