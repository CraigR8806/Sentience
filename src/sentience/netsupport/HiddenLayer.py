from parallellinear.datatypes.Matrix import Matrix
from parallellinear.datatypes.Vector import Vector
from sentience.netsupport.Layer import Layer

class HiddenLayer(Layer):
    
    
    weights=None
    biases=None

    def __init__(self, data, weights, biases, previousLayersNumberOfNodes):
        super.__init__(data)
        self.weights = Matrix(previousLayersNumberOfNodes, weights)
        self.biases = Vector(biases)

    def getWeights(self):
        return self.weights

    def getBiases(self):
        return self.biases

    def setWeightAtRowColumn(self, row, column, value):
        self.weights.setAtPos(row, column, value)

    def setBiasAtIndex(self, index, value):
        self.biases.setAtPos(index, value)