from src.sentience.netsupport.Layer import Layer
from parallellinear.datatypes.Vector import Vector
from parallellinear.datatypes.Matrix import Matrix

class OutputLayer(Layer):

    def __init__(self, data:Vector, weights:Matrix, biases:Vector):
        super().__init__(data)
        self.weights = weights
        self.biases = biases

    @classmethod
    def randomOutputLayer(cls, numberOfOutputNodes, previousLayersNumberOfNodes):
        return cls(data=Vector.zerosVector(numberOfOutputNodes), 
            weights=Matrix.randomMatrix(previousLayersNumberOfNodes, numberOfOutputNodes, 
            random_low=-1), biases=Vector.zerosVector(numberOfOutputNodes))