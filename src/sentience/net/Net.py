from src.sentience.netsupport.HiddenLayer import HiddenLayer
from src.sentience.netsupport.OutputLayer import OutputLayer
from src.sentience.netsupport.Layer import Layer
from parallellinear.datatypes.Vector import Vector
import parallellinear.calculations.ParallelLinear as pl
import numpy as np





class Net:
    

    activationFunctions = {
        "sigmoid": '$i = 1/(1+exp(-$i));'
    }



    def loadCustomActivationFunction(function_name, func):  
        Net.activationFunctions[function_name] = func
        pl.loadCustomFunction(function_name, func)

    def __init__(self, numberOfInputNodes:int, numberOfOutputNodes:int, numberOfNodesPerHiddenLayer:list, activationFunction="sigmoid"):
        self.numberOfInputNodes = numberOfInputNodes
        self.numberOfOutputNodes = numberOfOutputNodes
        self.numberOfNodesPerHiddenLayer = numberOfNodesPerHiddenLayer
        self.activationFunction = activationFunction
        self.layers = []

        previousLayersNumberOfNodes = numberOfInputNodes
        self.layers.append(Layer(Vector.zeros(numberOfInputNodes)))
        for num in numberOfNodesPerHiddenLayer:
            self.layers.append(HiddenLayer.randomHiddenLayer(num, previousLayersNumberOfNodes))
            previousLayersNumberOfNodes = num

        self.layers.append(OutputLayer.randomOutputLayer(numberOfOutputNodes, self.layers[-1].getNumberOfNodes()))

    @classmethod
    def randomWeightAndBiasNet(cls, numberOfInputNodes:int, numberOfOutputNodes:int, numberOfNodesPerHiddenLayer:list, activationFunction="sigmoid"):
        pl.loadPrograms()
        if activationFunction not in list(Net.activationFunctions.keys()):
            raise ValueError(activationFunction + " has not been loaded.  Please use Net.loadCustomActivationFunction(function_name, func) to load the function")
        pl.loadCustomFunction(activationFunction, Net.activationFunctions[activationFunction])
        return cls(numberOfInputNodes=numberOfInputNodes, numberOfOutputNodes=numberOfOutputNodes, numberOfNodesPerHiddenLayer=numberOfNodesPerHiddenLayer, activationFunction=activationFunction)
        

        

    def setInput(self, input):
        self.layers[0].setNodes(Vector.fromList(input))
    

    def forwardProp(self, input):
        self.setInput(input)
        inn = self.layers[0]
        for i in range(1, len(self.layers)):
            inn = Layer(inn.getNodes().multiply(self.layers[i].getWeights()))
            inn.getNodes().add(self.layers[i].getBiases())
            inn.getNodes().applyCustomFunction(self.activationFunction)
            self.layers[i].setNodes(inn.getNodes())

        return self.layers[-1].getNodes().exportToList()


    def backProp(self, learningRate:float, targetValues:np.ndarray):

        errorVector=None
        deltaVector=None
        targetValuesVector=Vector.fromList(targetValues)
        for i in range(len(self.layers)-1, 1, -1):
            if i == len(self.layers)-1:
                errorVector = targetValuesVector.sub(self.layers[-1].getNodes(), in_place=False)
                deltaVector = errorVector.elementWiseMultiply(self.sigmoidPrime(self.layers[-1].getNodes()), in_place=False)
            else:
                errorVector = deltaVector.multiply(self.layers[i+1].getWeights().transpose(in_place=False))
                deltaVector = errorVector.elementWiseMultiply(self.sigmoidPrime(self.layers[i].getNodes()), in_place=False)

            self.layers[i].getWeights().add(self.layers[i-1].getNodes().transpose(in_place=False).multiply(deltaVector).scale(learningRate, in_place=False), in_place=False)
       

    def train(self, learningRate:float, input:list, targetValues:list):
        self.forwardProp(input)
        self.backProp(learningRate, targetValues)

    def sigmoidPrime(self, s):
        return s.elementWiseMultiply(s.subScalerFrom(1, in_place=False), in_place=False)

    # def backward(self, X, y, o):
    #     # backward propgate through the network
    #     self.o_error = y - o  # error in output
    #     self.o_delta = self.o_error * self.sigmoidPrime(o)  # applying derivative of sigmoid to error

    #     self.z2_error = self.o_delta.dot(
    #         self.W2.T)  # z2 error: how much our hidden layer weights contributed to output error
    #     self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)  # applying derivative of sigmoid to z2 error

    #     self.W1 += X.T.dot(self.z2_delta)  # adjusting first set (input --> hidden) weights
    #     self.W2 += self.z2.T.dot(self.o_delta)  # adjusting second set (hidden --> output) weights