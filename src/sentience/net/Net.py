from sentience.netsupport.HiddenLayer import HiddenLayer
from sentience.netsupport.OutputLayer import OutputLayer
from sentience.netsupport.Layer import Layer
from parallellinear.datatypes.Vector import Vector
import parallellinear.calculations.ParallelLinear as pl
import numpy as np
import pickle as p





class Net:
    

    activationFunctions = {
        "sigmoid": '$i = 1/(1+exp(-$i));'
    }



    def loadCustomActivationFunction(function_name, func):  
        Net.activationFunctions[function_name] = func
        pl.loadCustomFunction(function_name, func)

    def __init__(self, numberOfInputNodes:int, numberOfOutputNodes:int, layers:list, activationFunction="sigmoid"):
        pl.loadPrograms()
        pl.loadCustomFunction(activationFunction, Net.activationFunctions[activationFunction])
        self.numberOfInputNodes = numberOfInputNodes
        self.numberOfOutputNodes = numberOfOutputNodes
        self.activationFunction = activationFunction
        self.layers = layers


    @classmethod
    def loadNetFromFile(cls, path):
        savedNet=None
        with open(path, 'rb') as file:
            savedNet=file.read()
        net=p.loads(savedNet)
        return cls(numberOfInputNodes=net.numberOfInputNodes, numberOfOutputNodes=net.numberOfOutputNodes, layers=net.layers)

    @classmethod
    def randomWeightAndBiasNet(cls, numberOfInputNodes:int, numberOfOutputNodes:int, numberOfNodesPerHiddenLayer:list, activationFunction="sigmoid"):
        if activationFunction not in list(Net.activationFunctions.keys()):
            raise ValueError(activationFunction + " has not been loaded.  Please use Net.loadCustomActivationFunction(function_name, func) to load the function")
        layers=[]

        previousLayersNumberOfNodes = numberOfInputNodes
        layers.append(Layer(Vector.zeros(numberOfInputNodes)))
        for num in numberOfNodesPerHiddenLayer:
            layers.append(HiddenLayer.randomHiddenLayer(num, previousLayersNumberOfNodes))
            previousLayersNumberOfNodes = num

        layers.append(OutputLayer.randomOutputLayer(numberOfOutputNodes, layers[-1].getNumberOfNodes()))
        
        return cls(numberOfInputNodes=numberOfInputNodes, numberOfOutputNodes=numberOfOutputNodes, layers=layers, activationFunction=activationFunction)
        

        

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

    def exportNetToFile(self, path):
        savedNet=p.dumps(self)
        with open(path, 'wb') as file:
            file.write(savedNet)