from src.sentience.netsupport.HiddenLayer import HiddenLayer
from src.sentience.netsupport.OutputLayer import OutputLayer
from src.sentience.netsupport.Layer import Layer
from parallellinear.datatypes.Vector import Vector
import parallellinear.calculations.ParallelLinear as pl





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
        

        


    

    def forwardProp(self, input):
        inn = Layer(Vector.vectorFromList(input))
        for layer in self.layers:
            inn = Layer(inn.getNodes().multiply(layer.getWeights()))
            inn.getNodes().add(layer.getBiases())
            inn.getNodes().applyCustomFunction(self.activationFunction)
            layer.setNodes(inn.getNodes())

        return self.layers[-1].getNodes().exportToList()
    