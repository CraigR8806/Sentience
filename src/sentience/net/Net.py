from src.sentience.netsupport.HiddenLayer import HiddenLayer
from src.sentience.netsupport.OutputLayer import OutputLayer
from src.sentience.netsupport.Layer import Layer
from parallellinear.datatypes.Vector import Vector
import parallellinear.calculations.ParallelLinear as pl





class Net:



    layers=[]
    numberOfInputNodes=0
    numberOfOutputNodes=0
    numberOfNodesPerHiddenLayer=[]
    activationFunction = "sigmoid"

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

        print(Net.activationFunctions)
        print(activationFunction)

        if activationFunction not in list(Net.activationFunctions.keys()):
            raise ValueError(activationFunction + " has not been loaded.  Please use Net.loadCustomActivationFunction(function_name, func) to load the function")
        self.activationFunction = activationFunction
        pl.loadPrograms()
        pl.loadCustomFunction(self.activationFunction, Net.activationFunctions[self.activationFunction])
        previousLayersNumberOfNodes = numberOfInputNodes
        for num in numberOfNodesPerHiddenLayer:
            self.layers.append(HiddenLayer.randomHiddenLayer(num, previousLayersNumberOfNodes))
            previousLayersNumberOfNodes = num

        self.layers.append(OutputLayer.randomOutputLayer(numberOfOutputNodes, self.layers[-1].getNumberOfNodes()))

        


    

    def forwardProp(self, input):
        inn = Layer(Vector.vectorFromList(input))
        for layer in self.layers:
            print(type(inn.getNodes().getData()))
            inn = Layer(inn.getNodes().multiply(layer.getWeights()))
            inn.getNodes().add(layer.getBiases())
            inn.getNodes().applyCustomFunction(self.activationFunction)
            layer.setNodes(inn)

        return layer[-1].getNodes().exportToList()
    