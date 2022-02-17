from sentience.netsupport.HiddenLayer import HiddenLayer
from sentience.netsupport.Layer import Layer




class Net:

    layers=[]
    numberOfInputNodes=0
    numberOfOutputNodes=0
    numberOfNodesPerHiddenLayer=0
    numberOfHiddenLayers=0

    def __init__(self, numberOfInputNodes, numberOfOutputNodes, numberOfNodesPerHiddenLayer, numberOfHiddenLayers):
        self.numberOfInputNodes = numberOfInputNodes
        self.numberOfOutputNodes = numberOfOutputNodes
        self.numberOfNodesPerHiddenLayer = numberOfNodesPerHiddenLayer
        self.numberHiddenLayers = numberOfHiddenLayers