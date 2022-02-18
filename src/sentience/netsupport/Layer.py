from parallellinear.datatypes.Vector import Vector


class Layer():

    nodeValues=None
    weights=None
    biases=None

    def __init__(self, data):
        self.nodeValues = data

    @classmethod
    def randomLayer(cls, numberOfNodes):
        return cls(data=Vector(numberOfNodes, random=True))

    def getNodes(self):
        return self.nodeValues

    def setNodes(self, nodes):
        self.nodeValues = nodes

    def setNodeValueAtIndex(self, index, value):
        self.nodeValues.setAtPos(index, value)



    def getWeights(self):
        return self.weights

    def getBiases(self):
        return self.biases

    def setWeightAtRowColumn(self, row, column, value):
        self.weights.setAtPos(row, column, value)

    def setBiasAtIndex(self, index, value):
        self.biases.setAtPos(index, value)

    def getNumberOfNodes(self):
        return len(self.nodeValues)


