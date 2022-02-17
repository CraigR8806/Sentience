from parallellinear.datatypes.Vector import Vector


class Layer():

    nodeValues=None

    def __init__(self, data):
        self.nodeValues = data

    @classmethod
    def randomLayer(cls, numberOfNodes):
        return cls(data=Vector(numberOfNodes, random=True))

    def getNodes(self):
        return self.nodeValues

    def setNodeValueAtIndex(self, index, value):
        self.nodeValues.setAtPos(index, value)



