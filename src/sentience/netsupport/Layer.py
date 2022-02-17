from parallellinear.datatypes.Vector import Vector


class Layer():

    nodeValues=None

    def __init__(self, data):
        self.nodeValues = Vector(data)

    def getNodes(self):
        return self.nodeValues

    def setNodeValueAtIndex(self, index, value):
        self.nodeValues.setAtPos(index, value)



