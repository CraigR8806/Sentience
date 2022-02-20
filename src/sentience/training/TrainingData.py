from sentience.training.TrainingDataSpecification import TrainingDataSpecification
from sentience.net.Net import Net
import numpy as np


class TrainingData:

    def __init__(self, trainingDataSpecification:TrainingDataSpecification, trainingData:list, net:Net):
        self.trainingDataSpecification = trainingDataSpecification
        self.trainingData = trainingData
        self.net = net



    @classmethod
    def fromFileWithNewNet(cls, dataPath:str, specificationPath:str, numberOfHiddenNodesPerHiddenLayer:None, data_delimiter=",", specification_field_delimiter=":", specification_value_delimiter=","):
        specification, trainingData = TrainingData._loadDataAndSpecificationfromFile(dataPath, specificationPath, data_delimiter=data_delimiter, specification_field_delimiter=specification_field_delimiter, specification_value_delimiter=specification_value_delimiter)
        if numberOfHiddenNodesPerHiddenLayer == None:
            numberOfHiddenNodesPerHiddenLayer=[int(np.ceil((specification.getNumberOfInputNodes() + specification.getNumberOfOutputNodes())/2))]
        
        net = Net.randomWeightAndBiasNet(specification.getNumberOfInputNodes(), specification.getNumberOfOutputNodes(), numberOfHiddenNodesPerHiddenLayer)
        return cls(trainingDataSpecification=specification, trainingData=trainingData, net=net)

    @classmethod
    def fromFileWithSavedNet(cls, dataPath:str, specificationPath:str, netPath:str, data_delimiter=",", specification_field_delimiter=":", specification_value_delimiter=","):
        specification, trainingData = TrainingData._loadDataAndSpecificationfromFile(dataPath, specificationPath, data_delimiter=data_delimiter, specification_field_delimiter=specification_field_delimiter, specification_value_delimiter=specification_value_delimiter)

        net = Net.loadNetFromFile(netPath)

        return cls(trainingDataSpecification=specification, trainingData=trainingData, net=net)

    def _loadDataAndSpecificationfromFile(dataPath, specificationPath,  data_delimiter=",", specification_field_delimiter=":", specification_value_delimiter=","):
        specification = TrainingDataSpecification.fromFile(specificationPath, dataPath, data_delimiter=data_delimiter, field_delimiter=specification_field_delimiter, value_delimiter=specification_value_delimiter)
        trainingData = []
        with open(dataPath, 'r') as file:
            for line in file.readlines():
                input = []
                target = []
                values=line.split(data_delimiter)
                for i in range(len(specification.getFeatures())):
                    input.append(values[i])
                for i in range(len(specification.getFeatures()), len(values)):
                    target.append(values[i])
                trainingData.append({"input":input, "target": target})

        return (specification, trainingData)


    