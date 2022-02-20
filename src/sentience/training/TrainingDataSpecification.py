from sentience.training.inputtype.Categorical import CategoricalInput
from sentience.training.inputtype.Continuous import ContinuousInput
from sentience.training.inputtype.Input import Input
from sentience.training.targettype.Categorical import CategoricalTarget
from sentience.training.targettype.Continuous import ContinuousTarget
from sentience.training.targettype.Target import Target
import numpy as np




class TrainingDataSpecification:


    def __init__(self, features:list, target:list, numberOfInputNodes:int, numberOfTargetNodes:int):
        self.features = features
        self.target = target
        self.numberOfInputNodes = numberOfInputNodes
        self.numberOfTargetNodes = numberOfTargetNodes




    @classmethod
    def fromFile(cls, specificationPath:str, dataPath:str, data_delimiter=",", field_delimiter=":", value_delimiter=","):
        features = []
        target = []
        datalines=[]
        numberOfInputNodes=0
        numberOfTargetNodes=0
        with open(dataPath, 'r') as file:
            datalines=[line.split(data_delimiter) for line in file.readlines()]
        with open(specificationPath, 'r') as file:
            lines=file.readlines()
            fieldNumber=0
            for line in lines:
                fields=line.split(field_delimiter)
                name=fields[1]
                if fields[0] == "input":
                    if fields[2] == "continuous":
                        min=np.min([np.float32(dataline[fieldNumber]) for dataline in datalines])
                        max=np.max([np.float32(dataline[fieldNumber]) for dataline in datalines])
                        features.append(ContinuousInput(name, min, max))
                        numberOfInputNodes+=1
                    elif fields[2] == "categorical":
                        categories=fields[3].split(value_delimiter)
                        features.append(CategoricalInput(name, categories))
                        numberOfInputNodes+=len(categories)
                elif fields[0] == "output":
                    if fields[2] == "continuous":
                        min=np.min([np.float32(dataline[fieldNumber]) for dataline in datalines])
                        max=np.max([np.float32(dataline[fieldNumber]) for dataline in datalines])
                        target.append(ContinuousTarget(name, min, max))
                        numberOfTargetNodes+=1
                    elif fields[2] == "categorical":
                        categories=fields[3].split(value_delimiter)
                        target.append(CategoricalTarget(name, categories))
                        numberOfTargetNodes+=len(categories)
                fieldNumber+=1
        return cls(features=features, target=target, numberOfInputNodes=numberOfInputNodes, numberOfTargetNodes=numberOfTargetNodes)

                


    def getFeatures(self):
        return self.features

    def getOutput(self):
        return self.output

    def getNumberOfInputNodes(self):
        return self.numberOfInputNodes

    def getNumberOfTargetNodes(self):
        return self.numberOfTargetNodes
