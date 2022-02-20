from sentience.training.outputtype.Output import Output




class Categorical(Output):


    def __init__(self, name:str, categoryEnumeration:list):
        super().__init__(name)
        self.categoryEnumeration = categoryEnumeration