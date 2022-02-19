from sentience.netsupport.training.inputtype.Input import Input



class CategoricalInput(Input):





    def __init__(self, categories:list):
        self.categories = categories


    def addValueToInput(self, value, input:list):
        for category in self.categories:
            if category == value:
                input.append(1.0)
            else:
                input.append(0.0)
        


