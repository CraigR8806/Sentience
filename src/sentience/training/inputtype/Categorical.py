from sentience.training.inputtype.Input import Input



class CategoricalInput(Input):





    def __init__(self, name:str, categoryEnumeration:list):
        super().__init__(name)
        self.categoryEnumeration = categoryEnumeration


    def addValueToInput(self, value, input:list):
        for category in self.categoryEnumeration:
            if category == value:
                input.append(1.0)
            else:
                input.append(0.0)
        


