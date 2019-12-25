import os

class Algorithm():
    def __init__(self, data, alg_obj, model_name):
        self.alg_func = alg_obj
        self.data = data
        self.name = model_name
        self.out_path = "./outputData/"
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)

    def showResults(self, plot=False):
        # TODO
        pass

def Beatlex(Algorithm):
    def run(self):
        self.alg_func(self.data, self.out_path, self.name)