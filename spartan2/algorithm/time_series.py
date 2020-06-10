import os
import torch
from ..models.beatgan.BeatGAN_CNN import BeatGAN_CNN
from ..models.beatgan.BeatGAN_RNN import BeatGAN_RNN
from ..models.beatgan.preprocess import preprocess_data


class SeriesAlgorithm():
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


class Beatlex(SeriesAlgorithm):
    def run(self, param):
        result = self.alg_func(self.data, param, self.out_path, self.name)
        return result
    
class BeatGAN(SeriesAlgorithm):
    def __init__(self,data,alg_obj,model_name):
        super(BeatGAN, self).__init__(data,alg_obj,model_name)
        
    def init_model(self,param,device):
        self.param=param
        if param["network"] == "CNN":
            self.model = BeatGAN_CNN(param, None, device, self.out_path)
        elif param["network"] == "RNN":
            self.model=BeatGAN_RNN(param, None, device, self.out_path)
        else:
            raise Exception("no this network:{}".format(param["network"]))
        
    def run(self,model,data,param,device):
        result=self.alg_func(model,data,param,self.out_path,self.name,device)
        return result
    
    def train(self,time_series):
        train_loader=preprocess_data(time_series,True,self.param)
        self.model.dataloader=train_loader
        self.model.train()
    
    def test(self,time_series):
        test_loader=preprocess_data(time_series,False,self.param)
        self.model.dataloader = test_loader
        return self.model.test()
        
    def export(self,path):
        self.model.save_model_to(path)
        return self.model
