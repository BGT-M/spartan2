
from  torch.utils.data import DataLoader,Dataset
import torch

class MyDataSet(Dataset):
    def __init__(self, time_series,window,stride):
        self.data=time_series
        self.stride=stride
        self.window=window

    def __getitem__(self, index):
        ts=self.data.attrlists[:,index*self.stride:index*self.stride+self.window]
        ts=ts.swapaxes(0,1)
        sample_X=torch.Tensor(ts)
        
        return sample_X

    def __len__(self):
        return (self.data.length-self.window)//self.stride + 1
    
    

def preprocess_data(data,is_train,param):
    window=param["seq_len"]
    stride=param["stride"]
    
    if window&(window-1)!=0:
        raise Exception("sequence length must be power of 2, current: {}".format(param["seq_len"]))
    
    dataset=MyDataSet(data,window,stride)
    
    data_loader=DataLoader(
        dataset=dataset,
        batch_size=param["batch_size"],
        shuffle=is_train,
        drop_last=is_train
        
    )
    
    return data_loader
    
