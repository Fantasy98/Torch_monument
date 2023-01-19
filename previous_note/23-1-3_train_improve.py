import torch
from torch import nn 
from torch.utils.data import DataLoader, TensorDataset
# from torchvision import io
import os
import matplotlib.pyplot as plt 
import numpy as np 
from src import network
from src.utils import detTs
from src.loss import Adv_loss,Cnt_loss,Enc_loss
from src.train import Train_One_Epoch
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# torch.autograd.set_detect_anomaly(True)

root_path = "/home/yuning/DL/Monument/Train_data/"
tensor_path = root_path+"all.pt"
Tensor = torch.load(tensor_path)
torch.manual_seed(1024)
train_dl  = DataLoader(Tensor,batch_size=2,shuffle=True)

sample = iter(train_dl).next()
print(sample[0].size())
BATCHSIZE,CHANNEL,HEIGHT,WIDTH =sample[0].size() 
print(CHANNEL,HEIGHT,WIDTH)
knsize = [5,3]
feature_dims = [32,64,128]
zdim = 32
g_e = network.g_e(HEIGHT,WIDTH,CHANNEL,device,knsize,zdim,feature_dims)


g_d = network.g_d(HEIGHT,WIDTH,CHANNEL,device,knsize,zdim,feature_dims)


encoder = network.g_e(HEIGHT,WIDTH,CHANNEL,device,knsize,zdim,feature_dims)


dis = network.dis(HEIGHT,WIDTH,CHANNEL,device,knsize,zdim,feature_dims)


# g_e_optimizer = torch.optim.Adam(g_e.parameters(),lr= 1e-3)
# g_d_optimizer = torch.optim.Adam(g_d.parameters(),lr= 1e-3)
# d_optimizer = torch.optim.Adam(dis.parameters(),lr=1e-3)

g_e.to(device);g_d.to(device);encoder.to(device);dis.to(device)

d_loss = nn.BCELoss(reduction="sum")

encoder.train(False)

for x in train_dl:
    g_loss,d_loss = Train_One_Epoch(g_d,g_e,encoder,dis,x,device)
    print(g_loss)
    print(d_loss)
    
    