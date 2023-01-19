import torch
from torch import nn 
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt 
import numpy as np 
from src.network import net_ge,net_gd,net_dis
from src.train_util import get_optim, fit


device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# torch.autograd.set_detect_anomaly(True)

root_path = "/home/yuning/DL/Monument/Train_data/"
tensor_path = root_path+"all.pt"
Tensor = torch.load(tensor_path)
torch.manual_seed(1024)
train_dl  = DataLoader(Tensor,batch_size=2,shuffle=True)

g_e = net_ge(device);g_e_optim = get_optim(g_e)
g_d = net_gd(device);g_d_optim = get_optim(g_d)
encoder = net_ge(device);encoder.train(False)
dis = net_dis(device);d_optim = get_optim(dis)

d_loss, g_loss = fit(g_e,g_d,encoder,dis,
                        g_e_optim,g_d_optim,d_optim,
                        train_dl,device)
print(f"d loss ={d_loss},g_loss = {g_loss}")


torch.save(g_e,"g_e_{}.pt".format(1))
torch.save(g_d,"g_d_{}.pt".format(1))
torch.save(dis,"dis_{}.pt".format(1))