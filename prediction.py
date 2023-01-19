import torch 
from torch import nn 
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from src.utils import predict_snapshot
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)

epoch = 1200
g_e = torch.load("/home/yuning/DL/Monument/g_e_{}.pt".format(epoch))
g_d = torch.load("/home/yuning/DL/Monument/g_d_{}.pt".format(epoch))
dis = torch.load("/home/yuning/DL/Monument/dis_{}.pt".format(epoch))

# test_dl = DataLoader(torch.load("/home/yuning/DL/Monument/Train_data/Artificial_Aug_2_exposure.pt"),batch_size=1,shuffle=1)

import os
pwd = os.getcwd()
predict_snapshot(g_e,g_d,dis,test_path="/home/yuning/DL/Monument/Train_data/Artificial_Aug_2_exposure.pt",save_dir=pwd,single=True)
