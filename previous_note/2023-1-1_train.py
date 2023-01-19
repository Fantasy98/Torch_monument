import torch
from torch import nn 
from torch.utils.data import DataLoader, TensorDataset
# from torchvision import io
import os
import matplotlib.pyplot as plt 
import numpy as np 
from src import network
from src.network import Init_Conv
from src.utils import detTs
from src.loss import Adv_loss,Cnt_loss,Enc_loss

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
g_e.apply(Init_Conv)

g_d = network.g_d(HEIGHT,WIDTH,CHANNEL,device,knsize,zdim,feature_dims)
g_d.apply(Init_Conv)

encoder = network.g_e(HEIGHT,WIDTH,CHANNEL,device,knsize,zdim,feature_dims)
encoder.apply(Init_Conv)

dis = network.dis(HEIGHT,WIDTH,CHANNEL,device,knsize,zdim,feature_dims)
dis.apply(Init_Conv)

g_e_optimizer = torch.optim.Adam(g_e.parameters(),lr= 1e-3,weight_decay=1e-5)
g_d_optimizer = torch.optim.Adam(g_d.parameters(),lr= 1e-3,weight_decay=1e-5)
d_optimizer = torch.optim.Adam(dis.parameters(),lr=1e-3,weight_decay=1e-5)

g_e.to(device);g_d.to(device);encoder.to(device);dis.to(device)

d_loss = nn.BCEWithLogitsLoss()

encoder.train(False)

num_step = 1000
i = 0 
for x in train_dl:
    i +=1
   
    g_e_optimizer.zero_grad(); g_d_optimizer.zero_grad();d_optimizer.zero_grad()


    x = x[0]
    ori = x.float().to(device).clone()

    g_e.train(False);g_d.train(False)

    ge = g_e(ori)
    gan = g_d(ge)
    
################################################
    dis.train(True)
    
    dis_ori = dis(ori)
    one_array = torch.ones(dis_ori.size()).float().to(device)
    stddev = nn.Softplus()(torch.rand_like(dis_ori).float().to(device))*0.1
    one_array +=  stddev*torch.rand_like(dis_ori)
    
    dis_gan = dis(gan)
    stddev = nn.Softplus()(torch.rand_like(dis_ori).float().to(device))*0.1
    zero_array = torch.zeros(dis_gan.size()).float().to(device)
    zero_array += stddev*torch.rand_like(dis_gan).float().to(device)
    dis_all = torch.stack([dis_ori,dis_gan],dim=0)
    dis_tar = torch.stack([one_array,zero_array],dim=0)

    
    loss_d = d_loss(dis_all,dis_tar)
    print(loss_d.item())
    loss_d.backward(retain_graph=True)

    d_optimizer.step()

    dis.train(False)
 #################################################   
    g_e.train(True);g_d.train(True)
    enc_ori = encoder(ori)
    enc_gan = encoder(gan)


    with torch.no_grad():
        # dconv_ori = dis.dis_conv(ori).requires_grad_(False)
        dconv_ori = dis.dis_conv(ori)

        # dconv_gan = dis.dis_conv(gan).requires_grad_(False)
        dconv_gan = dis.dis_conv(gan)
    adv_loss = Adv_loss(dconv_ori,dconv_gan)
    # print(adv_loss.item())
    cnt_loss = Cnt_loss(ori,gan)
    # print(cnt_loss.item())
    enc_loss = Enc_loss(enc_ori,enc_gan)
    # print(enc_loss.item())
    g_loss = 40*cnt_loss + adv_loss + enc_loss
    
    g_loss.backward(retain_graph=True)
    g_e_optimizer.step()
    g_d_optimizer.step()
    print(g_loss.item())

    # g_e.train(False);g_d.train(False)
######################################################
    
    # print(d_optimizer.state)
    if i == num_step:
        break

torch.save(g_e,"g_e_{}.pt".format(num_step))
torch.save(g_d,"g_d_{}.pt".format(num_step))
torch.save(dis,"dis_{}.pt".format(num_step))