import torch
from torch import nn 
from torch.utils.data import DataLoader, TensorDataset
from torchvision import io
import os
import matplotlib.pyplot as plt 
import numpy as np 
from src import network
from src.utils import detTs

device = ("cuda" if torch.cuda.is_available else "cpu")
print(device)

file_path= "/home/yuning/DL/Monument/Train"
file_type = [ os.path.join(file_path,image) for image in os.listdir(file_path)  ]
print(f"There are {len(file_type)} types of argument in the path")
print(f"A sample of file is {file_type[0]}")


sample_path = [os.path.join(file_type[0],i) for i in os.listdir(file_type[0])][0]
print(sample_path)
channels,width,height = io.read_image(sample_path).size()
knsize = [5,3]
feature_dims = [32,64,128]
zdim = 32
print(channels,height,width)



g_e = network.g_e(height,width,channels,device,knsize,zdim,feature_dims)
# print(g_e.g_conv.eval)
# print(g_e.g_mlp.eval)

x_sample = io.read_image(sample_path).float().reshape(1,channels,height,width)
# Check the output of convnet
# pred = g_e.g_conv(x_sample)
# print(pred.detach().cpu().size())

# Check the output of latent vector
pred_out = g_e(x_sample)
print(pred_out.detach().cpu().size())


g_d = network.g_d(height,width,channels,device,knsize,zdim,feature_dims)
# print(g_d.d_mlp.eval)
# print(g_d.d_conv.eval)


# d_out_mlp = g_d.d_mlp(pred_out)
# print(d_out_mlp.detach().cpu().size())

d_out_final = g_d(pred_out)
print(d_out_final.detach().cpu().size())

# imag_tensor  = detTs(d_out_final).squeeze(0).permute(2,1,0)
# print(imag_tensor.size(),imag_tensor.dtype)


dis = network.dis(height,width,channels,device,knsize,zdim,feature_dims)
# print(dis.dis_conv.eval())
dis_out = dis.dis_conv(x_sample)
print(dis_out.size())
# print(detTs(dis_out))


