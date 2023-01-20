import torch
from torch import nn 
from torch.utils.data import DataLoader, TensorDataset
from torchvision import io
import os
import matplotlib.pyplot as plt 
import numpy as np 
from src import network
# from src.utils import detTs
from tqdm import tqdm

device = ("cuda" if torch.cuda.is_available else "cpu")
print(device)

file_path= "/home/yuning/DL/Monument/pt1_data/pt1_train_img"
file_type = [ os.path.join(file_path,image) for image in os.listdir(file_path)  ]
print(f"There are {len(file_type)} types of argument in the path")
print(f"A sample of file is {file_type[0]}")


sample_path = [os.path.join(file_type[0],i) for i in os.listdir(file_type[0])][0]
print(sample_path)
channels,width,height = io.read_image(sample_path).size()

sample_tensor = io.read_image(sample_path)
print(sample_tensor/255.0)
print(type(sample_tensor))

image_tensor = []

type_list = os.listdir(file_path)
for types,name  in zip(file_type,type_list):
    print(f"Saving data set of {name}")
    print(types)
    type_path = [ os.path.join(types,i) for i in os.listdir(types)  ]
    print(type_path[0])
    for files in tqdm(type_path):
        # print(f"saving file {files}")
        imag = io.read_image(files)/255.0
        # print(imag.size())
        image_tensor.append(imag)
    
tensordata = TensorDataset(torch.stack(image_tensor,dim=0))
print(f"shape of dataset is {tensordata.tensors[0].size()}")
torch.save(tensordata,"/home/yuning/DL/Monument/pt1_data/pt1_tensor/{}.pt".format("all"))
print(f"{name}.pt saved !")
image_tensor.clear()
