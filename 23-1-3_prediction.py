import torch 
from torch import nn 
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)

g_e = torch.load("/home/yuning/DL/Monument/models/23-1-2/g_e_200.pt")
g_d = torch.load("/home/yuning/DL/Monument/models/23-1-2/g_d_200.pt")
dis = torch.load("/home/yuning/DL/Monument/models/23-1-2/dis_200.pt")

test_dl = DataLoader(torch.load("/home/yuning/DL/Monument/Train_data/basic.pt"),batch_size=1,shuffle=1)


g_e.to(device);g_d.to(device);dis.to(device)
with torch.no_grad():
    for x in test_dl:
        x = x[0]
        ori = x.float().to(device)
        
        zvector = g_e(ori)
        pred = g_d(zvector)
        break

pred = pred.cpu().squeeze().permute(1,2,0)
ori = ori.cpu().squeeze().permute(1,2,0)
plt.figure(0)
plt.imshow(pred)
plt.figure(1)
plt.imshow(ori)
plt.show()

