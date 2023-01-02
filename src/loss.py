import torch
from torch import nn 

def Adv_loss(g_feature,d_feature):
    loss = nn.L1Loss(reduction="sum")(g_feature,d_feature)
    
    return loss

def Cnt_loss(ori,gan):
    l1 = nn.L1Loss(reduction="sum")(ori,gan)
    # loss = torch.mean(l1.item())
    return l1
def Enc_loss(ori_enc,gan_enc):
    loss= nn.MSELoss(reduction="sum")(ori_enc,gan_enc)
    return loss
