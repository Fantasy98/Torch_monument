
import torch
from src.loss import Adv_loss,Cnt_loss,Enc_loss
from torch import nn 
def Train_One_Epoch(g_d,g_e,encoder,dis,x,device):

    g_e.optim.zero_grad(); g_d.optim.zero_grad();dis.optim.zero_grad()
    x = x[0]
    ori = x.float().to(device).clone()

    g_e.train(False);g_d.train(False)
    ge = g_e(ori)
    gan = g_d(ge)
    
################################################
    dis.train(True)
    dis_ori = dis(ori)
    one_array = torch.ones(dis_ori.size()).float().to(device)
    
    dis_gan = dis(gan)
    zero_array = torch.zeros(dis_gan.size()).float().to(device)
    
    dis_all = torch.stack([dis_ori,dis_gan],dim=0)
    dis_tar = torch.stack([one_array,zero_array],dim=0)  
    loss_d = nn.BCELoss()(dis_all,dis_tar)
    loss_d.backward(retain_graph=True)
    dis.optim.step()
    dis.train(False)
####################################################  
    g_e.train(True);g_d.train(True)
    enc_ori = encoder(ori)
    enc_gan = encoder(gan)

    with torch.no_grad():
        dconv_ori = dis.dis_conv(ori).requires_grad_(False)
        dconv_gan = dis.dis_conv(gan).requires_grad_(False)
    adv_loss = Adv_loss(dconv_ori,dconv_gan)
    cnt_loss = Cnt_loss(ori,gan)
    enc_loss = Enc_loss(enc_ori,enc_gan)
    g_loss = 40*cnt_loss + adv_loss + enc_loss
    
    g_loss.backward(retain_graph=True)
    g_e.optim.step()
    g_d.optim.step()
    
    return g_loss.item(), loss_d.item()

