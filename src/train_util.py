import torch
from torch.optim import Adam
from src.config import train_config
from src.loss import Adv_loss,Cnt_loss,Enc_loss

def get_optim(net):
    optimimzer = Adam(net.parameters(),
                        lr=train_config.lr,
                        eps=train_config.eps,weight_decay=1e-5)
    return optimimzer

def fit(g_e,g_d,encoder,dis,
        g_e_optimizer,g_d_optimizer,d_optimizer
        ,train_dl,device):
    
    from tqdm import tqdm
    from torch import nn
    from src.loss import Adv_loss,Cnt_loss,Enc_loss 

    d_loss = nn.BCEWithLogitsLoss()
    loss_weight = train_config.loss_weight

    d_loss_hist = 0
    g_loss_hist = 0
    for x in tqdm(train_dl):
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
        stddev = nn.Softplus()(torch.rand_like(dis_ori).float().to(device))*0.005
        one_array +=  stddev*torch.rand_like(dis_ori)

        dis_gan = dis(gan)
        stddev = nn.Softplus()(torch.rand_like(dis_ori).float().to(device))*0.005
        zero_array = torch.zeros(dis_gan.size()).float().to(device)
        zero_array += stddev*torch.rand_like(dis_gan).float().to(device)
        
        dis_all = torch.stack([dis_ori,dis_gan],dim=0)
        dis_tar = torch.stack([one_array,zero_array],dim=0)
        
        loss_d = d_loss(dis_all,dis_tar)
        loss_d.backward(retain_graph=True)
        d_optimizer.step()
        dis.train(False)
        d_loss_hist += loss_d.item()
    #################################################   
        g_e.train(True);g_d.train(True)
        enc_ori = encoder(ori)
        enc_gan = encoder(gan)

        with torch.no_grad():
                dconv_ori = dis.dis_conv(ori)
                dconv_gan = dis.dis_conv(gan)
      
        adv_loss = Adv_loss(dconv_ori,dconv_gan)
        cnt_loss = Cnt_loss(ori,gan)
        enc_loss = Enc_loss(enc_ori,enc_gan)

        g_loss =loss_weight["cnt_loss"]*cnt_loss +\
                loss_weight["adv_loss"]*adv_loss+\
                loss_weight["enc_loss"]*enc_loss
        g_loss.backward(retain_graph=True)
        g_e_optimizer.step()
        g_d_optimizer.step()
        g_loss_hist+=g_loss.item()

    return d_loss_hist/len(train_dl), g_loss_hist/len(train_dl)
