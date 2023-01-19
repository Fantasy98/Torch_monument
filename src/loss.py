import torch
from torch import nn 
"""
Loss for training
"""
def Adv_loss(g_feature,d_feature):
    """
    Adversial Loss 
    The loss to evaluate how discrimator 
    tell the original and encoded image
    
    It will be backprop to update the weight of generator 


    Args:
        g_feature: tensor size=[N,zdim] the latent-vector output from generator encoder

        d_feature: tensor size=[N,zdim] the latenet-vector output from feature extractor of discrimator
    Return:
        adv_loss
    """
    
    loss = nn.L1Loss(reduction="mean")(g_feature,d_feature)
    
    return loss

def Cnt_loss(ori,gan):
    """
    Construction loss
    The loss to compare 
    the original input and encoded output by GAN
    The most important loss for generator network


    Args:
        ori: tensor size=[N,C,H,W] Original batch

        gan: tensor size=[N,C,H,W] Fake image by GAN

    Return:
        l1: Then MAE error of ori and gan
    """
    
    l1 = nn.L1Loss(reduction="mean")(ori,gan)
    
    return l1



def Enc_loss(ori_enc,gan_enc):
    
    """
    The loss to evaluate the difference 
    between output of generator encoder and Encoder after the decoder
    Which could help for better train the latent express 


    Args:
        ori_enc: tensor size=[N,zdim] original data encoded by generator encoder

        gan_enc: tensor size=[N,zdim] reconstructured data encoded by untrained encoder
    
    Return:
        Enc_loss: Mean-Squared Error between two vector
    """
    
    loss= nn.MSELoss(reduction="mean")(ori_enc,gan_enc)
    return loss
