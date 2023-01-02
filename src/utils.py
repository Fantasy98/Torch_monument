import torch 

def detTs(input:torch.Tensor):
    # Detach a tensor from gpu and let the dtype to be float
    return input.detach().cpu().float()
