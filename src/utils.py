import torch 

def predict_snapshot(g_e,g_d,dis,test_path,save_dir,single:True):
    import torch 
    from torch import nn 
    from torch.utils.data import DataLoader
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np 
    import os
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    
    
    g_d.eval();g_e.eval();dis.eval()
    g_e.to(device);g_d.to(device);dis.to(device)
    if single:
        test_dl = DataLoader(torch.load(test_path),batch_size=1,shuffle=True)

        x = iter(test_dl).next()
        x = x[0]
        ori = x.float().to(device)
                
        zvector = g_e(ori)
        pred = g_d(zvector)
        is_true = dis(pred)
        print(f"Dis value ={is_true.item()}")
        pred = pred.cpu().squeeze().permute(1,2,0).detach()
        ori = ori.cpu().squeeze().permute(1,2,0).detach()
        plt.figure(0,figsize=(16,9))
        plt.imshow(pred)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(save_dir,"pred"),bbox_inches="tight")
        
        plt.figure(1,figsize=(16,9))
        plt.imshow(ori)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(save_dir,"target"),bbox_inches="tight")

    else:
        test_dl = DataLoader(torch.load(test_path),batch_size=1,shuffle=False)

        with torch.no_grad():
            pred_list = []
            tar_list = []
            dis_list = []
            for x in test_dl:
                x = x[0]
                ori = x.float().to(device)
                
                zvector = g_e(ori)
                pred = g_d(zvector)
                is_true = dis(pred)
                pred = pred.cpu().squeeze().permute(1,2,0).detach().numpy()
                ori = ori.cpu().squeeze().permute(1,2,0).detach().numpy()
                pred_list.append(pred)
                tar_list.append(ori)
                dis_list.append(is_true.item())
        pred_array = np.array(pred_list)
        tar_array = np.array(tar_list)
        dis_array = np.array(dis_list)
        del pred_list,tar_list,dis_list
        np.save(os.path.join(save_dir,"pred.npy"),pred_array)
        print(f"predict data saved, with shape of {pred_array.shape}")
    
        np.save(os.path.join(save_dir,"tar.npy"),tar_array)
        print(f"Target data saved, with shape of {tar_array.shape}")
        
        np.save(os.path.join(save_dir,"dis.npy"),dis_array)
        print(f"Dis data saved, with shape of {dis_array.shape}")