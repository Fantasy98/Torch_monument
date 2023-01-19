import torch 

def predict_snapshot(g_e,g_d,dis,test_path,save_dir,single:True):

    """
    A function for predict to generate encoded output in .npy format
    Args:
        g_e: generator encoder
        g_d: generator decoder
        test_path: the file path store the image
        save_dir: where to save the .npy data
        single: boolean: to shown a prediction of snapshot or give out whole data
    Return:
    
    """
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


def Encode_And_Mask(
                    abnormal_imag_path,
                    save_dir,
                    g_e,g_d, device,
                    imag_quality = 100
                    ):
    """
    Function to get the Encoded output and absolute-difference between origin and output in grey-scale
    
    Args:
        abnormal_imag_path: where store the image to be processed
        save_dir: where to save output image
                    The image is in .jpg format
        g_e,g_d: encoder and decoder network
        device: cuda to be used
        image_quality: Could be 70~100
    Return:
        A folder with encoded images and mask images in jpg format
    """
    import torch 
    import torchvision
    from torchvision.transforms.functional import rgb_to_grayscale
    from torchvision import io
    import os
    from tqdm import tqdm


    abnormal_imag_path_list = [os.path.join(abnormal_imag_path,i) for i in os.listdir(abnormal_imag_path)]
    abnormal_image_name_list = os.listdir(abnormal_imag_path)
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
        print(f"Made :{save_dir}")
    
    for image_id in tqdm(range(len(abnormal_imag_path_list))):
        abnormal_imag_snap = io.image.read_image(abnormal_imag_path_list[image_id])/255.0
        abnormal_imag_snap_gpu= abnormal_imag_snap.unsqueeze(0).float().to(device)
        with torch.no_grad():
            abnormal_imag_snap_encoded_gpu =g_d(g_e(abnormal_imag_snap_gpu))

        abnormal_imag_snap_encoded_uint8 = torch.tensor(255.0*abnormal_imag_snap_encoded_gpu.clone().detach().cpu().squeeze(),dtype=torch.uint8)
        
        abnormal_image_snap_mask = torch.abs(255.0*abnormal_imag_snap_encoded_gpu.clone().detach().cpu() - 255.0*abnormal_imag_snap_gpu.clone().detach().cpu())
        abnormal_image_snap_mask_grey = rgb_to_grayscale(abnormal_image_snap_mask.clone().detach().cpu().squeeze())
        abnormal_imag_snap_mask_uint8 = torch.tensor(abnormal_image_snap_mask_grey,dtype=torch.uint8)


        io.write_jpeg(abnormal_imag_snap_mask_uint8,
                os.path.join(save_dir,"mask_{}".format(abnormal_image_name_list[image_id])),
                                                                                    quality=imag_quality)
        io.write_jpeg(abnormal_imag_snap_encoded_uint8,
                os.path.join(save_dir,"encoded_{}".format(abnormal_image_name_list[image_id]))
                                                                                    ,quality=imag_quality)
    print("All images have been encoded")      