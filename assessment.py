import torch 
from src.utils import Encode_And_Mask

device = ("cuda" if torch.cuda.is_available() else "cpu")


epoch = 1200
g_e = torch.load("/home/yuning/DL/Monument/models/pt1_g_e_{}.pt".format(epoch));g_e.eval()
g_d = torch.load("/home/yuning/DL/Monument/models/pt1_g_d_{}.pt".format(epoch));g_d.eval()
# dis = torch.load("/home/yuning/DL/Monument/models/dis_{}.pt".format(epoch));dis.eval()
print("NOTE: All model has been loaded")


abnormal_imag_path = "/home/yuning/DL/Monument/pt1_data/pt1_train_img/Artificial_Aug_1_basic"
# abnormal_imag_path = "/home/yuning/DL/Monument/pt2_data/pt2_test_img/Test_72/abnormal"


save_dir = "/home/yuning/DL/Monument/pt1_data/pt1_abn_out_Epoch={}".format(epoch)
print(f"Save eocoded and mask image to{save_dir}")

Encode_And_Mask(abnormal_imag_path,save_dir,
                         g_e,g_d,device)

print(f"NOTE:Assessment has been done!")