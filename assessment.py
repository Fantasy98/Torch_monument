import torch 
from src.utils import Encode_And_Mask

device = ("cuda" if torch.cuda.is_available() else "cpu")


g_e = torch.load("/home/yuning/DL/Monument/models/pt2_g_e_1200.pt");g_e.eval()
g_d = torch.load("/home/yuning/DL/Monument/models/pt2_g_d_1200.pt");g_d.eval()
dis = torch.load("/home/yuning/DL/Monument/models/pt2_dis_1200.pt");dis.eval()
print("NOTE: All model has been loaded")


normal_imag_path = "/home/yuning/DL/Monument/pt2_data/pt2_test_img/Test_72/normal/"
abnormal_imag_path = "/home/yuning/DL/Monument/pt2_data/pt2_test_img/Test_72/abnormal/"


save_dir = "/home/yuning/DL/Monument/pt2_data/pt2_n_out"
print(f"Save eocoded and mask image to{save_dir}")

Encode_And_Mask(normal_imag_path,save_dir,
                         g_e,g_d,device)

print(f"NOTE:Assessment has been done!")