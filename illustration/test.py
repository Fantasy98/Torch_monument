from statistics import mode
import Assessment
from Assessment.img import image_generator
model_path =  "/home/yuning/DL/Monument/1800epoch"
abn_path = "/home/yuning/DL/Monument/Test_IMG/Test_72/abnormal"
n_path = "/home/yuning/DL/Monument/Test_IMG/Test_72/normal"
tar_path = "/home/yuning/DL/Monument/target"
gen = image_generator(model_path=model_path,
                        n_path=n_path,
                        abn_path=abn_path,
                        tar_path=tar_path)
gen.abn_image()
gen.n_image()