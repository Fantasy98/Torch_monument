from Assessment.img import image_generator


model_path = "/home/yuning/DL/Monument/trained/1800epoch"
n_path = "/home/yuning/DL/Monument/Test_IMG/12_abn_12_n/normal"
abn_path = "/home/yuning/DL/Monument/Test_IMG/12_abn_12_n/abnormal"
tar_path = "/home/yuning/DL/Monument/23-1-6/1800epoch"
img_g = image_generator(
                        model_path,
                        n_path ,
                        abn_path,
                        tar_path)

img_g.abn_image()
img_g.n_image()

