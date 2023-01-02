from multiprocessing.util import is_exiting
import tensorflow as tf
from tensorflow import keras
import numpy as np 
from tensorflow.keras.backend import clear_session
from PIL import Image

import zipfile
from Assessment import Load
import os 

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


from datetime import date

class image_generator():
    """Image Generator"""
    ## This class is for generate encoded normal and abnormal image
    ## The generated data will be zipped and saved
    ## Input 
    # 1 model_path: The dir contains all the model
    # 2 n_path : Dir contains all normal image
    # 3 abn_path: Dir contains all abnormal image
    # 4 tar_path: Target path to save the zip file 
    def __init__(self,model_path,n_path,abn_path,tar_path):
        self.model_path = model_path
        self.n_path = n_path
        self.abn_path = abn_path
        self.tar_path = tar_path
        model = Load.models(self.model_path)
        self.AE = model.get_AE()

    def abn_image(self):
        
        
        print(self.AE.summary())
        list_abn = os.listdir(self.abn_path)
        list_abn_join = [os.path.join(self.abn_path,i) for i in list_abn]
        Lenth = len(list_abn_join)
        print(Lenth)
        for idx in np.arange(0,Lenth,12):
            img_abn = [ np.array(Image.open(img)) for img in list_abn_join[idx:idx+12]]
            img_abn_array = np.array(img_abn)
            print(img_abn_array.shape)
            encoded_abn = self.AE.predict(img_abn_array)
            print(encoded_abn.shape)
            encoded_n_img = []
            for img in np.arange(encoded_abn.shape[0]):
                image = Image.fromarray(np.uint8(encoded_abn[img,:,:,:]))
                encoded_n_img.append(image)
            
            abn_tar = os.path.join(self.tar_path,"abn")
            if os.path.exists(abn_tar)== False:
                os.makedirs(abn_tar)
            abn_name_tar = [os.path.join(abn_tar,i) for i in list_abn[idx:idx+12]]


            for image,name in zip(encoded_n_img,abn_name_tar):

                image.save(name)
            zipname = "abn_{}_{}.zip".format(idx+12,date.today())
            zip_path = os.path.join(abn_tar,zipname)
            if os.path.exists(zip_path) == False:
                zipf = zipfile.ZipFile(zip_path,"w")
                for img in abn_name_tar:
                    zipf.write(img)
                zipf.close()
            for img in abn_name_tar:
                os.remove(img)
    
    def n_image(self):
        # AE =self.AE
        print(self.AE.summary())
        list_n = os.listdir(self.n_path)
        list_n_join = [os.path.join(self.n_path,i) for i in list_n]
        Lenth = len(list_n_join)
        print(Lenth)
        
        for idx in np.arange(0,Lenth,12):
            img_n = [ np.array(Image.open(img)) for img in list_n_join[idx:idx+12]]
            img_n_array = np.array(img_n)
            print(img_n_array.shape)
            encoded_n = self.AE.predict(img_n_array)
            print(encoded_n.shape)
            encoded_n_img = []
            for img in np.arange(encoded_n.shape[0]):
                image = Image.fromarray(np.uint8(encoded_n[img,:,:,:]))
                encoded_n_img.append(image)
            
            n_tar = os.path.join(self.tar_path,"n")
            if os.path.exists(n_tar)== False:
                os.makedirs(n_tar)
            n_name_tar = [os.path.join(n_tar,i) for i in list_n]
            for image,name in zip(encoded_n_img,n_name_tar):

                image.save(name)
            zipname = "n_{}_{}.zip".format(idx+12,date.today())
            zip_path = os.path.join(n_tar,zipname)
            if os.path.exists(zip_path) == False:
                zipf = zipfile.ZipFile(zip_path,"w")
                for img in n_name_tar:
                    zipf.write(img)
                zipf.close()
            for img in n_name_tar:
                os.remove(img)

    


