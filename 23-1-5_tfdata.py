#%%
import tensorflow as tf 
from tensorflow import keras
import keras.backend as K 
from keras import layers
import matplotlib.pyplot as plt 
import os 
import numpy as np
import random

# %%
basic_path = "/home/yuning/DL/Monument/Train/basic"
expo_path = "/home/yuning/DL/Monument/Train/exposure"
white_path = "/home/yuning/DL/Monument/Train/whitebalance"
# %%
basic_imags_path = [os.path.join(basic_path,i) for i in os.listdir(basic_path)   ]
expo_imags_path = [os.path.join(expo_path,i) for i in os.listdir(expo_path)   ]
white_imags_path = [os.path.join(white_path,i) for i in os.listdir(white_path)   ]
# %%
from keras.utils.image_utils import load_img,img_to_array
all_path = basic_imags_path + expo_imags_path + white_imags_path
# %%
tensor = [ img_to_array(load_img(i))/255.0 for i in all_path ]
# %%
dataset =  tf.data.Dataset.from_tensor_slices(tensor)
# %%
tf.data.experimental.save(dataset,"/home/yuning/DL/Monument/tfdata")
# %%
