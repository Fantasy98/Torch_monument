#%%
import tensorflow as tf 

dataset = tf.data.experimental.load("/home/yuning/DL/Monument/tfdata")
# %%
sample = iter(dataset).next()
# %%
npsample = sample.numpy()
# %%
from tensorflow import keras
from keras.utils.image_utils import array_to_img
import matplotlib.pyplot as plt
imag = array_to_img(npsample)
plt.figure(0)
plt.imshow(imag)
# %%
