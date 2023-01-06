import tensorflow as tf 
from tensorflow import keras
from keras import layers

def AdvLoss(f_ori,f_gan):
    # fe = d.d.get_layer("feature_extractor")
    
    return tf.reduce_mean(tf.square(f_ori-f_gan))


def CntLoss(ori,gan):
    return tf.reduce_mean(tf.reduce_sum(keras.losses.MAE(ori,gan)))

def EncLoss(enc_ori,enc_gan):
    return tf.reduce_sum(keras.losses.MSE(enc_ori,enc_gan))