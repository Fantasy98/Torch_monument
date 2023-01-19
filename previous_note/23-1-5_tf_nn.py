#%%
import tensorflow as tf
from tensorflow import keras
from keras import layers
from utils import networks as nn
from utils import loss 
import keras.backend as K
K.clear_session()
HEIGHT = 360;WIDTH=640;CHANNEL=3;LATENT_DIM = 32

#%%
encod = nn.AE_Encoder(HEIGHT,WIDTH,CHANNEL,LATENT_DIM)
decod = nn.AE_Decoder(CHANNEL,LATENT_DIM)
Encoder = nn.Encoder(HEIGHT,WIDTH,CHANNEL,LATENT_DIM)
fe = nn.feature_extractor(HEIGHT,WIDTH,CHANNEL)
d = nn.Descriminator(HEIGHT,WIDTH,CHANNEL,fe)

print(encod.summary())
print(decod.summary())
print(Encoder.summary())
print(d.summary())
input_layer = tf.keras.layers.Input((HEIGHT,WIDTH,CHANNEL))
ae= tf.keras.models.Sequential([encod,decod],name="ae")
# ae = tf.keras.models.Model(input_layer,AE) 
#%%
import numpy as np

optim = keras.optimizers.Adam()
optim1 = keras.optimizers.Adam()

for i in range(20):
    ori = np.random.randint(0,1,(1,HEIGHT,WIDTH,CHANNEL))
    gan = ae(ori)
    with tf.GradientTape() as tape:
        real = d(ori)
        real_zero = np.zeros(real.numpy().shape)
        fake = d(gan)
        fake_one = np.ones(fake.numpy().shape)
        d_loss = keras.losses.binary_crossentropy(real_zero,real)+\
                keras.losses.binary_crossentropy(fake_one,fake)
        

    grad = tape.gradient(d_loss,d.trainable_weights)
    optim.apply_gradients(zip(grad,d.trainable_weights))
            
    
    
    fe = d.get_layer("feature_extractor")
    g_e = ae.get_layer("ae_encoder")
    with tf.GradientTape() as tape:
        f_ori = fe(ori);f_gan = fe(gan)
        adv_loss = loss.AdvLoss(f_ori,f_gan)
        # print(adv_loss)
        cnt_loss = loss.CntLoss(ori,gan)
        # print(cnt_loss)
        enc_ori = Encoder(gan);enc_gan = encod(ori)
        del g_e
        # print(enc_ori)
        # print(enc_gan)
        enc_loss = loss.EncLoss(enc_ori,enc_gan)
        # print(enc_loss)
        gloss = 40*cnt_loss + adv_loss + enc_loss
        # gloss1 = 40*cnt_loss + adv_loss + enc_loss
        print(gloss)
        
    gradient= tape.gradient(gloss,ae.trainable_variables)
    # gradient1= tape.gradient(gloss,decod.trainable_variables[:])
    # gradient_de= tape.gradient(gloss,decod.trainable_weights)
    print(gradient[-2])
    optim1.apply_gradients(zip(gradient,ae.trainable_variables))
    # optim1.apply_gradients(
    #                         (grad, var) 
    #                         for (grad, var) in zip(gradient, ae.trainable_variables) 
    #                         if grad is not None)
    # optim1.apply_gradients(zip(gradient1,decod.trainable_variables[:]))
    # optim.apply_gradients(zip(gradient,decod.trainable_weights))
    

# %
# %%
