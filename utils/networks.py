import tensorflow as tf
from tensorflow import keras
from keras import layers


## Try larger strides and kernel and more CNN layer to make sure when it is flatten the parameter wont be too much !
def AE_Encoder(HEIGHT,WIDTH,CHANNEL,LATENT_DIM):
    encoder_inputs = keras.Input(name="Encoder_inputs",shape = (HEIGHT,WIDTH,CHANNEL))
    
    x = layers.Conv2D(32,(5,5),strides=(2,2),padding ="same",name= "conv_1")(encoder_inputs)


    x = layers.Conv2D(64,(3,3),strides=(2,2),padding ="same",name= "conv_2")(x)

    x = layers.Conv2D(128,(3,3),strides=(2,2),padding ="same",name= "conv_3")(x)

    x = layers.Flatten()(x)

    x = layers.Dense(128,activation="relu",name="encoder_dense2")(x)
    
    x = layers.Dense(LATENT_DIM,name= "hyper_dense")(x)

    return keras.Model(inputs =encoder_inputs,outputs =x,name="ae_encoder")


def AE_Decoder(CHANNEL,LATENT_DIM):
  latent_inputs= layers.Input(shape=(LATENT_DIM,),name="latent_input")
  x = layers.Dense(128,activation="relu",name="decoder_dense1")(latent_inputs)
  x= layers.Dense(45*80*128,activation="relu",name="decoder_dense3")(x)

  x = layers.Reshape((45,80,128))(x)

  x = layers.Conv2DTranspose(128,(3,3),strides=(2,2),padding="same",activation="relu",name="decoder_T2D_1")(x)

  x = layers.Conv2DTranspose(64,(3,3),strides=(2,2),padding="same",activation="relu",name="decoder_T2D_2")(x)
  
  x = layers.Conv2DTranspose(32,(3,3),strides=(2,2),padding="same",activation="relu",name="decoder_T2D_3")(x)
  
  decoder_outputs = layers.Conv2D(CHANNEL,(3,3),strides=(1,1),padding="same",name="aede_out")(x)

  return keras.Model(latent_inputs,decoder_outputs,name="ae_decoder")


## Try larger strides and kernel and more CNN layer to make sure when it is flatten the parameter wont be too much !
def Encoder(HEIGHT,WIDTH,CHANNEL,LATENT_DIM):
    encoder_inputs = keras.Input(name="Encoder_inputs",shape = (HEIGHT,WIDTH,CHANNEL))
    
    x = layers.Conv2D(32,(5,5),strides=(2,2),padding ="same",name= "conv_1")(encoder_inputs)
    x = layers.Conv2D(64,(3,3),strides=(2,2),padding ="same",name= "conv_2")(x)

    x = layers.Conv2D(128,(3,3),strides=(2,2),padding ="same",name= "conv_3")(x)
    x = layers.Flatten()(x)

    x = layers.Dense(128,activation="relu",name="encoder_dense2")(x)    
    x = layers.Dense(LATENT_DIM,name= "hyper_dense")(x)
    return keras.Model(inputs =encoder_inputs,outputs =x,name="encoder")

def feature_extractor(HEIGHT,WIDTH,CHANNEL):
    input_layer = layers.Input(name="extractor_input",shape=(HEIGHT,WIDTH,CHANNEL))
    x = layers.Conv2D(32,(5,5),strides=(2,2),padding ="same",name= "extractor_conv_1",kernel_regularizer="l2")(input_layer)
    x = layers.BatchNormalization(name="extractor_norm_1")(x)
    x = layers.LeakyReLU(name="extractor_leaky_1")(x)
    # x = layers.MaxPooling2D()(x)


    x = layers.Conv2D(64,(3,3),strides=(2,2),padding ="same",name= "extractor_conv_2",kernel_regularizer="l2")(x)
    x = layers.BatchNormalization(name="extractor_norm_2")(x)
    x = layers.LeakyReLU(name="extractor_leaky_2")(x)

    # x = layers.MaxPooling2D()(x)


    x = layers.Conv2D(128,(3,3),strides=(2,2),padding ="same",name= "extractor_conv_3",kernel_regularizer="l2")(x)
    x = layers.BatchNormalization(name="extractor_norm_3")(x)
    x = layers.LeakyReLU(name="extractor_leaky_3")(x)
    
    
    return keras.Model(input_layer,x,name="feature_extractor")

def Descriminator(HEIGHT,WIDTH,CHANNEL,fe):
  input_layer = layers.Input(shape=(HEIGHT,WIDTH,CHANNEL))

  x = fe(input_layer)
  x = layers.GlobalAveragePooling2D(name="glb_avg")(x)
    
  x = layers.Dense(1,activation = "sigmoid",name="d_output")(x)
  return keras.Model(input_layer,x,name="discriminator")

