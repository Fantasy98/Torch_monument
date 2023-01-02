
import os
from tensorflow import keras

class models():
    def __init__(self,path:str):
        self.path = path
        
        self.ae_path = os.path.join(path,"AE.h5")
        self.ae_weight_path = os.path.join(path,"AE_weights.h5")
        self.d_path = os.path.join(path,"Discriminator.h5")
        self.d_weight_path = os.path.join(path,"Discriminator_weights.h5")
        self.Enc_path = os.path.join(path,"Encoder.h5")
        self.Enc_weight_path = os.path.join(path,"Encoder_weights.h5")
        
    def get_AE(self):
        AE = keras.models.load_model(self.ae_path)
        AE.load_weights(self.ae_weight_path)
        return AE
    
    def get_d(self):
        d = keras.models.load_model(self.d_path)
        d.load_weights(self.d_weight_path)
        return d

    def get_Enc(self):
        Enc = keras.models.load_model(self.Enc_path)
        Enc.load_weights(self.Enc_weight_path)
        return Enc
        