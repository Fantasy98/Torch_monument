import torch 
from torch import nn 
from src.config import nn_config
###### A package for all models used in GANomlay 


def net_ge(device):
    height,width,channels = nn_config.HEIGHT,nn_config.WIDTH,nn_config.CHANNEl
    knsize, feature_dim,zdim = nn_config.knsize,nn_config.feature_dims,nn_config.zdim
    net_ge = g_e(height,width,channels,
                    device,
                    knsize,zdim,feature_dim)
    net_ge.apply(Init_Conv)
    net_ge.to(net_ge.device)
    return net_ge

def net_gd(device):
    height,width,channels = nn_config.HEIGHT,nn_config.WIDTH,nn_config.CHANNEl
    knsize, feature_dim,zdim = nn_config.knsize,nn_config.feature_dims,nn_config.zdim
    net_gd = g_d(height,width,channels,
                device,
                knsize,zdim,feature_dim)
    net_gd.apply(Init_Conv)
    net_gd.to(net_gd.device)
    return net_gd

def net_dis(device):
    height,width,channels = nn_config.HEIGHT,nn_config.WIDTH,nn_config.CHANNEl
    knsize, feature_dim,zdim = nn_config.knsize,nn_config.feature_dims,nn_config.zdim
    net_dis = dis(height,width,channels,
                    device,
                    knsize,zdim,feature_dim)
    net_dis.apply(Init_Conv)
    net_dis.to(net_dis.device)
    return net_dis



def Init_Conv(m):
    """
    A function for initializing conv and convtranspose layer 
    The weight will be init by xavier_uniform and bias will be init by zeros
    Example:
        model = FCN()
        model.apply(Init_Conv)
    """
    
    print(f"Checking: {m}",flush=True)
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        # print(f"the layer is {m.__class__.__name__}")
        nn.init.xavier_uniform_(m.weight,gain=nn.init.calculate_gain("leaky_relu"))
        nn.init.zeros_(m.bias)
        print("It has been initialized",flush=True)
    


class Flatten(nn.Module):
    def forward(self,input:torch.Tensor):
        # Reserve the batchsize and flatten other dimension
        return input.view(input.size(0),-1)



# 1 Encoder 
class g_e(nn.Module):
    # Inputs:
    # height ,width, channels:  Image shape 
    # knsize  kernel size
    # zdim: Dimension of latent vector
    # device: "cuda"
    def __init__(self,height,width,channels,\
                    device,\
                    knsize:list,zdim,feature_dims:list):
        super(g_e,self).__init__()
        
        self.height,self.width,self.channels = height,width,channels
        self.knsize, self.feature_dims = knsize,feature_dims
        self.zdim = zdim
        self.device = device

        # The convolution layer for encoder
   
        g_conv = nn.Sequential()
            
        g_conv.add_module("conv1", nn.Conv2d(in_channels=channels, out_channels=feature_dims[0],
                                                kernel_size=knsize[0],stride=2))
        g_conv.add_module("BN{}".format(1),
                                nn.BatchNorm2d(num_features=feature_dims[0],eps=1e-3,momentum=0.99) )
        g_conv.add_module("LkELU{}".format(1),
                                nn.LeakyReLU())
        for i, out_channel in enumerate(feature_dims[:-2]):
            g_conv.add_module("conv{}".format(i+2),
                                nn.Conv2d(in_channels=out_channel,out_channels=feature_dims[i+1],
                                            kernel_size=knsize[-1],stride=2))
            g_conv.add_module("BN{}".format(i+2),
                                nn.BatchNorm2d(num_features=feature_dims[i+1],eps=1e-3,momentum=0.99) )
            g_conv.add_module("LkELU{}".format(i+2),
                                nn.LeakyReLU())
            
        g_conv.add_module("conv{}".format(len(feature_dims)), nn.Conv2d(in_channels=feature_dims[-2], 
                                                                        out_channels=feature_dims[-1],
                                                                        kernel_size=knsize[-1],stride=2,padding=(knsize[-1]-1,knsize[-1]-1),padding_mode="replicate"))
        
        # Flatten and dense output
        # The times of being convlouated
        n_conv = len(self.feature_dims)
        # Final dimension of features
        n_feature = self.feature_dims[-1]
        # Compute the input dimension for mlp
        conv_flatten_dim = (self.height//(2**n_conv)) * (self.width//(2**n_conv)) * n_feature
        
        g_mlp = nn.Sequential()
        g_mlp.add_module("flatten",Flatten())
        g_mlp.add_module("dense1",
                        nn.Linear(in_features=conv_flatten_dim, out_features= 128))
        g_mlp.add_module("relu1", nn.ReLU())
        g_mlp.add_module("dense2",
                        nn.Linear(in_features=128, out_features= 32))
        g_mlp.add_module("relu2", nn.ReLU())
        
       
        self.g_conv = g_conv
        self.g_mlp = g_mlp
    
    
    def forward(self,inputs):
            conv_output = self.g_conv(inputs)
            output = self.g_mlp(conv_output)
            return output


##### 2 Decoder
class g_d(nn.Module):
    def __init__(self,height,width,channels,\
                    device,\
                    knsize:list,zdim,feature_dims:list):
        super(g_d,self).__init__()
        self.height,self.width,self.channels = height,width,channels
        self.knsize, self.feature_dims = knsize,feature_dims
        self.zdim = zdim
        self.device = device

        n_conv = len(self.feature_dims)
        # Final dimension of features
        n_feature = self.feature_dims[-1]
        # Compute the input dimension for mlp
        self.reheight = self.height//(2**n_conv)
        self.rewidth = (self.width//(2**n_conv))

        conv_flatten_dim = ( self.reheight * self.rewidth * n_feature)
        
        d_mlp = nn.Sequential()
        d_mlp.add_module("mlp1",
                            nn.Linear(in_features=zdim,out_features=128))
        d_mlp.add_module("mlp_relu1",nn.ReLU())

        d_mlp.add_module("mlp2",
                            nn.Linear(in_features=128,out_features=conv_flatten_dim))
        d_mlp.add_module("mlp_relu2",nn.ReLU())
        
      
        
        d_conv = nn.Sequential()
        # To realize the "same" padding like in keras, 
        # the input_padding and output_padding should be the same 
        d_conv.add_module("TransConv{}".format(1),
                            nn.ConvTranspose2d(in_channels=feature_dims[-1],out_channels=feature_dims[-2],
                                                kernel_size=knsize[-1],stride=(2,2),padding=(1,1)
                                                ,output_padding=(1,1),padding_mode="zeros"))
        d_conv.add_module("BN{}".format(1),
                                nn.BatchNorm2d(num_features=feature_dims[-2],eps=1e-3,momentum=0.99))
        d_conv.add_module("lkelu{}".format(1),
                                    nn.LeakyReLU())      

        d_conv.add_module("TransConv{}".format(2),
                            nn.ConvTranspose2d(in_channels=feature_dims[-2],out_channels=feature_dims[-3],
                                                kernel_size=knsize[-1],stride=(2,2)
                                                ,padding=(1,1),output_padding=(1,1),padding_mode="zeros"))
        d_conv.add_module("BN{}".format(2),
                                nn.BatchNorm2d(num_features=feature_dims[-3],eps=1e-3,momentum=0.99))
        d_conv.add_module("lkelu{}".format(2),
                                nn.LeakyReLU())

        d_conv.add_module("Channels",
                            nn.ConvTranspose2d(in_channels=feature_dims[0],out_channels=3,
                                                kernel_size=knsize[-1],
                                                stride=(2,2),
                                                padding=(1,1),output_padding=(1,1),padding_mode="zeros"))
        

        self.d_conv = d_conv
        self.d_mlp = d_mlp

    def forward(self,inputs):
        dense_out = self.d_mlp(inputs)
        dense_reshape = dense_out.view(dense_out.size(0),self.feature_dims[-1],self.reheight,self.rewidth)
        conv_out = self.d_conv(dense_reshape)
        return conv_out


class dis(nn.Module):
    def __init__(self,height,width,channels,\
                    device,\
                    knsize:list,zdim,feature_dims:list):
        super(dis,self).__init__()
        self.height,self.width,self.channels = height,width,channels
        self.knsize, self.feature_dims = knsize,feature_dims
        self.zdim = zdim
        self.device = device

        n_conv = len(self.feature_dims)
        # Final dimension of features
        n_feature = self.feature_dims[-1]
        # Compute the input dimension for mlp
        self.reheight = self.height//(2**n_conv)
        self.rewidth = (self.width//(2**n_conv))
        dis_conv = nn.Sequential()
        
        dis_conv.add_module("conv1", nn.Conv2d(in_channels=channels, out_channels=feature_dims[0],
                                                kernel_size=knsize[0],stride=2))
        
        dis_conv.add_module("BN1",nn.BatchNorm2d(num_features=[feature_dims[0]],eps=1e-3,momentum=0.99))
        dis_conv.add_module("Leaky1",nn.LeakyReLU())

        for i, out_channel in enumerate(feature_dims[:-2]):
            dis_conv.add_module("conv{}".format(i+2),
                                nn.Conv2d(in_channels=out_channel,out_channels=feature_dims[i+1],
                                            kernel_size=knsize[-1],stride=2))
            dis_conv.add_module("BN{}".format(i+2),nn.BatchNorm2d(num_features=[feature_dims[i+1]],eps=1e-3,momentum=0.99))
            dis_conv.add_module("Leaky{}".format(i+2),nn.LeakyReLU())

        dis_conv.add_module("conv{}".format(len(feature_dims)), nn.Conv2d(in_channels=feature_dims[-2], 
                                                                        out_channels=feature_dims[-1],
                                                                        kernel_size=knsize[-1],stride=2,padding=(knsize[-1]-1,knsize[-1]-1)))
        
        dis_conv.add_module("conv{}".format(len(feature_dims)+1), nn.Conv2d(in_channels=feature_dims[-1], 
                                                                        out_channels=1,
                                                                        kernel_size=knsize[-1],stride=2,padding=(knsize[-1]-1,knsize[-1]-1)))
        
        dis_out = nn.Sequential()
        dis_out.add_module("GlobAvgPool",nn.AdaptiveAvgPool2d(output_size=(1)))
        dis_out.add_module("Sigmoid",nn.Sigmoid())
        
        self.dis_conv = dis_conv
        self.dis_out = dis_out
    def initial(self):
        for parameter in  self.dis_conv.parameters():
            if parameter.names == "conv":
                nn.init.xavier_uniform_(parameter,gain=nn.init.calculate_gain("leaky_relu"))
    def forward(self,inputs):
        model_out = self.dis_conv(inputs)

        output =self.dis_out(model_out)
        return output
    