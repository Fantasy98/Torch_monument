class nn_config():
    """
    Configuration for parameter of neural network
    """
    CHANNEl = 3
    HEIGHT = 360
    WIDTH = 640
    knsize = [5,3]
    feature_dims = [32,64,128]
    zdim = 32

class train_config():
    """
    Configuration for training
    """
    random_seed = 1024
    if_random = True
    lr = 1e-3
    eps = 1e-7
    Batch_Size = 16
    loss_weight = {"cnt_loss":40.0,
                    "adv_loss":1.0,
                    "enc_loss":1.0}
    