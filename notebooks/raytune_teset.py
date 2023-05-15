import kmodels as kmk
import torch 
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from ray import tune
import ray
import os
from torch.utils.data import TensorDataset
from ray.train.torch import TorchTrainer
from ray.air import ScalingConfig
def scale_data(x, binary_dim=0):
    if binary_dim>0:
        bin_scaler = MinMaxScaler()
        float_scaler = StandardScaler()
        x_bin = bin_scaler.fit_transform(x[:, :binary_dim])
        x_float = float_scaler.fit_transform(x[:, binary_dim:])
        print('binary scaler: ', x_bin.shape)
        print('float scaler: ', x_float.shape)
        x = np.concatenate((x_bin, x_float), axis=1)
    else:
        bin_scaler = None
        float_scaler = StandardScaler()
        x = float_scaler.fit_transform(x)
        print('float scaler: ', x.shape)
    return x, bin_scaler, float_scaler

def train_model(config: dict):
    trainer = kmk.Trainer(**{'binary_dim':0, 'binary_dim_y':0, 
                           'model_type':'NN', 'epochs':11,
                           'lr_init':1e-3, 'batch_size':1000,
                           'beta':0, 'gamma':0, 'layers':5, 
                           'layer_size':100
                           }
                          )
    trainer.set_params(**config)
    trainer.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    return trainer

def make_data(n=1000, n_features=10, binary_dim=0, float_dim=10):
    float_dim = n_features - binary_dim
    x_bin = np.random.randint(2, size=(n, binary_dim))
    x_float = np.random.rand(n, float_dim)
    x = np.concatenate((x_bin, x_float), axis=1)
    y = np.sin(x[:,0]) + np.cos(x[:,1])

    x, bin_scaler, float_scaler = scale_data(x, binary_dim=binary_dim)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    dataset = TensorDataset(x, y)
    return dataset, bin_scaler, float_scaler

# create the  main instance

def __main__():
    # get the logging
    ray.init(num_cpus=1, log_to_driver=False)
    # make the data
    dataset, bin_scaler, float_scaler = make_data(n=1000, n_features=10, binary_dim=0, float_dim=10)
    # For GPU Training, set `use_gpu` to True.
    # trainer = Trainer(backend="torch", num_workers=2, use_gpu=True)
    scaling_config = ScalingConfig(train_model, num_workers=1)
    config = {}
    trainer = TorchTrainer(train_loop_per_worker=train_model, train_loop_config=config)
    trainer.fit()
    return trainer

if __name__ == "__main__": 
    trainer = __main__()
    print(trainer)