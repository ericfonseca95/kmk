# Imports
from typing import Any
import matplotlib.pyplot as plt
from . import utils
import pandas as pd
import time
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from torch.nn import functional as F
# get ModuleList from torch.nn

from torch.nn import ModuleList
from torch.utils.data import Dataset, DataLoader
from torch import optim
# create a cv function
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def TLNN(model, X, change_layers=1):
    new = NN(layers=model.layers, layer_size=model.layer_size,
                n_inputs=model.n_inputs, n_outputs=model.n_outputs)
    test = new(X)
    new.load_state_dict(model.state_dict())
    children = [child for child in new.children()]
    for child in children:
        for param in child.parameters():
            param.requires_grad = False
    total_layers = len(children)
    for i in range(change_layers):
        layer = children[total_layers-i-1]
        layer_params = layer.parameters()
        for p in layer_params:
            p.requires_grad = True
    new.params.update(locals())
    try:
        new.params.pop('kwargs')
    except:
        pass
    try:
        new.parmas.pop('X')
    except:
        pass
    try:
        new.params.pop('self')
    except:
        pass
    return new

def TLCNN(model, X, change_layers=1):
    new = CNN(fc_layers=model.fc_layers, n_outputs=model.n_outputs, 
              fc_size=model.fc_size, n_inputs=model.n_inputs, 
              kernel_size=model.kernel_size, out_channels=model.out_channels, 
              conv_layers=model.conv_layers)
    test = new(X)
    new.load_state_dict(model.state_dict())
    children = [child for child in new.children()]
    for child in children:
        for param in child.parameters():
            param.requires_grad = False
    total_layers = len(children)
    for i in range(change_layers):
        layer = children[total_layers-i-1]
        layer_params = layer.parameters()
        for p in layer_params:
            p.requires_grad = True
    new.params.update(locals())
    try:
        new.params.pop('kwargs')
    except:
        pass
    try:
        new.parmas.pop('X')
    except:
        pass
    try:
        new.params.pop('self')
    except:
        pass
    return new


class NN(nn.Module):
    def __init__(self, n_inputs=106, n_outputs=1, fc_layers=3, fc_size=75, change_layers=0, **kwargs):
        """
        Initialize the NN model with a given number of layers and layer size.
        
        Args:
        - num_classes (int): The number of classes the model should output.
        - layers (int): The number of fully connected layers in the model.
        - layer_size (int): The number of neurons in each fully connected layer.
        """
        super(NN, self).__init__()
        self.estimator_type = 'NN'
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.fc_size = fc_size
        self.fc_layers = fc_layers
        self.change_layers = change_layers
        # set all keys in kwargs as attributes
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.fc1 = nn.Linear(self.n_inputs, self.fc_size)
        self.fcs = ModuleList([nn.Linear(self.fc_size, self.fc_size) for i in range(self.fc_layers)])
        # if layers=0 then we need to make fcs a ModuleList with 0 elements
        if self.fc_layers == 0:
            self.fcs = ModuleList([])
            self.fc1 = nn.Linear(self.n_inputs, self.n_outputs)
        self.fout = nn.Linear(self.fc_size, self.n_outputs)

        self.params = {'estimator_type':self.estimator_type, 'n_inputs': self.n_inputs, 
                       'n_outputs': self.n_outputs, 'fc_layers': self.fc_layers, \
                           'fc_size': self.fc_size, 'change_layers': self.change_layers}
        return 

    def forward(self, x):
        """
        Forward pass of the NN model.
        
        Args:
        - x (torch.Tensor): The input tensor of shape (batch_size, input_dim)
        
        Returns:
        - y (torch.Tensor): The output tensor of shape (batch_size, num_classes)
        """
        if self.fc_layers == 0:
            x = F.relu(self.fc1(x))
        else:
            x = F.relu(self.fc1(x))
            for fc in self.fcs:
                x = F.relu(fc(x))
            x = self.fout(x)
        return x
    
    def update_params(self, config: dict):
        for key in config:
            if hasattr(self, key):
                setattr(self, key, config[key])
        # update config
        self.params.update({i : config[i] for i in config if i in self.params})
        # Re-initialize the model with the updated parameters
        self.__init__(
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            fc_layers=self.fc_layers,
            fc_size=self.fc_size,
            change_layers=self.change_layers
        )
        return self

class CNN(nn.Module):
    def __init__(self, fc_layers=3, n_outputs=1, fc_size=75, n_inputs=52, kernel_size=3, 
                 out_channels=(3, 10), conv_layers=2, **kwargs):
        """
        Initialize the convolutional model with a given number of fc_layers, output size, and layer size.
        
        Args:
        - fc_layers (int): The number of fully connected layers in the model.
        - n_outputs (int): The number of output classes for the model.
        - fc_size (int): The number of neurons in each fully connected layer.
        - n_inputs (int): The number of input features for the model.
        - kernel_size (int): The kernel size for the convolutional layers.
        - out_channels (tuple): The number of output channels for the convolutional layers.
        - conv_layers (int): The number of convolutional layers in the model.
        """
        super(CNN, self).__init__()
        # take all the inputs and make them attributes
        self.estimator_type = 'CNN'
        self.fc_layers = fc_layers
        self.fc_size = fc_size
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.conv_layers = conv_layers
        
        if conv_layers != len(out_channels):
            raise ValueError("The length of `out_channels` must match `conv_layers`.")
        
        self.conv_channels = [1] + list(out_channels)
        
        # Create convolutional layers
        self.cs = ModuleList([
            nn.Conv1d(self.conv_channels[i], self.conv_channels[i+1], kernel_size)
            for i in range(conv_layers)
        ])
        
        # Calculate the output size after convolutional layers
        self.input_fc_size = n_inputs
        for _ in range(conv_layers):
            self.input_fc_size = ((self.input_fc_size - kernel_size) // 1) + 1  # Considering stride=1
            self.input_fc_size = self.input_fc_size // kernel_size  # After average pooling
        
        self.input_fc_size *= self.conv_channels[-1]  # multiply by the number of last conv layer channels
        self.input_fc_size = int(self.input_fc_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.input_fc_size, fc_size)
        self.fcs = ModuleList([nn.Linear(fc_size, fc_size) for _ in range(fc_layers - 1)])
        self.fout = nn.Linear(fc_size, n_outputs)
        
        self.params = {
            'fc_layers': fc_layers, 'fc_size': fc_size, 'n_outputs': n_outputs, 'n_inputs': n_inputs, 
            'kernel_size': kernel_size, 'out_channels': out_channels, 'conv_layers': conv_layers
        }
        
        # Update self with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
            if key in self.params:
                self.params[key] = value
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        
        # Convolutional layers with ReLU and AvgPool
        for conv in self.cs:
            x = F.relu(conv(x))
            x = F.avg_pool1d(x, self.kernel_size)
        
        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1)
        
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        for fc in self.fcs:
            x = F.relu(fc(x))
        
        # Output layer
        x = self.fout(x)
        
        return x

# class CNN(nn.Module):
#     def __init__(self, fc_layers=3, n_outputs=1, fc_size=75, n_inputs=52, kernel_size=3, 
#                  out_channels=(3, 10), conv_layers=2, **kwargs):
#         """
#         Initialize the convolutional model with a given number of fc_layers, output size, and layer size.
        
#         Args:
#         - fc_layers (int): The number of fully connected fc_layers in the model.
#         - n_outputs (int): The number of output classes for the model.
#         - fc_size (int): The number of neurons in each fully connected layer.
#         - n_inputs (int): The number of input features for the model.
#         - kernel_size (int): The kernel size for the convolutional fc_layers.
#         - out_channels (tuple): The number of output channels for the convolutional fc_layers.
#         """
#         super(CNN, self).__init__()
#         self.fc_layers = fc_layers
#         self.estimator_type = 'CNN'
#         self.conv_padding = 0
#         self.fc_size = fc_size
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.n_inputs = n_inputs
#         self.n_outputs = n_outputs
#         self.conv_layers = conv_layers
#         self.conv_channels = [1] + list(out_channels)
#         self.cs = ModuleList([nn.Conv1d(self.conv_channels[i], self.conv_channels[i+1], self.kernel_size) for i in range(len(out_channels)-1)])
#         # get the dimension of the last cs layer
#         self.stride = 1
#         # calculate the output size of conv layers
#         self.input_fc_size = self.n_inputs
#         for i in range(self.conv_layers):
#             self.input_fc_size = ((self.input_fc_size - self.kernel_size + 2*self.conv_padding) / self.stride) + 1

#         # reduce size by factor of kernel size after average pooling in each conv layer
#         self.input_fc_size = self.input_fc_size // self.kernel_size**self.conv_layers

#         self.input_fc_size *= self.conv_channels[-1]  # multiply by the number of last conv layer channels

#         self.input_fc_size = int(self.input_fc_size)

#         self.fc1 = nn.Linear(138, self.fc_size)
#         self.fcs = ModuleList([nn.Linear(self.fc_size, self.fc_size) for i in range(fc_layers-1)])
#         self.fout = nn.Linear(self.fc_size, n_outputs)
#         self.params = {'estimator_type':'CNN', 'fc_layers': self.fc_layers, 'fc_size': self.fc_size, 
#                        'out_channels': self.out_channels, 'kernel_size': self.kernel_size, 
#                        'conv_layers': self.conv_layers,
#                        'n_inputs': self.n_inputs, 'n_outputs': self.n_outputs, 'conv_padding': self.conv_padding}
#          # set all keys in kwargs as attributes
#         for key in kwargs:
#             setattr(self, key, kwargs[key])
            
#         if self.conv_layers != len(self.out_channels):
#             self.out_channels = [out_channels[0] for i in range(conv_layers)]

#         # update params with matching keys in self.__dict__
#         self.params.update({i : self.__dict__[i] for i in self.__dict__ if i in self.params})
    
#     def forward(self,x):
#         rows = x.shape[0]
#         cols = x.shape[1]
#         x = x.reshape(rows, 1, cols)
#         # check if x is on the same device as the model, if not, move it
#         if x.device != self.device:
#             x = x.to(self.device)
#         for conv in self.cs:
#             x = F.relu(conv(x))
#             # pool average
#             x = F.avg_pool1d(x, self.kernel_size)
#         x = x.flatten().reshape(rows, -1)
#         cols = x.shape[1]
#         try:
#             x = F.relu(self.fc1(x))
#         except:
#             try:
#                 self.fc1 = nn.Linear(cols, self.fc_size)
#                 x = F.relu(self.fc1(x))
#             except:
#                 self.fc1 = nn.Linear(cols, self.fc_size).to('cuda')
#                 x = F.relu(self.fc1(x))
#         for fc in self.fcs:
#             x = F.relu(fc(x))
#         x = self.fout(x)
#         return x


    def update_params(self, config: dict):
        for key in config:
            if hasattr(self, key):
                setattr(self, key, config[key])
        # update config
        self.params
        
    def to(self, device):
        self.device = device
        for i in self.fcs:
            i.to(device)
        return super().to(device)


class lstm(nn.Module):
    def __init__(self, n_inputs=5, hidden_size=24, n_outputs=612, n_linear_layers=1, 
                 layer_size=10, seq_len = 204):
        super().__init__()
        self.lstm = nn.LSTM(n_inputs, hidden_size, batch_first=True)
        self.fcs = nn.ModuleList([nn.Linear(layer_size, layer_size) for i in range(n_linear_layers)])
        self.fcs[0] = nn.Linear(hidden_size*seq_len, layer_size)
        self.layer_size = layer_size
        self.hidden_size = hidden_size
        self.n_linear_layers = n_linear_layers 
        self.output = nn.Linear(layer_size, n_outputs)
        
    def forward(self, x):
        
        rows = x.shape[0]
        
        # passes data through LSTM layers
        x, _ = self.lstm(x)
        x = x.reshape(rows, -1)
        
        # Passes data through linear layers
        for i, fc in enumerate(self.fcs):
            x = F.relu(fc(x))
        x = self.output(x)
        return x
    
    def unfreeze_lstm(self):
        lstm_children = self.lstm.children()
        for child in lstm_children:
            for param in child.parameters():
                param.requires_grad = True
 
def TLLSTM(model, X, change_layers=1, unfreeze_lstm=False):
    n_inputs = model.lstm.input_size
    new = lstm(n_linear_layers=model.n_linear_layers, n_outputs=model.output.out_features, layer_size=model.layer_size, hidden_size = model.hidden_size, n_inputs = n_inputs)
    test = new(X)
    new.load_state_dict(model.state_dict())
    children = [child for child in new.children()]
    for child in children:
        for param in child.parameters():
            param.requires_grad = False
    total_layers = len(children)
    for i in range(change_layers):
        layer = children[total_layers-i-1]
        layer_params = layer.parameters()
        for p in layer_params:
            p.requires_grad = True
    if unfreeze_lstm == True:
        new.unfreeze_lstm()
    return new
       
class LSTM(nn.Module):
    def __init__(self, n_lstm_layers=3, n_lstm_outputs=50, 
                 lstm_hidden_size=3, n_inputs=8, n_outputs=3, 
                 n_timesteps=204, n_linear_layers=1, linear_layer_size=50, **kwargs):
        super(LSTM, self).__init__()
        self.lstm_layers = n_lstm_layers
        self.n_lstm_hidden_size = lstm_hidden_size
        self.n_lstm_outputs = n_lstm_outputs
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_timesteps = n_timesteps
        self.n_linear_layers = n_linear_layers
        self.linear_layer_size = linear_layer_size
        self.lstm_output_dim = n_timesteps*n_lstm_layers
        self.estimator_type = 'LSTM'
        
        for key in kwargs:
            self.params[key] = kwargs[key]
            
        self.params = {'estimator_type':'LSTM', 'n_lstm_layers': self.lstm_layers,
                        'n_lstm_outputs': self.n_lstm_outputs, 'lstm_hidden_size': self.n_lstm_hidden_size,
                        'n_inputs': self.n_inputs, 'n_outputs': self.n_outputs, 'n_timesteps': self.n_timesteps,
                        'n_linear_layers': self.n_linear_layers, 'linear_layer_size': self.linear_layer_size,
                        'lstm_output_dim': self.lstm_output_dim}

        
        self.lstm = nn.LSTM(input_size=n_inputs, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=n_lstm_layers)
        # compute the lstm output dim . use the state dict of self.lstm to get the output dim
        self.lstm_output_dim = self.lstm.state_dict()['weight_hh_l0'].shape[1]*n_lstm_layers
        self.linear_layers = nn.ModuleList()
        self.first_linear_layer = nn.Linear(self.lstm_output_dim, linear_layer_size)
        for i in range(n_linear_layers):
            self.linear_layers.append(nn.Linear(linear_layer_size, linear_layer_size))
        self.output_layer = nn.Linear(linear_layer_size, n_outputs*self.n_timesteps)

        try:
            self.params.pop('self')
        except KeyError:
            pass
        return
    
    def forward(self, x):
        x, _ = self.lstm(x)
        # make sure x in flattened
        x = x.flatten()
        x = self.first_linear_layer(x)
        for layer in self.linear_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x.reshape(self.n_timesteps, self.n_outputs)
    
    
    def update_params(self, param_dict):
        for key in param_dict:
            if hasattr(self, key):
                setattr(self, key, param_dict[key])
        
        # Re-initialize the model with the updated parameters
        self.__init__(
            n_lstm_layers=self.n_lstm_layers,
            n_lstm_outputs=self.n_lstm_outputs,
            lstm_hidden_size=self.lstm_hidden_size,
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            n_timesteps=self.n_timesteps,
            n_linear_layers=self.n_linear_layers,
            linear_layer_size=self.linear_layer_size
        )
        
        return self
    
    
  
# lets make another VAE that can have variable number of layers
class VAE(nn.Module):
    # make all the inputs here have default so that we can use kwargs   
    #def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, formula_dim):
    def __init__(self, input_dim=106, hidden_dim=75, latent_dim=50, n_layers=3, binary_dim=3, **kwargs):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.binary_dim = binary_dim
        self.encoder_layers = n_layers  
        self.decoder_layers = n_layers
        # update with kwargs
        for key in kwargs:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
        # update params
        self.params = {}
        # encoder with n_layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.encoder_layers = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for i in range(self.n_layers)])
        self.out1 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.out2 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.encoder = nn.Sequential(
            self.fc1,
            self.encoder_layers,
            self.out1,
            self.out2
        )
        # decoder with n_layers
        self.fc3 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.decoder_layers = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for i in range(self.n_layers)])
        self.out3 = nn.Linear(self.hidden_dim, self.input_dim)
        self.decoder = nn.Sequential(
            self.fc3,
            self.decoder_layers,
            self.out3
        )
        self.dropout = nn.Dropout(p=0.2)
        return 
    
    def encode(self, x):
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        for layer in self.encoder_layers:
            x = F.relu(layer(x))
        # add the dropout
        return self.out1(x), self.out2(x)
    
    def reparameterize(self, mu, logvar):
        # use a bernoulli distribution to sample from the latent space 
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        z = F.relu(self.fc3(z))
        for layer in self.decoder_layers:
            z = F.relu(layer(z))
        out = self.out3(z)
        return out
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def update_params(self, param_dict):
        for key in param_dict:
            if hasattr(self, key):
                setattr(self, key, param_dict[key])
        
        # Re-initialize the model with the updated parameters
        self.__init__(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            n_layers=self.n_layers,
            binary_dim=self.binary_dim,
            **self.kwargs
        )
        
        return self
    
# create an identical VAE class but make it heirarchical
class Hierarch_VAE(nn.Module):
    def __init__(self, input_dim=100, latent_dim=2, n_layers=2, binary_dim=0, **kwargs):
        super(Hierarch_VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        for key in kwargs:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
        
        self.layer_topologies = get_topology(self.input_dim, self.n_layers, self.latent_dim)
        self.fc1 = nn.Linear(self.input_dim, self.layer_topologies[0]) 
        for i in range(self.n_layers+1):
            setattr(self, 'encoder_layer_{}'.format(i), nn.Linear(self.layer_topologies[i], self.layer_topologies[i+1]))
        self.out1 = nn.Linear(self.layer_topologies[-1], self.latent_dim)
        self.out2 = nn.Linear(self.layer_topologies[-1], self.latent_dim)
        self.encoder_layers = [getattr(self, 'encoder_layer_{}'.format(i)) for i in range(self.n_layers)]
        self.fc3 = nn.Linear(self.latent_dim, self.layer_topologies[-1])
        for i in range(self.n_layers+1):
            setattr(self, 'decoder_layer_{}'.format(i), nn.Linear(self.layer_topologies[-i-1], self.layer_topologies[-i-2]))
        self.out3 = nn.Linear(self.layer_topologies[0], self.input_dim) 
        self.decoder_layers = [getattr(self, 'decoder_layer_{}'.format(i)) for i in range(self.n_layers)]
        self.params = {}

    
    def encode(self, x):
        x = F.relu(self.fc1(x))
        for layer in self.encoder_layers:
            x = F.relu(layer(x))
        return self.out1(x), self.out2(x)
    
    def reparameterize(self, mu, logvar):
        # use a bernoulli distribution to sample from the latent space 
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        z = F.relu(self.fc3(z))
        for layer in self.decoder_layers:
            z = F.relu(layer(z))
        out = self.out3(z)
        out[:, :self.binary_dim] = F.softmax(out[:, :self.binary_dim], dim=1)
        return out
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def update_params(self, param_dict):
        for key in param_dict:
            if hasattr(self, key):
                setattr(self, key, param_dict[key])
        
        # Re-initialize the model with the updated parameters
        self.__init__(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            n_layers=self.n_layers,
            binary_dim=self.binary_dim,
            **self.kwargs
        )
        
        return self
    
# this function will return a list of layer topologies for a given number of layers to create a 
# hierachical VAE
def get_topology(input_dim, layers, latent_dim):
    # the first layer will be the input dim, then it should taper down to the latent dim
    # the last layer should be the latent dim
    
    layer_topologies = [input_dim]
    
    # layers should step be dx = (input_dim - latent_dim) / (layers + 1)
    dx = (input_dim - latent_dim) / (layers + 1)
    for i in range(layers):
        layer_topologies.append(int(input_dim - dx*(i+1)))
    layer_topologies.append(latent_dim)
    return layer_topologies
    

class global_input_NN(utils.Trainer):
    def __init__(self, reg, n_layers, layer_size, n_outputs, global_inputs):
        # if optimizer_class not in config set it to Adam
        if 'optimizer_class' not in reg.config.keys():
            reg.config['optimizer_class'] = 'Adam'
        reg.config['verbose'] = 10
        super().__init__() 
        self.reg = reg
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.n_outputs = n_outputs
        self.estimator = self.reg.estimator
        self.global_inputs = global_inputs
        
        for param in self.reg.estimator.parameters():
            param.requires_grad = False
        self.reg.estimator.fcs2 = nn.ModuleList() 
        for i in range(self.n_layers):
            if i == 0:
                self.reg.estimator.fcs2.append(nn.Linear(self.global_inputs + self.reg.layer_size, self.layer_size))
            else:
                self.reg.estimator.fcs2.append(nn.Linear(self.layer_size, self.layer_size))
        self.reg.estimator.fout = nn.Linear(self.layer_size, self.n_outputs)
        self.reg.estimator.forward = self.forward
        
    
    def forward(self, x):
        # first pass the data through the original fcs and model
        x1, x2 = x[:, :self.global_inputs], x[:,self.global_inputs:]
        # x2 contains the global inputs
        x1 = self.reg.estimator.fc1(x1)
        for fc in self.reg.estimator.fcs:
            x = fc(x1)
        # pass the global inputs through the new fcs and the embedding from the original model
        x2 = torch.cat([x2, x1], dim=1)
        for fc in self.reg.estimator.fcs2:
            x2 = fc(x2)
        x2 = self.reg.estimator.fout(x2)
        return x2
    # make it so that this replaces the forward method in the estimator
    # make it so that the new forward method takes in the global inputs and the original inputs
    def __call__(self, x):
        return self.forward(x)
        



# We are creating a new class 'global_model', which is a PyTorch module (nn.Module).
class global_model(nn.Module):
    
    # Here, we're initializing our class with a pre-trained model (ptmodel), 
    # size of the layers in the model (layer_size), number of layers (n_layers),
    # and the number of global inputs that the model will accept (n_global_inputs).
    def __init__(self, ptmodel, layer_size=5, n_layers=1, n_global_inputs=2, freeze_ptmodel=True, change_layers=1):
        # Call the parent (nn.Module) constructor.
        super(global_model, self).__init__()

        # Set the pre-trained model, the number of global inputs, 
        # and the output dimensionality of the pre-trained model.
        self.ptmodel = deepcopy(ptmodel)
        self.n_global_inputs = n_global_inputs
        self.ptmodel_output_dim = ptmodel.output.out_features
        self.n_ptmodel_layers = len(ptmodel.fcs)
        
        # freeze the ptmodel parameters
        if freeze_ptmodel:
            # first freeze everything
            for child in self.ptmodel.children():
                for param in child.parameters():
                    param.requires_grad = False
            # unfreeze the last "change layers" in self.ptmodel.fcs + self.ptmodel.output.
            # if change_layers == 1, then we only unfreeze the self.ptmodel.output
            # if change_layers > 1, then we start unfreezing layers in self.ptmodel.fcs
            for i in range(change_layers-1):
                layer_number = self.n_ptmodel_layers - i - 1
                layer = self.ptmodel.fcs[layer_number]
                layer_params = layer.parameters()
                for p in layer_params:
                    p.requires_grad = True 
            if change_layers > 0:
                for param in self.ptmodel.output.parameters():
                    param.requires_grad = True
        else:
            for child in self.ptmodel.children():
                for param in child.parameters():
                    param.requires_grad = True
                
        
        # Define the first linear layer which will receive the concatenated 
        # global inputs and the output of the pre-trained model.
        self.input = nn.Linear(n_global_inputs + self.ptmodel_output_dim, layer_size)
        # Define the inner fully connected layers of the model.
        self.fcs = nn.ModuleList([nn.Linear(layer_size, layer_size) for i in range(n_layers)])

        # Define the final output layer which will have the same size 
        # as the output of the pre-trained model.
        self.output = nn.Linear(layer_size, self.ptmodel.output.out_features)
        
        # Get which layers requires grad
        pnames, gradstates = [], []
        for paramname,param in zip(self.ptmodel.state_dict(),self.ptmodel.parameters()):
            pnames.append(paramname)
            gradstates.append(param.requires_grad)
        # Format pnames and gradstates into a dataframe
        grad_df = pd.DataFrame()
        grad_df['PT Model Parameter'] = pnames
        grad_df['Requires Grad'] = gradstates
        self.grad_df = grad_df
        
    
    # Define the forward pass of the model.
    def forward(self, x_ptmodel, x_global):
        rows = x_ptmodel.shape[0]
        # Reshape x_global to ensure it is a 2D tensor.
        x_global = x_global.reshape(x_global.shape[0], -1)


        # Feed x_ptmodel into the pre-trained model and store the output.
        #x_ptmodel = self.ptmodel(x_ptmodel)
        # pass the ptmodel.lstm
        x_ptmodel, (h, c) = self.ptmodel.lstm(x_ptmodel)
        x_ptmodel = x_ptmodel.reshape(rows, -1)
        # pass the ptmodel.fc
        for fc in self.ptmodel.fcs:
            x_ptmodel = F.relu(fc(x_ptmodel))
        
        x_ptmodel = self.ptmodel.output(x_ptmodel)
        
        # Concatenate the output of the pre-trained model with the global inputs.
        x = torch.cat((x_ptmodel, x_global), dim=1)
        # Apply the first layer (input) to x and then apply a ReLU activation function.
        x = F.relu(self.input(x))

        # For each fully connected layer, apply the layer to x and 
        # then apply a ReLU activation function.
        for i, fc in enumerate(self.fcs):
            x = F.relu(fc(x))

        # Apply the output layer to x.
        x = self.output(x)

        # Return the output of the model.
        return x

    # Define a function to return the output of the pre-trained model.
    def get_ptmodel_output(self, x_ptmodel):
        # Feed x_ptmodel into the pre-trained model and return the output.
        return self.ptmodel(x_ptmodel)
