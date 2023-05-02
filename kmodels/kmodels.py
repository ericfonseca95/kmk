# Imports
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler
import torch
from . import utils
import torch.nn as nn
from torch.nn import functional as F
# get ModuleList from torch.nn
from torch.nn import ModuleList
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
    return new

def TLLSTM(model, X, change_layers=1):
    new = LSTM(n_lstm_layers=model.lstm_layers, n_lstm_outputs=model.n_lstm_outputs,
               lstm_hidden_size=model.n_lstm_hidden_size, n_inputs=model.n_inputs,
               n_timesteps=model.n_timesteps, n_linear_layers=model.n_linear_layers,
               linear_layer_size=model.linear_layer_size)
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
    return new



class NN(nn.Module):
    def __init__(self, n_inputs=106, n_outputs=1, layers=3, layer_size=75):
        """
        Initialize the NN model with a given number of layers and layer size.
        
        Args:
        - num_classes (int): The number of classes the model should output.
        - layers (int): The number of fully connected layers in the model.
        - layer_size (int): The number of neurons in each fully connected layer.
        """
        super(NN, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.layer_size = layer_size
        self.layers = layers
        
        self.fc1 = nn.Linear(n_inputs, layer_size)
        self.fcs = ModuleList([nn.Linear(layer_size, layer_size) for i in range(layers)])
        self.fout = nn.Linear(layer_size, n_outputs)

    def forward(self, x):
        """
        Forward pass of the NN model.
        
        Args:
        - x (torch.Tensor): The input tensor of shape (batch_size, input_dim)
        
        Returns:
        - y (torch.Tensor): The output tensor of shape (batch_size, num_classes)
        """
        try:
            x = F.relu(self.fc1(x))
        except:
            try:
                self.fc1 = nn.Linear(x.shape[1], self.layer_size)
                x = F.relu(self.fc1(x))
            except:
                self.fc1 = nn.Linear(x.shape[1], self.layer_size).to('cuda')
                x = F.relu(self.fc1(x))
        for fc in self.fcs:
            x = F.relu(fc(x))
        x = self.fout(x)
        return x
    
class LSTM(nn.Module):
    def __init__(self, n_lstm_layers=3, n_lstm_outputs=50, 
                 lstm_hidden_size=3, n_inputs=8, n_outputs=3, 
                 n_timesteps=204, n_linear_layers=1, linear_layer_size=50):
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
        
        self.lstm = nn.LSTM(input_size=n_inputs, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=n_lstm_layers)
        
        self.linear_layers = nn.ModuleList()
        self.first_linear_layer = nn.Linear(self.lstm_output_dim, linear_layer_size)
        for i in range(n_linear_layers):
            self.linear_layers.append(nn.Linear(linear_layer_size, linear_layer_size))
        self.output_layer = nn.Linear(linear_layer_size, n_outputs*self.n_timesteps)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        # make sure x in flattened
        x = x.flatten()
        x = self.first_linear_layer(x)
        for layer in self.linear_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x.reshape(self.n_timesteps, self.n_outputs)
    
    
  
# lets make another VAE that can have variable number of layers
class VAE(nn.Module):
    # make all the inputs here have default so that we can use kwargs   
    #def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, formula_dim):
    def __init__(self, input_dim=106, hidden_dim=75, latent_dim=50, n_layers=3, binary_dim=3):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.binary_dim = binary_dim
        self.encoder_layers = n_layers  
        self.decoder_layers = n_layers
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
    
    
# create an identical VAE class but make it heirarchical
class Hierach_VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, n_layers):
        super(Hierach_VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.layer_topologies = get_topology(self.input_dim, self.n_layers, self.latent_dim)
        self.fc1 = nn.Linear(self.input_dim, self.layer_topologies[0]) 
        for i in range(self.n_layers):
            setattr(self, 'encoder_layer_{}'.format(i), nn.Linear(self.layer_topologies[i], self.layer_topologies[i+1]))
        self.out1 = nn.Linear(self.layer_topologies[-1], self.latent_dim)
        self.out2 = nn.Linear(self.layer_topologies[-1], self.latent_dim)
        self.encoder_layers = [getattr(self, 'encoder_layer_{}'.format(i)) for i in range(self.n_layers)]
        self.fc3 = nn.Linear(self.latent_dim, self.layer_topologies[-1])
        for i in range(self.n_layers):
            setattr(self, 'decoder_layer_{}'.format(i), nn.Linear(self.layer_topologies[-i-1], self.layer_topologies[-i-2]))
        self.out3 = nn.Linear(self.layer_topologies[0], self.input_dim) 
        self.decoder_layers = [getattr(self, 'decoder_layer_{}'.format(i)) for i in range(self.n_layers)]
        
    def encode(self, x):
        x = F.relu(self.fc1(x))
        for layer in self.encoder_layers:
            x = F.relu(layer(x))
        return self.out1(x), self.out2(x)
    # we need to change the reparameterization function to use a bernoulli distribution for the latent space   
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        z = F.relu(self.fc3(z))
        for layer in self.decoder_layers:
            z = F.relu(layer(z))
        return F.relu(self.out3(z))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar