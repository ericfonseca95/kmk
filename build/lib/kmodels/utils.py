# Imports
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.nn import functional as F
# get ModuleList from torch.nn
from torch.nn import ModuleList
# create a cv function
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# get base form sklearn
from sklearn.base import BaseEstimator, RegressorMixin

# train the VAE
from torch.functional import F
from torch import optim
from sklearn.linear_model import LinearRegression
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
import time
import gc

# lets get the MultiLRstep from torch to schedule the learning rate
from torch.optim.lr_scheduler import MultiStepLR
from . import kmodels as models
 
def to_torch(X):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X.astype(np.float32))
    elif isinstance(X, pd.DataFrame):
        X = X.values
        X = torch.from_numpy(X.astype(np.float32))
    return X
def to_plot_var(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    return x

class LR_scheduler():
    def __init__(self, optimizer, milestones, gamma):
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
    def step(self, epoch):
        if epoch in self.milestones:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.gamma
        self.optimizer.step()
        self.optimizer.zero_grad()
    

class VAE_loss(nn.Module):
    def __init__(self, binary_dim):
        super(VAE_loss, self).__init__()
        self.binary_dim = binary_dim
        self.reg_loss = torch.tensor([])  # initialize as an empty tensor

    def forward(self, recon_x, x, mu, logvar):
        if self.binary_dim > 0:
            binary_loss = F.binary_cross_entropy(recon_x[:, :self.binary_dim], x[:, :self.binary_dim], reduction='sum')
            scaler_loss = F.mse_loss(recon_x[:, self.binary_dim:], x[:, self.binary_dim:], reduction='sum')
        else:
            binary_loss = 0
            scaler_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        reconstruction_loss = binary_loss + scaler_loss
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = reconstruction_loss + KLD
        self.reg_loss = torch.cat((self.reg_loss, torch.tensor([loss.item()]).to(self.reg_loss.device)))  # add new loss to the tensor
        return loss

    def to(self, device):
        self.device = device
        self.reg_loss = self.reg_loss.to(device)  # move tensor to device
        return self

class Regression_loss(nn.Module):
    def __init__(self, reg_factor):
        super(Regression_loss, self).__init__()
        self.reg_factor = reg_factor
        self.reg_loss = []
    def __call__(self, z, y):
        # save the model
        self.reg = LinearRegression().fit(z.detach().cpu().numpy(), y.detach().cpu().numpy())
        y_pred = self.reg.predict(z.detach().cpu().numpy())
        self.reg_loss.append(F.mse_loss(torch.from_numpy(y_pred), y))
        return self.reg_loss[-1]
    def clear_vars(self):
        self.reg_loss = []
        self.reg = None
    def __iter__(self):
        return iter(self.reg_loss)
    def forward(self, z, y):
        return self.__call__(z, y)
    
class L2_regularization(nn.Module):
    def __init__(self, gamma):
        super(L2_regularization, self).__init__()
        self.gamma = gamma
        self.reg_loss = []
    def __call__(self, model):
        for param in model.parameters():
            self.reg_loss.append(0.5 * self.gamma * torch.sum(torch.pow(param, 2)))
        return self.reg_loss[-1]
    def forward(self, model):
        return self.__call__(model)

class L1_regularization(nn.Module):
    def __init__(self, beta):
        super(L1_regularization, self).__init__()
        self.beta = beta
        self.reg_loss = []
    def __call__(self, model):
        for param in model.parameters():
            self.reg_loss.append(self.beta * torch.sum(torch.abs(param)))
        return self.reg_loss[-1]
    def __iter__(self):
        return iter(self.reg_loss)
    def forward(self, model):
        return self.__call__(model)
    

class MSE_Loss(nn.Module):
    def __init__(self, binary_dim_y=0):
        super(MSE_Loss, self).__init__()
        self.binary_dim_y = binary_dim_y
        # for the binary variables we want to use binary cross entropy
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.reg_loss = torch.tensor([])  # initialize as an empty tensor
        
    def __call__(self, x, y):
        if self.binary_dim_y > 0:
            self.binary_loss = self.bce(x[:, :self.binary_dim_y], y[:, :self.binary_dim_y])
            self.scaler_loss = self.mse(x[:, self.binary_dim_y:], y[:, self.binary_dim_y:])
            # put binary_loss on the same scale as scaler_loss
            self.binary_loss = self.binary_loss * self.scaler_loss
            self.total_loss = self.binary_loss + self.scaler_loss
            return self.total_loss
        else:
            loss = self.mse(x, y)
            # concat to regloss
            self.reg_loss = torch.cat((self.reg_loss, torch.tensor([loss.item()]).to(self.reg_loss.device)))  # add new loss to the tensor
            return loss
    def forward(self, x, y):
        return self.__call__(x, y)

    
# lets write a general training class that will work for all of our models. We also want this to 
# incorporate with all the sklearn functionality so we can use gridsearchCV and other tools
# lets predefine all params in the init function and then have a fit function that will train the model
def get_model(estimator_type: str, config: dict):
    if 'lr_init' not in config.keys():
        config['lr_init'] = 0.001

    if estimator_type == 'NN':
        model = models.NN(**config)
    elif estimator_type == 'VAE':
        model = models.VAE(**config)
    elif estimator_type == 'LSTM':
        model = models.LSTM(**config)
    elif estimator_type == 'CNN':
        model = models.CNN(**config)
    elif estimator_type == 'Hierarch_VAE':
        model = models.Hierarch_VAE(**config)
    else:
        raise ValueError(f"Invalid estimator_type: {estimator_type}")
    if config['optimizer_class'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr_init'])    
    elif config['optimizer_class'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['lr_init'])
    else:
        # default to Adam
        optimizer = optim.Adam(model.parameters(), lr=config['lr_init'])
    return model, optimizer


def get_model_params(estimator_type: str, config: dict):
    if 'lr_init' not in config.keys():
        config['lr_init'] = 0.001

    if estimator_type == 'NN':
        model = models.NN(**config)
    elif estimator_type == 'VAE':
        model = models.VAE(**config)
    elif estimator_type == 'LSTM':
        model = models.LSTM(**config)
    elif estimator_type == 'CNN':
        model = models.CNN(**config)
    elif estimator_type == 'Hierarch_VAE':
        model = models.Hierarch_VAE(**config)
    else:
        raise ValueError(f"Invalid estimator_type: {estimator_type}")

    return model.params

class Trainer():
    def __init__(self, **kwargs):
        self.config = kwargs.copy()
        # if optimizer class is not chosen set it to Adam
        if 'optimizer_class' not in self.config.keys():
            self.config['optimizer_class'] = 'Adam'
        self.device = 'cpu'
        self.estimator_type = self.config.pop('estimator_type', 'NN')
        if 'estimator_type' == None:
            self.estimator_type = 'NN'
        self.estimator_params = get_model_params(estimator_type = self.estimator_type, config=self.config)
        self.training_params = {
            'epochs': 100,
            'batch_size': 100,
            'lr_init': 0.001,
            'metric': r2_score,
            'lr_gamma': 0.1,
            'scheduler': LR_scheduler,
            'device': 'cpu',
            'is_VAE': False,
            'n_inputs': 10,
            'n_outputs': 1,
            'binary_dim': 0,
            'reg_factor': 0.0,
            'beta': 0,
            'gamma': 0,
            'binary_dim_y': 0,
            'verbose': 0
        }
        self.optimizer_keys = ['lr_init','weight_decay']
        self.training_param_keys = list(self.training_params.keys())
        self.estimator_params.update({k: v for k, v in self.config.items() if k in self.estimator_params})
        # first apply the default training params
        self.training_params.update({k: v for k, v in self.config.items() if k in self.training_params})
        # now update training_params wit config
        self.training_params.update({k: v for k, v in self.config.items() if k in self.training_params})
        # now update config with training_params
        self.config.update(self.training_params)
        self.config.update(self.estimator_params)
        self.optimizer_params = {k: v for k, v in self.config.items() if k in self.optimizer_keys}
        
        # update the class attributes with the config
        for key, value in self.config.items():
            setattr(self, key, value)

        self.loss_terms = []
        
        if self.is_VAE:
            self.loss_func = VAE_loss(self.binary_dim)
        else:
            self.loss_func = MSE_Loss(self.binary_dim_y)
        self.loss_terms.append(self.loss_func)
        
        if self.reg_factor > 0:
            self.regression_loss = Regression_loss(self.reg_factor)
            self.regression_loss = self.regression_loss.to(self.device)
            self.loss_terms.append(self.regression_loss)
        
        if self.beta > 0:
            self.L1_reg = L1_regularization(self.beta)
            self.L1_reg = self.L1_reg.to(self.device)
            self.loss_terms.append(self.L1_reg)
        
        if self.gamma > 0:
            self.L2_reg = L2_regularization(self.gamma)
            self.L2_reg = self.L2_reg.to(self.device)
            self.loss_terms.append(self.L2_reg)

        self.estimator, self.optimizer = get_model(self.estimator_type, self.config)
        if self.scheduler == True:
            self.milestones = [self.epochs // 2, self.epochs // 4 * 3]
            self.scheduler = LR_scheduler(self.optimizer, milestones=self.milestones, gamma=self.lr_gamma)

        self.estimator = self.estimator.to(self.device)
        self.loss_func = self.loss_func.to(self.device)
        self.optimizer_params = {k: v for k, v in self.config.items() if k in self.optimizer_keys}
        self.optimizer_class = torch.optim.Adam
        # change 'lr_init' to 'lr' for the optimizer
        self.optimizer_params['lr'] = self.optimizer_params.pop('lr_init')
        self.optimizer = self.optimizer_class(self.estimator.parameters(), **self.optimizer_params)
        
        
    def fit(self, x, y):
        self.n_inputs = x.shape[1]
        self.n_outputs = y.shape[1]
        self.train_observations = x.shape[0]
        self.config['n_inputs'] = self.n_inputs
        self.config['n_outputs'] = self.n_outputs
        self.config['train_observations'] = self.train_observations

        if x is None or y is None:
            raise ValueError("x and/or y is None")
        else:
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y).float()

        #self.estimator, self.optimizer = get_model(self.estimator_type, self.config
        self.estimator, self.optimizer = get_model(self.estimator_type, self.config)
        self.estimator = self.estimator.to(self.device)
        self.dataloader = DataLoader(TensorDataset(x, y), batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.train()
        return self
        
    # rewrite the predict functino to use a dataloader to batch the data
    def predict(self, x):
        # make sure x is a torch tensor, if not convert it and send it to the device
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float().to(self.device)
        else:
            x = x
        # create a dataloader
        dataloader = DataLoader(TensorDataset(x), batch_size=self.batch_size, shuffle=False)
        # make a list to store the predictions
        predictions = []
        predictions = [self.estimator(batch[0]) for batch in dataloader]  # extract the tensor from the batch
        # concatenate the predictions
        # check if VAE, if so take the first element of the tuple
        if self.is_VAE:
            predictions = torch.cat([i[0] for i in predictions], dim=0)
        else:   
            predictions = torch.cat(predictions, dim=0)
        return predictions


    def score(self, x, y, metric=None):
        if metric is None:
            metric = self.metric

        if not callable(metric):
            raise ValueError("metric must be a callable function or a valid sklearn metric string")

        if isinstance(x, (np.ndarray, pd.DataFrame)):
            x = torch.from_numpy(x.astype('float32')).to(self.device)
        elif isinstance(x, torch.Tensor):
            x = x.float().to(self.device)
        else:
            raise TypeError("x must be a numpy array, pandas DataFrame, or torch tensor")

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).to(self.device)
        elif isinstance(y, torch.Tensor):
            y = y.to(self.device)
        else:
            raise TypeError("y must be a numpy array or torch tensor")

        if x.shape[1] != self.n_inputs:
            raise ValueError(f"x should have {self.n_inputs} features, but got {x.shape[1]} features")

        
        # Bring y and pred back to cpu for scoring as metric might not be GPU compatible
        y = y.cpu().numpy()
        pred = self.predict(x).detach().cpu().numpy()


        score = metric(y, pred)
        if isinstance(score, np.ndarray):
            score = np.mean(score)

        return score

 
    def _get_losses(self):
        losses = {}
        for loss in self.loss_terms:
            losses[loss.__class__.__name__] = loss.reg_loss
        return losses

    @classmethod
    def _get_param_names(cls):
        return list(cls.estimator_params.keys()) + list(cls.training_params.keys())

    def get_params(self, deep=True):
        params = {}
        for attr, value in self.__dict__.items():
            if attr in self.estimator_params or attr in self.training_params:
                params[attr] = value
        return params

    
    def get_estimator_params(self):
        estimator_params = self.estimator.params
        # remove everything with "__"
        estimator_params = {k: v for k, v in estimator_params.items() if "__" not in k}
        return estimator_params
    
    def set_params(self, **kwargs):
        if kwargs is not None:
            self.kwargs = kwargs
        self.__dict__.update(self.kwargs)
        self.kwargs.update(self.__dict__)
        self.kwargs = {key:value for key, value in self.kwargs.items() if key in self.config.keys()}
        if 'self' in self.kwargs.keys():
            self.kwargs.pop('self')
        self.__init__(**self.kwargs)
        self.estimator.update_params(self.kwargs)
        return self
        
    def train(self):
        self.estimator = self.estimator.to(self.device)
        self.losses = torch.tensor([]).to(self.device)
        self.loss_func = self.loss_func.to(self.device)
        self.optimizer = torch.optim.Adam(self.estimator.parameters(), lr=self.lr_init)
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            self.losses = torch.cat((self.losses, torch.tensor([train_loss]).to(self.device)))
        return self

    def train_epoch(self, epoch):
        start = time.time()
        for batch_idx, data in enumerate(self.dataloader):
            y = data[1].to(self.device)
            x = data[0].to(self.device)
            # batch_idx = np.arange(batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size)
            # batch_idx = torch.from_numpy(batch_idx).to(self.device)
            # if batch_idx[-1] > x[0].shape[0]:
            #     batch_idx = batch_idx[:x[0].shape[0]]
                
            self.optimizer.zero_grad()
            loss = self.evaluate_loss(x, y)
            self.train_loss = loss.item()
            loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()
            gc.collect()
        end = time.time()
        
        if self.verbose != 0 and epoch % self.verbose == 0:
            print('====> Epoch: {} Average loss: {:.9f} Time: {:.2f}'.format(epoch, loss, end - start))
            print('Loss components:', {k: v[-1] for k, v in self._get_losses().items()})
        return self.train_loss

    def evaluate_loss(self, X, y):
        if self.is_VAE == True:
            recon_batch, mu, logvar = self.estimator(X)
            loss = self.loss_func(recon_batch, y, mu, logvar)
            loss += self.compute_additional_losses()
        else:
            recon_batch = self.estimator(X)
            loss = self.loss_func(recon_batch, y)
            loss += self.compute_additional_losses()
        return loss

    def compute_additional_losses(self):
        
        total_loss = 0
        # L1 selfularization (Lasso)
        if self.beta > 0:
            L1_loss = self.L1_reg(self.estimator)
            total_loss += L1_loss
            
        # L2 selfularization (Ridge)
        if self.gamma > 0:
            L2_loss = self.L2_reg(self.estimator)
            total_loss += L2_loss
        
        # VAE selfression loss
        #if y is not None and self.reg_factor > 0 and self.is_VAE == True:
           #  z = self.estimator.encode(X)[0]
            #self.ression_loss = self.regression_loss(z, z_y) # typos in the original code
            #total_loss += selfression_loss
        return total_loss


    
class custom_loss(nn.MSELoss):
    def __init__(self, config : dict):
        super(custom_loss, self).__init__()
        self.loss_keys = ['beta','gamma','reg_factor']
        self.loss_params = {key: config[key] for key in config if key in self.loss_keys}
        self.L1_reg = L1_regularization(self.loss_params['beta'])
        self.L2_reg = L2_regularization(self.loss_params['gamma'])
        self.regression_loss = Regression_loss(self.loss_params['reg_factor'])
        self.beta = self.loss_params['beta']
        self.gamma = self.loss_params['gamma']
        self.reg_factor = self.loss_params['reg_factor']
        
    def compute_additional_losses(self, recon_batch, data, batch_idx):
        
        total_loss = 0
        
        # L1 regularization (Lasso)
        if self.beta > 0:
            L1_loss = self.L1_reg(self.model)
            total_loss += L1_loss
        # L2 regularization (Ridge)
        
        if self.gamma > 0:
            L2_loss = self.L2_reg(self.model)
            total_loss += L2_loss
        

        if self.y is not None and self.reg_factor > 0:
            z = self.model.encode(data)[0]
            regression_loss = self.regression_loss(z, self.y[batch_idx])
            total_loss += regression_loss
        
        return total_loss
    
    def __call__(self, y, y_pred, batch=None):
        if batch is None:
            if self.reg_factor > 0:
                print('batch is None, cannot compute loss for the regression (VAE regression loss)')
            return
        elif batch is not None and self.reg_factor > 0:
            extra_loss = self.compute_additional_losses(y, y_pred, batch)
            
        else:
            reg_factor = self.loss_params['reg_factor']
            
        mse_loss = F.mse_loss(y, y_pred)
        total_loss = mse_loss + extra_loss
        return total_loss
    


def run_Pytorch(model, X_train, Y_train, n_epochs=100, learning_rate=1e-5, batch_size=int(1e5), device='cuda', optimizer=None):
    
    torch.cuda.empty_cache()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    losses = train_pytorch(model, 
                 X_train, 
                 Y_train,
                 n_epochs=n_epochs,
                 batch_size=batch_size, 
                 learning_rate=learning_rate)
    return losses

def run_epochs(model, X_train, Y_train, loss_func, optimizer, batches, n_epochs=100, device='cuda'):
    t1 = time.time()
    losses = []
    for epoch in range(n_epochs):
        for i in batches:
           # i = indicies[i]
            optimizer.zero_grad()   # clear gradients for next train
            x = X_train[i,:].to(device)
            y = Y_train[i,:].to(device).flatten()
            pred = model(x).flatten()
            # check if y and pred are the same shape
            if y.shape != pred.shape:
                print('y and pred are not the same shape')
                print(y.shape, pred.shape)
                break
            loss = loss_func(pred, y) # must be (1. nn output, 2. target)
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
        losses.append(loss)
        torch.cuda.empty_cache()
        gc.collect()
        if epoch%10 == 0:
            t2 = time.time()
            print('EPOCH : ', epoch,', dt: ',
                  t2 - t1, 'seconds, losses :', 
                  float(loss.detach().cpu())) 
            t1 = time.time()
    return losses


def train_pytorch(model, X_train, Y_train, n_epochs=1000, batch_size=int(1e3), learning_rate=1e-3, device='cuda', optimizer=None):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    losses = []
    batches = batch_data(X_train, batch_size)
    model = model.to(device)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.MSELoss()
    losses = run_epochs(model, X_train, Y_train, loss_func, optimizer, batches, n_epochs=n_epochs)
    return [i.detach().cpu() for i in losses]

def batch_data(Y, batch_size):
    batch_size = int(batch_size)
    n_observations = int(Y.shape[0])
    batch_index = np.arange(0, n_observations, batch_size)
    #np.random.shuffle(batch_index)
    batches = np.array([np.arange(batch_index[i], batch_index[i+1]) \
                   for i in range(len(batch_index)-1)])
    shape = batches.shape
    temp = batches.reshape(-1,1)
    np.random.shuffle(temp)
    batches = temp.reshape(shape[0], shape[1])
    np.random.shuffle(batches)
    n_batches = len(batches)
    return batches

# lets make a randomizer to see if we can get a better model
# function that returns a dictionary of random parameters
def get_random_params(layers=[2, 10], layer_size=[10, 100], learning_rate=[0.000001, 0.01], weight_decay=[0.0001, 0.1], hidden_size = [2,50], change_layers = []):
    params = {}
    if len(layers) == 1:
        params['layers'] = layers[0]
    else:
        params['layers'] = np.random.randint(layers[0], layers[1])
    if len(layer_size) == 1:
        params['layer_size'] = layer_size[0]
    else:
        params['layer_size'] = np.random.randint(layer_size[0], layer_size[1])
    if len(hidden_size) == 1:
        params['hidden_size'] = hidden_size[0]
    else:
        params['hidden_size'] = np.random.randint(hidden_size[0], hidden_size[1])
    if change_layers:
        params['change_layers'] = np.random.randint(change_layers[0], change_layers[1]+1)
        
        
        
    params['learning_rate'] = np.random.uniform(learning_rate[0], learning_rate[1])
    params['weight_decay'] = np.random.uniform(weight_decay[0], weight_decay[1])
   
    return params


def get_lstm_model(params):
    model = lstm(n_inputs=8, hidden_size=30, n_outputs=600, n_linear_layers=1, 
                 layer_size=10, lstm_n_outputs=30)
    lr = params['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=params['weight_decay'])
    return model, optimizer


# create a cv function
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def get_cv_models_lstm(df, xcols, ycols, n_splits=5, random_state=None, n_inputs=8, hidden_size=30, n_outputs=600, n_linear_layers=[1,5], 
                 layer_size=[10,100], lstm_n_outputs=30, learning_rate=[0.000001, 0.01], weight_decay=[0.0001, 0.1], device='cuda', n_epochs=100,params=None, batch_size = 2048):
    
    # split the data into train and test
    subjects = df['Subject'].unique()
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=False)
    results = []
    if params is None:
        params = get_random_params(n_linear_layers=n_linear_layers, layer_size=layer_size, learning_rate=learning_rate, weight_decay=weight_decay)
    for train_index, test_index in kf.split(subjects):
        print(f"Training on {len(train_index)} subjects and testing on {len(test_index)} subjects")
        
        # get the train and test data
        train_df = df[~df['Subject'].isin(subjects[test_index])]
        test_df = df[df['Subject'].isin(subjects[test_index])]
        X_train = train_df[xcols].values
        Y_train = train_df[ycols].values
        X_test = test_df[xcols].values
        Y_test = test_df[ycols].values
        
        # convert to torch float tensors
        X_train = torch.from_numpy(X_train).float().to(device)
        Y_train = torch.from_numpy(Y_train).float().to(device)
        X_test = torch.from_numpy(X_test).float().to(device)
        Y_test = torch.from_numpy(Y_test).float()
        
        # get the model and optimizer
        model, optimizer = get_model(params)
        
        # train the model
        losses = run_Pytorch(model, X_train, Y_train, n_epochs=n_epochs, learning_rate=params['learning_rate'], batch_size=batch_size, optimizer=optimizer, device=device)
        
        # predict on the test data
        pred = model(X_test).detach().cpu().numpy()
        
        # get the statistics
        mse = mean_squared_error(Y_test, pred)
        mae = mean_absolute_error(Y_test, pred)
        r2 = r2_score(Y_test, pred)
        
        # append the model and statistics to the list
        results.append({'model': model, 'mse': mse, 'r2': r2, 'mae': mae, 'params': params})
        
    # print the average statistics
    avg_mse = np.mean([result['mse'] for result in results])
    avg_r2 = np.mean([result['r2'] for result in results])
    avg_mae = np.mean([result['mae'] for result in results])
    
    # pretty print the results
    print(f"Average Mean Squared Error: {avg_mse:.2f}")
    print(f"Average R2 Score: {avg_r2:.2f}")
    print(f"Average Mean Absolute Error: {avg_mae:.2f}")
    
    # print the params 
    print(f"Params: {params}")
    return results

# make a function to create a random search for model params

def random_search(df, xcols, ycols, outpath, n_splits=5, random_state=None, n_linear_layers=[2, 10], layer_size=[4, 100], hidden_size = 30,
                  learning_rate=[0.000001, 0.01], weight_decay=[0.0001, 0.1], device='cuda', n_epochs=100, n_iter=10, batch_size=2048, lstm_n_outputs=3000):
    # create a list to store the results
    results = []
    avg_results = []
    n_inputs = len(xcols)
    n_outputs = len(ycols)
    for i in range(n_iter):
        # get the cv models
        cv_results = get_cv_models_lstm(df, xcols, ycols, n_splits=n_splits, random_state=random_state, n_inputs=n_inputs, hidden_size=hidden_size, n_outputs=n_outputs, n_linear_layers=n_linear_layers, 
                 layer_size=layer_size, lstm_n_outputs=lstm_n_outputs, learning_rate=learning_rate, weight_decay=weight_decay, device=device, n_epochs=100,params=params, batch_size = batch_size)
                    
        # append the results
        results.append(cv_results)
        avg_result = {'mse':np.mean([model['mse'] for model in cv_results]), 
                       'r2': np.mean([model['r2'] for model in cv_results]), 
                       'mae': np.mean([model['mae'] for model in cv_results])}
        avg_results.append(avg_result)
        
        # save the model architecture and parameters
        statsdf = pd.DataFrame(cv_results)
        statsdf.to_csv(outpath, mode='a', index=False, header=False)
    
    # print the best model
    # result df 
    df = pd.DataFrame(avg_results)
    
    # get the best  model 
    best_model = results[np.argmin(df['mse'])][0]
    best_avg_results = avg_results[np.argmin(df['mse'])]
    best_params = best_model['params']
    
    print(f"Best Model: {best_model}")
    print(f"Best Average Results: {best_avg_results}")
    
    return results, best_model, best_avg_results, best_params

def batch_predict(model, input, batch_size=500):
    n_batches = int(np.ceil(input.shape[0] / batch_size))
    for i in range(n_batches):
        gc.collect()
        torch.cuda.empty_cache()
        if i == 0:
            output = model(input[i*batch_size:(i+1)*batch_size]).detach().cpu()
        else:
            output = torch.cat((output, model(input[i*batch_size:(i+1)*batch_size]).detach().cpu())).detach().cpu()
    return output



                
def run_Pytorch_global(model, X_train, X_global, Y_train, n_epochs=100, learning_rate=1e-5, batch_size=int(1e5), device='cuda', optimizer=None):
    
    torch.cuda.empty_cache()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    losses = train_pytorch_global(model, 
                 X_train, 
                 X_global,
                 Y_train,
                 n_epochs=n_epochs,
                 batch_size=batch_size, 
                 learning_rate=learning_rate)
    return losses

def run_epochs_global(model, X_train, X_global, Y_train, loss_func, optimizer, batches, n_epochs=100, device='cuda'):
    import time
    t1 = time.time()
    losses = []
    for epoch in range(n_epochs):
        for i in batches:
           # i = indicies[i]
            optimizer.zero_grad()   # clear gradients for next train
            x = X_train[i,:].to(device)
            x_global = X_global[i,:].to(device)
            y = Y_train[i,:].to(device).flatten()
            pred = model(x, x_global).flatten()
            # check if y and pred are the same shape
            if y.shape != pred.shape:
                print('y and pred are not the same shape')
                print(y.shape, pred.shape)
                break
            loss = loss_func(pred, y) # must be (1. nn output, 2. target)
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
        losses.append(loss)
        torch.cuda.empty_cache()
        gc.collect()
        if epoch%10 == 0:
            t2 = time.time()
            print('EPOCH : ', epoch,', dt: ',
                  t2 - t1, 'seconds, losses :', 
                  float(loss.detach().cpu())) 
            t1 = time.time()
    return losses


def train_pytorch_global(model, X_train, X_global, Y_train, n_epochs=1000, batch_size=int(1e3), learning_rate=1e-3, device='cuda', optimizer=None):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    losses = []
    batches = batch_data(X_train, batch_size)
    model = model.to(device)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.MSELoss()
    losses = run_epochs_global(model, X_train, X_global, Y_train, loss_func, optimizer, batches, n_epochs=n_epochs)
    return [i.detach().cpu() for i in losses]
