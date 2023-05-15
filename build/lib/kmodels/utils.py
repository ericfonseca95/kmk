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

# lets get the MultiLRstep from torch to schedule the learning rate
from torch.optim.lr_scheduler import MultiStepLR
from . import kmodels as models



# Functions
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
def get_random_params(layers=[2, 10], layer_size=[10, 100], learning_rate=[0.0001, 0.01], weight_decay=[0.0001, 0.1]):
    params = {}
    params['layers'] = np.random.randint(layers[0], layers[1])
    params['layer_size'] = np.random.randint(layer_size[0], layer_size[1])
    params['learning_rate'] = np.random.uniform(learning_rate[0], learning_rate[1])
    params['weight_decay'] = np.random.uniform(weight_decay[0], weight_decay[1])
    return params

def create_model(params, xcols = ['EPB', 'EPL', 'FPL', 'APL', 'ADD', 'FCU', 'FPB', 'OPP'], ycols = ['Fx','Fy','Fz']):
    model = models.NN(n_inputs=len(xcols), n_outputs=len(ycols), layer_size=params['layer_size'], layers=params['layers'])
    lr = params['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=params['weight_decay'])
    return model, optimizer



def get_cv_models_NN(df, xcols = ['EPB', 'EPL', 'FPL', 'APL', 'ADD', 'FCU', 'FPB', 'OPP'], ycols = ['Fx','Fy','Fz'], n_splits=5, random_state=42, layers=[2, 10], layer_size=[10, 100], 
                     learning_rate=[0.0001, 0.01], weight_decay=[0.0001, 0.1], device='cuda', n_epochs=100,
                     params=None, batch_size = 2048):
    # split the data into train and test
    subjects = df['Subject'].unique()
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    results = []
    if params is None:
        params = get_random_params(layers=layers, layer_size=layer_size, learning_rate=learning_rate, weight_decay=weight_decay)
    for train_index, test_index in kf.split(subjects):
        print(f"Training on {len(train_index)} subjects and testing on {len(test_index)} subjects")
        # get the train and test data
        train_df = df[~df['Subject'].isin(subjects[test_index])]
        test_df = df[df['Subject'].isin(subjects[test_index])]
        # get the train and test data
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

def random_search(df, xcols = ['EPB', 'EPL', 'FPL', 'APL', 'ADD', 'FCU', 'FPB', 'OPP'], 
                  ycols = ['Fx','Fy','Fz'], n_splits=5, random_state=42, 
                  layers=[2, 10], layer_size=[10, 100], 
                  learning_rate=[0.0001, 0.01], weight_decay=[0.0001, 0.1], 
                  device='cuda', n_epochs=100, n_iter=10, batch_size=2048):
    # create a list to store the results
    results = []
    avg_results = []
    for i in range(n_iter):
        # get the cv models
        cv_results = get_cv_models_NN(df, xcols, ycols,
                                      n_splits=n_splits, random_state=random_state, layers=layers, layer_size=layer_size, 
                                      learning_rate=learning_rate, weight_decay=weight_decay,
                                      device=device, n_epochs=n_epochs, batch_size = batch_size)
        # append the results
        results.append(cv_results)
        avg_result = {'mse':np.mean([model['mse'] for model in cv_results]), 
                       'r2': np.mean([model['r2'] for model in cv_results]), 
                       'mae': np.mean([model['mae'] for model in cv_results])}
        avg_results.append(avg_result)
    
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

# lets make a dataloader to make it easier to batch the data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, xcols = ['EPB', 'EPL', 'FPL', 'APL', 'ADD', 'FCU', 'FPB', 'OPP'], ycols = ['Fx','Fy','Fz'], sort_column=['Event','Subject']):
        self.df = df
        self.window_size = 100
        self.stride = 10
        # make sure the df is ordered by subject
        self.df = self.df.sort_values(by=sort_column)
        self.subject_index = [i.values for i in self.df.groupby(sort_column).apply(lambda x: x.index)]
        self.subject_index = [sorted(i) for i in self.subject_index]
        self.subjects = list(self.df.groupby(sort_column).groups.keys())
        # sort the subject list index. self.subjects contains a list of indices for that belong to each subject. lets sort each list in ascending order
        
        
        
        # for each subject multiple trails are run. lets split them up and make a list of lists for each subject. The way we know a 
        # new event has started is when the time column resets to its minimum value.
        self.subject_events = []
        time_splits = []
        for subject in self.subject_index:
            # get the time column for the subject
            time = self.df.loc[subject]['Time'].values
            # get the indices where the time resets to its minimum value
            time_splits.append(np.where(time == time.min())[0])
            # split the subject index into events
            self.subject_events.append(np.split(subject, time_splits[-1]))
        # reorganize the dataframe in the order of subject_events
        self.df = pd.concat([pd.concat([self.df.loc[subject] for subject in event]) for event in self.subject_events])
        self.df = self.df.reset_index(drop=True)
        self.subject_index = [i.values for i in self.df.groupby(sort_column).apply(lambda x: x.index)]
        self.subject_index = [sorted(i) for i in self.subject_index]
        # remake subjects_events to be consistent with the new dataframe
        self.subject_events = []
        time_splits = []
        for subject in self.subject_index:
            # get the time column for the subject
            time = self.df.loc[subject]['Time'].values
            # get the indices where the time resets to its minimum value
            time_splits.append(np.where(time == time.min())[0])
            # split the subject index into events
            self.subject_events.append(np.split(subject, time_splits[-1]))
        print(len(self.subject_events))
        # get rid of the empty lists in the list of lists
        self.subject_events = [[i for i in subject if len(i) > 0] for subject in self.subject_events]
        # for each subject-event we need to decompose the sequence using a window and stride
        self.subject_event_windows = []
        for subject in self.subject_events:
            subject_windows = []
            for event in subject:
                # get the number of windows
                n_windows = int(np.floor((len(event) - self.window_size)/self.stride) + 1)
                # get the windows
                windows = [event[i*self.stride:i*self.stride+self.window_size] for i in range(n_windows)]
                subject_windows.append(windows)
            self.subject_event_windows.append(subject_windows)
        
        
        self.xcols = xcols
        self.ycols = ycols
        self.X = df[xcols].values
        self.Y = df[ycols].values
        self.X = torch.from_numpy(self.X).reshape(-1, len(xcols))
        self.Y = torch.from_numpy(self.Y).reshape(-1, len(ycols))
        self.n_time_steps = len(self.subject_events[0][0])
        print(self.X[self.subject_event_windows[0][0][0]].shape)
        # using the subject_windows lets concatenate the data into a single tensor. the first dim is the total number of windows in the dataset
        # the second dim is the number of time steps in each window. the third dim is the number of features in each time step
        self.X_lstm = torch.cat([torch.cat([self.X[subject_event_window].view(1, -1, len(xcols)) for subject_event_window in subject_event_windows], dim=0) for subject_event_windows in self.subject_event_windows], dim=0)
        self.Y_lstm = self.Y_lstm.float()
        #self.X_lstm = self.X.view(-1, self.n_time_steps, len(xcols))
        #self.X_lstm = self.X_lstm.float()
        self.Y = self.Y.float()
        #self.Y_lstm = self.Y.view(-1, self.n_time_steps, len(ycols))
    
        
    def __getitem__(self, index):
        return self.X[self.subject_index[index]].view(-1, len(self.xcols)), self.Y[self.subject_index[index]]
    
    def __len__(self):
        return len(self.subject_index)
    
 

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
    
class VAE_loss():
    def __init__(self, binary_dim):
        self.binary_dim = binary_dim
        self.reg_loss = []
    def __call__(self, recon_x, x, mu, logvar):
        binary_loss = F.binary_cross_entropy(recon_x[:, :self.binary_dim], x[:, :self.binary_dim], reduction='sum')
        scaler_loss = F.mse_loss(recon_x[:, self.binary_dim:], x[:, self.binary_dim:], reduction='sum')
        # put binary_loss on the same scale as scaler_loss
        binary_loss = binary_loss * scaler_loss
        total_loss = binary_loss + scaler_loss
        BCE = torch.mean(total_loss)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        self.reg_loss.append(BCE + KLD)
        return BCE + KLD

class Regression_loss():
    def __init__(self, reg_factor):
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
    
class L2_regularization():
    def __init__(self, gamma):
        self.gamma = gamma
        self.reg_loss = []
    def __call__(self, model):
        for param in model.parameters():
            self.reg_loss.append(0.5 * self.gamma * torch.sum(torch.pow(param, 2)).detach().numpy())
        return self.reg_loss[-1]

class L1_regularization():
    def __init__(self, beta):
        self.beta = beta
        self.reg_loss = []
    def __call__(self, model):
        for param in model.parameters():
            self.reg_loss.append(self.beta * torch.sum(torch.abs(param)))
        return self.reg_loss[-1]
    

class MSE_Loss(nn.Module):
    def __init__(self, binary_dim_y=0):
        super(MSE_Loss, self).__init__()
        self.binary_dim_y = binary_dim_y
        # for the binary variables we want to use binary cross entropy
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
    def __call__(self, x, y):
        if self.binary_dim_y > 0:
            self.binary_loss = self.bce(x[:, :self.binary_dim_y], y[:, :self.binary_dim_y])
            self.scaler_loss = self.mse(x[:, self.binary_dim_y:], y[:, self.binary_dim_y:])
            # put binary_loss on the same scale as scaler_loss
            self.binary_loss = self.binary_loss * self.scaler_loss
            self.total_loss = self.binary_loss + self.scaler_loss
            return self.total_loss
        else:
            return self.mse(x, y)
# lets write a general training class that will work for all of our models. We also want this to 
# incorporate with all the sklearn functionality so we can use gridsearchCV and other tools
# lets predefine all params in the init function and then have a fit function that will train the model

def get_model(config : dict):
    if 'lr_init' not in config.keys():
        config['lr_init'] = 0.001
    model_type = config['estimator_type']
    if model_type == 'NN':
        model = models.NN(**config)
    elif model_type == 'VAE':
        model = models.VAE(**config)
    elif model_type == 'LSTM':
        model = models.LSTM(**config)
    elif model_type == 'CNN':
        model = models.CNN(**config)
    return model, optim.Adam(model.parameters(), lr=config['lr_init'])

    
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
    
    
def get_model_params(config: dict):
    model, opt =   get_model(config)
    return model.params

class Trainer(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.config = {}
        self.device = 'cpu'
        self.estimator_type = 'NN'
        if 'estimator_type' in kwargs.keys():
            self.estimator_type = kwargs['estimator_type']
        self.config['estimator_type'] = self.estimator_type
        self.estimator_params = get_model_params({'estimator_type':self.estimator_type})
        self.epochs = 100
        self.batch_size = 100
        self.lr_init = 0.001
        self.metric = r2_score
        self.optimizer = optim.Adam
        self.lr_gamma = 0.1
        self.scheduler = LR_scheduler
        self.is_VAE = False
        self.n_inputs = 10
        self.n_outputs = 1
        self.binary_dim = 0
        self.reg_factor = 0.0
        self.beta = 1.0
        self.gamma = 1.0
        self.binary_dim_y = 0
        self.verbose = 0
        self.estimator_param_keys = list(self.estimator_params.keys())
        self.training_params_keys = ['epochs', 'batch_size', 'lr_init', 'metric', 'optimizer', 'lr_gamma', 'scheduler', 'device', 'is_VAE', 'binary_dim', 'reg_factor', 'beta', 'gamma']
        self.estimator_params = {key: val for key, val in locals().items() if key in self.estimator_param_keys}
        self.training_params = {key: val for key, val in locals().items() if key in self.training_params_keys}
        
        
        self.config.update(self.training_params)
        self.config.update(self.estimator_params)
        self.__dict__.update(self.config)
        for key in self.config:
            setattr(self, key, self.config[key])
        for att, value in kwargs.items():
            self.config[att] = value
        for att in kwargs:
            setattr(self, att, kwargs[att])
        for att, value in locals().items():
            if '__' or 'self' not in att:
                self.config.update({att:value})
        for att, value in self.config.items():
            setattr(self, att, value)
        self.config.update(kwargs)
        self.kwargs.update(self.config)
        self.config = {key: val for key, val in self.config.items() if '__' not in key or 'self' not in key}
        # set the
        
        
        # set all the conditionals and save values to the class
        ####################################################################
        self.loss_terms = []
        
        if self.is_VAE:
            self.loss_func = VAE_loss(self.binary_dim)
        else:
            self.loss_func = MSE_Loss(self.binary_dim_y)
            
        if self.reg_factor > 0:
            self.regression_loss = Regression_loss(self.reg_factor)
            self.loss_terms.append(self.regression_loss)
        
        if self.beta > 0:
            self.L1_reg = L1_regularization(self.beta)
            self.loss_terms.append(self.L1_reg)
        
        
        if self.gamma > 0:
            self.L2_reg = L2_regularization(self.gamma)
            self.loss_terms.append(self.L2_reg)
            
        # initialize the losses
        self.losses = []
        self.train_losses = []
        self.test_losses = []
        self.train_scores = []
        self.test_scores = []
        if 'self' in self.config.keys():
            self.config.pop('self')
        self.estimator, self.optimizer = get_model(self.config)
        self.estimator = self.estimator
        self.estimator_params = self.estimator.params
        self.kwargs = kwargs
        if self.scheduler == True:
            self.milestones = [self.epochs//2, self.epochs//4*3]
            self.scheduler = LR_scheduler(self.optimizer, milestones=self.milestones, gamma=self.lr_gamma)
        ####################################################################

        
    def train(self):
        self.losses = torch.tensor([]).to(self.device)
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)

    def train_epoch(self, epoch):
        train_loss = 0
        start = time.time()
        for batch_idx, data in enumerate(self.dataloader):
            y = data[1].to(self.device)
            data = data[0].to(self.device)
            batch_idx = np.arange(batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size)
            if batch_idx[-1] > data[0].shape[0]:
                batch_idx = np.arange(batch_idx[0], data[0].shape[0])
            if self.is_VAE == True:
                recon_batch, mu, logvar = self.estimator(data)
                loss = self.loss_func(recon_batch, data, mu, logvar) # type: ignore
            else:
                recon_batch = self.estimator(data)
                loss = self.loss_func(recon_batch, data) # type: ignore
            loss += self.compute_additional_losses(data, batch_idx, y= y if self.reg_factor > 0 else None)
            loss = loss / len(data)
            self.optimizer.zero_grad() # type: ignore
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step() # type: ignore
        # if self.scheduler is not None:
        # #     print(epoch, self.scheduler.milestones)
        #     self.scheduler.step(epoch) #type: ignore
        end = time.time()

        if epoch % 10 == 0 and self.verbose != 0:
            # print the time and the components of the loss
            print('====> Epoch: {} Average loss: {:.9f} Time: {:.2f}'.format(epoch, train_loss, end - start))
            print('Loss components: ', # print last value of all the dicts
                    {k: v[-1] for k, v in self._get_losses().items()})
        self.losses = torch.cat((self.losses, torch.tensor([train_loss]).to(self.device))) # type: ignore
        return 

    def compute_additional_losses(self, data, batch_idx, y=None):
        
        total_loss = 0
        # L1 regularization (Lasso)
        if self.beta > 0:
            L1_loss = self.L1_reg(self.estimator)
            total_loss += L1_loss
            
        # L2 regularization (Ridge)
        if self.gamma > 0:
            L2_loss = self.L2_reg(self.estimator)
            total_loss += L2_loss
        
        # VAE regression loss
        if y is not None and self.reg_factor > 0:
            z = self.estimator.encode(data)[0]
            regression_loss = self.regression_loss(z, y[batch_idx])
            total_loss += regression_loss
        return total_loss
        
    def fit(self, x, y):
        self.n_inputs = x.shape[1]
        self.n_outputs = y.shape[1]
        self.train_observations = x.shape[0]
        self.config['input_dim'] = self.n_inputs
        self.config['output_dim'] = self.n_outputs
        self.config['train_observations'] = self.train_observations
        # self.config['y'] = self.y
        # self.config['x'] = self.x
        if 'self' in self.config.keys():
            self.config.pop('self')
        if x is None or y is None:
            raise ValueError("x and/or y is None")
        else:
            if type(x) == np.ndarray:
                x = torch.from_numpy(x).float().to(self.device)
            if type(y) == np.ndarray:
                y = torch.from_numpy(y).float().to(self.device)
            self.dataloader = DataLoader(TensorDataset(x, y), batch_size=self.batch_size, shuffle=True)
            if self.estimator is None or self.config is None:
                self.estimator, self.optimizer = get_model(self.config)
                self.estimator = self.estimator.to(self.device)
                self.estimator.update_params(self.config)
                self.estimator = self.estimator
                self.estimator_params = self.estimator.params
                self.config.update(self.estimator_params)
            self.estimator_params = {k:v for k, v in self.config.items() if k in self.estimator_params}
            self.train()
        return self
    
    def predict(self, x):
        if self.is_VAE == True:
            return self.estimator(x)[0].detach().cpu()
        else:
            return self.estimator(x).detach().cpu()
    
    def score(self, x, y, metric=None):
        # mae of x and y
        # make sure x and y are torch tensors
        if x.shape[1] != self.n_inputs:
            # warning that the model is being generated because it has not been fit
            raise ValueError("x has the wrong number of features")
        if y.shape[1] != self.n_outputs:
            raise ValueError("y has the wrong number of features")
        if type(x) == np.ndarray:
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        else:
            x = x.to(self.device)
        pred = self.predict(x)
        if type(y) == np.ndarray:
            pass
        else:
            y.detach().cpu().numpy()
        return self.metric(y, pred)
    
        # get loss curves
    def _get_losses(self):
        # create a dictionary of all the components of the loss
        #losses = {'vae_loss': self.vae_loss.reg_loss, 'regression_loss': self.regression_loss.reg_loss, 'L1_loss': self.L1_reg.reg_loss, 'L2_loss': self.L2_reg.reg_loss}
        # get the loss attributes from the loss classes
        losses = {}
        for att in self.__dict__:
            if att.endswith('loss') or att.endswith('reg'):
                    # check if the attribute is a list with more than one element
                    if len(self.__dict__[att].reg_loss) > 0:
                        losses[att] = torch.concat([torch.tensor(self.__dict__[att].reg_loss)]).detach().cpu().numpy()
                    
        # for att in losses.keys():
        #     if len(losses[att]) > 1:
        #         print(att, losses[att])
        #         losses[att] = torch.concat(losses[att]).detach().cpu().numpy()
        return losses
    
    def get_params(self, deep=True):
        return self.config
    
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