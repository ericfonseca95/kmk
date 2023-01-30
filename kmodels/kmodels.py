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

def get_model(params, xcols = ['EPB', 'EPL', 'FPL', 'APL', 'ADD', 'FCU', 'FPB', 'OPP'], ycols = ['Fx','Fy','Fz']):
    model = NN(n_inputs=len(xcols), n_outputs=len(ycols), layer_size=params['layer_size'], layers=params['layers'])
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
  
        
        self.xcols = xcols
        self.ycols = ycols
        self.X = df[xcols].values
        self.Y = df[ycols].values
        self.X = torch.from_numpy(self.X).reshape(-1, len(xcols))
        self.Y = torch.from_numpy(self.Y).reshape(-1, len(ycols))
        self.n_time_steps = len(self.subject_events[0][0])
        self.X_lstm = self.X.view(-1, self.n_time_steps, len(xcols))
        self.X_lstm = self.X_lstm.float()
        self.Y = self.Y.float()
        self.Y_lstm = self.Y.view(-1, self.n_time_steps, len(ycols))
    
        
    def __getitem__(self, index):
        return self.X[self.subject_index[index]].view(-1, len(self.xcols)), self.Y[self.subject_index[index]]
    
    def __len__(self):
        return len(self.subject_index)
    
    
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