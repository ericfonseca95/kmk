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
import dask.array as da
import dask.dataframe as dd
import dask

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
        print(X[self.subject_event_windows[0][0][0]].shape)
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
    
class lstm(nn.Module):
    def __init__(self, n_inputs=8, hidden_size=30, n_outputs=300, n_linear_layers=1, 
                 layer_size=10, lstm_n_outputs=30):
        super().__init__()
        self.lstm = nn.LSTM(n_inputs, hidden_size, batch_first=True)
        self.fcs = nn.ModuleList([nn.Linear(layer_size, layer_size) for i in range(n_linear_layers)])
        self.layer_size = layer_size
        self.n_linear_layers = n_linear_layers 
        self.lstm_n_outputs = lstm_n_outputs
        self.output = nn.Linear(layer_size, n_outputs)
        
    def forward(self, x):
        rows = x.shape[0]
        x, _ = self.lstm(x)
        x = x.reshape(rows, -1)
        for i, fc in enumerate(self.fcs):
            if fc == self.fcs[0]:
                if x.shape[1] != int(fc.in_features):
                    try:    
                        self.fcs[0] = nn.Linear(x.shape[1], self.layer_size)
                        x = F.relu(self.fcs[0](x))
                    except:
                        self.fcs[0] = nn.Linear(x.shape[1], self.layer_size).to('cuda')
                        x = F.relu(self.fcs[0](x))
                else:
                    x = F.relu(fc(x))
            else:
                x = F.relu(fc(x))
        x = self.output(x)
        return x


class Dataset_LSTM_opensim(torch.utils.data.Dataset):
    def __init__(self, df, window_size = 50, stride = 10, xcols = ['EPB', 'EPL', 'FPL', 'APL', 'ADD', 'FCU', 'FPB', 'OPP'], ycols = ['Fx','Fy','Fz'], sort_column=['Event','Subject']):
        self.df = df
        self.xcols = xcols
        self.ycols = ycols
        self.window_size = window_size
        self.stride = stride
        
        # get the subjects from the dataframe
        subjects = split_df_into_subjects(df)
        subject_events = [split_subject_df(subject) for subject in subjects]
        # for each event create a sliding window using the window size and stride
        subject_event_windows = [[sliding_event_df(event, window_size, stride) for event in subject] for subject in subject_events]
        # flatten the list of lists
        subject_event_windows = [window for subject in subject_event_windows for event in subject for window in event]
        self.x_windows = [window[xcols].values for window in subject_event_windows]
        self.y_windows = [window[ycols].values for window in subject_event_windows]
        # save X for an LSTM model. the X shape should be (num_windows, window_size, num_features)
        self.X = np.array([np.expand_dims(x, axis=0) for x in self.x_windows])
        self.X = self.X.reshape(self.X.shape[0], self.X.shape[2], self.X.shape[3])
        # save Y for an LSTM model. the Y shape should be (num_windows, window_size, num_features)
        self.Y = np.array([np.expand_dims(y, axis=0) for y in self.y_windows])
        self.Y = self.Y.reshape(self.Y.shape[0], self.Y.shape[2], self.Y.shape[3])
        # save the number of windows
        self.num_windows = self.X.shape[0]
        # save the number of features
        self.num_features = self.X.shape[2]
        # save the number of outputs
        self.num_outputs = self.Y.shape[2]
        # save all the variables we used to the class
        self.subjects = subjects
        self.subject_events = subject_events
        self.subject_event_windows = subject_event_windows
        self.sort_column = sort_column
    
    def __len__(self):
        return self.num_windows
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class Dataset_LSTM_experimental(torch.utils.data.Dataset):
    def __init__(self, df, window_size = 50, stride = 10, xcols = ['EPB', 'EPL', 'FPL', 'APL', 'ADD', 'FCU', 'FPB', 'OPP'], ycols = ['Fx','Fy','Fz'], sort_column=['Event','Subject']):
        self.df = df
        self.xcols = xcols
        self.ycols = ycols
        # the window size represents how many samples to include in each window
        self.window_size = window_size
        # the stride represents how many samples to skip between windows
        self.stride = stride
        
        
        # make sure the df is ordered by subject
        self.df = self.df.sort_values(by=sort_column)
        self.df = self.df.reset_index(drop=True)
        self.subject_index = [i.values for i in self.df.groupby(sort_column).apply(lambda x: x.index)]
        self.subjects = list(self.df.groupby(sort_column).groups.keys())
        self.n_features = len(xcols)
        # sort the subject list index. self.subjects contains a list of indices for that belong to each subject. lets sort each list in ascending order
        # for each subject multiple trails are run. lets split them up and make a list of lists for each subject. The way we know a 
        # new event has started is when the time column resets to its minimum value.
        
        
        
        # split the subject index into events
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
        # get rid of the empty lists in the list of lists
        self.subject_events = [[i for i in subject if len(i) > 0][0] for subject in self.subject_events]
        
        # for each subject we need to create a vector containig the indices for each window
        self.subject_windows = []
        for subject in self.subject_events:
            # get the number of windows for the subject
            n_windows = (len(subject) - self.window_size) // self.stride + 1
            # create a list of lists for the subject
            self.subject_windows.append([subject[i*self.stride:i*self.stride+self.window_size] for i in range(n_windows)])
        # get rid of the empty lists in the list of lists
        self.subject_windows = [[i for i in subject if len(i) > 0] for subject in self.subject_windows]
        # we should now be able to access the dataframe referencing the n-th frame of any subject using the __getitem__ method
        #self.subject_windows = np.concatenate(self.subject_windows)
        # reorganize self.subject_windows to be a single list of lists
        self.subject_windows = [i for subject in self.subject_windows for i in subject]
        # organize X for an LSTM model. The LSTM model expects the input to be in the shape (batch_size, seq_len, n_inputs)
        self.X = np.array([self.df.loc[window][self.xcols].values for window in self.subject_windows])
        self.X = self.X.reshape(-1, self.window_size, self.n_features)
        self.X = torch.from_numpy(self.X).float()
        
        # get the output
        self.Y = np.array([self.df.loc[window][self.ycols].values for window in self.subject_windows])
        self.Y  = torch.from_numpy(self.Y).float()
        self.Y = self.Y.view(self.X.shape[0], -1)
        self.X = self.X.to('cuda')
        self.Y = self.Y.to('cuda')
        
        # make sure we saved all the variables we used to the class
        self.sort_column = sort_column
        self.n_windows = len(self.subject_windows)
        self.time_splits = time_splits
        self.time = time
        return 
        
    
        
    def __getitem__(self, index):
        # the index is the index of the window in the subject_windows lis
        return self.df.loc[self.subject_windows[index]][self.xcols].values, self.df.loc[self.subject_windows[index]][self.ycols].values
    
    def __len__(self):
        return len(self.subject_index)
    
    
    # okay so the kmk.run_Pytorch function returns the loss of the model. and model 
# is now trained. Lets use this to create a random search for the LSTM hyperparameters.
# the hyperparameters we will be searching for are the number of layers, the number of nodes in each layer
# and batch_size
# the random_models_LSTM function will take inputs of the ranges of the hyperparameters, generate n_iter random models with those hyperparameters. 
# the random_search function will take the random_models_LSTM function and the run_Pytorch function and run the random search
# and return the best model and the best loss, and all the models trained and their losses. A dictionary 
# of the hyperparameters and the loss will also be returned.
# !-----------------
# reference lstm model params
# !-------
# lstm hidden_size=30, n_outputs=300, n_linear_layers=1, 
#                 layer_size=10, lstm_n_outputs=30)
# ! ----------------
def random_models_LSTM(n_iter, lstm_hidden_size_range=[10, 100], n_linear_layers_range=[1, 5], 
                       layer_size_range=[10, 100], lstm_n_outputs_range=[10, 100], batch_size_range=[10, 100], learning_rate_range=[0.0001, 0.1]):
    # create a dataframe to store the models
    models = pd.DataFrame(columns=['lstm_hidden_size', 'n_linear_layers', 'layer_size', 'lstm_n_outputs'])
    # loop through the number of models
    for i in range(n_iter):
        # get the hyperparameters
        lstm_hidden_size = np.random.randint(lstm_hidden_size_range[0], lstm_hidden_size_range[1])
        n_linear_layers = np.random.randint(n_linear_layers_range[0], n_linear_layers_range[1])
        layer_size = np.random.randint(layer_size_range[0], layer_size_range[1])
        lstm_n_outputs = np.random.randint(lstm_n_outputs_range[0], lstm_n_outputs_range[1])
        batch_size = np.random.randint(batch_size_range[0], batch_size_range[1])
        learning_rate = np.random.uniform(learning_rate_range[0], learning_rate_range[1])
        # store the hyperparameters
        models.loc[i, 'lstm_hidden_size'] = lstm_hidden_size
        models.loc[i, 'n_linear_layers'] = n_linear_layers
        models.loc[i, 'layer_size'] = layer_size
        models.loc[i, 'lstm_n_outputs'] = lstm_n_outputs
        models.loc[i, 'batch_size'] = batch_size
        models.loc[i, 'learning_rate'] = learning_rate
    return models



# use random_models_LSTM to generate the models
# use run_Pytorch to train the models
# return the best model and the best loss, and all the models trained and their losses. A dataframe
# of the hyperparameters and the loss will also be returned. store the model on the df as well. 
def train_LSTMs(models, train_data, test_data, n_epochs=100, learning_rate=None):
    # create a dataframe to store the losses
    losses = pd.DataFrame(columns=['loss'])
    # loop through the models
    for i in range(len(models)):
        # get the hyperparameters
        lstm_hidden_size = models.loc[i, 'lstm_hidden_size']
        n_linear_layers = models.loc[i, 'n_linear_layers']
        layer_size = models.loc[i, 'layer_size']
        lstm_n_outputs = models.loc[i, 'lstm_n_outputs']
        batch_size = models.loc[i, 'batch_size']
        if learning_rate is None:
            learning_rate = models.loc[i, 'learning_rate']
        else:
            pass
        # create the model
        model = lstm(n_linear_layers=n_linear_layers,lstm_n_outputs=lstm_n_outputs, layer_size=layer_size, n_outputs=train_data.Y.shape[1], hidden_size=lstm_hidden_size)
        model = model.to('cuda')
        print(model(train_data.X[0].to('cuda')))
        print(model)
        # train the model
        print('training model', i)
        print('batch_size', batch_size)
        # print the how many observations are in the training data
        print('training data size', len(train_data.X))
        print('testing data size', len(test_data.X))
        print('shape of training data', train_data.X.shape, train_data.Y.shape)
        print('shape of testing data', test_data.X.shape, test_data.Y.shape)
              
        loss = run_Pytorch(model, train_data.X, train_data.Y, batch_size=models.loc[i, 'batch_size'], 
                               learning_rate=models.loc[i, 'learning_rate'], n_epochs=n_epochs, device='cuda')
        # store the loss
        pred = model(test_data.X.to('cuda')).detach().cpu().numpy()
        mae = mean_absolute_error(test_data.Y.detach().cpu().numpy(), pred)
        r2 = r2_score(test_data.Y.detach().cpu().numpy(), pred)
        mse = mean_squared_error(test_data.Y.detach().cpu().numpy(), pred)
        losses.loc[i, 'mae'] = mae
        losses.loc[i, 'r2'] = r2
        losses.loc[i, 'mse'] = mse
        # store the model
        models.loc[i, 'model'] = model
    # get the best model and loss
    # best_loss = losses['loss'].min()
    # best_model = models.loc[losses['loss'].idxmin(), 'model']
    # TypeError: reduction operation 'argmin' not allowed for this dtype
    # try again
    # use the last index in each loss list
    best_score = losses['r2'].max()
    best_model = models.loc[losses['mae'].idxmin(), 'model']
    return best_model, best_score, models, losses, models


def split_df_into_subjects(df):
    subject_names = df['Subject'].unique()
    return [df[df['Subject'] == subject] for subject in subject_names]

def split_subject_df(subject_df):
    # get the time column for the subject
    time = subject_df['Time'].values
    # get the indices where the time resets to its minimum value
    time_splits = np.where(time == time.min())[0]
    # split the subject index into events
    subject_events = np.split(subject_df.index, time_splits)
    # get rid of the empty lists in the list of lists
    subject_events = [i for i in subject_events if len(i) > 0]
    # return the list of events
    return [subject_df.loc[event] for event in subject_events]

def sliding_event_df(event_df, window_size, stride):
    # get the number of rows in the event
    num_rows = event_df.shape[0]
    # get the number of windows in the event
    num_windows = (num_rows - window_size) // stride + 1
    # get the indices of the windows
    window_indices = [np.arange(i, i + window_size) for i in range(0, num_windows * stride, stride)]
    # return the list of windows
    return [event_df.iloc[window] for window in window_indices]

class Dataset_LSTM(torch.utils.data.Dataset):
    def __init__(self, df, window_size = 50, stride = 10, xcols = ['EPB', 'EPL', 'FPL', 'APL', 'ADD', 'FCU', 'FPB', 'OPP'], ycols = ['Fx','Fy','Fz'], sort_column=['Event','Subject']):
        self.df = df
        self.xcols = xcols
        self.ycols = ycols
        self.window_size = window_size
        self.stride = stride
        
        # get the subjects from the dataframe
        subjects = split_df_into_subjects(df)
        subject_events = [split_subject_df(subject) for subject in subjects]
        # for each event create a sliding window using the window size and stride
        subject_event_windows = [[sliding_event_df(event, window_size, stride) for event in subject] for subject in subject_events]
        # flatten the list of lists
        subject_event_windows = [window for subject in subject_event_windows for event in subject for window in event]
        self.x_windows = [window[xcols].values for window in subject_event_windows]
        self.y_windows = [window[ycols].values for window in subject_event_windows]
        # save X for an LSTM model. the X shape should be (num_windows, window_size, num_features)
        self.X = np.array([np.expand_dims(x, axis=0) for x in self.x_windows])
        self.X = self.X.reshape(self.X.shape[0], self.X.shape[2], self.X.shape[3])
        # save Y for an LSTM model. the Y shape should be (num_windows, window_size, num_features)
        self.Y = np.array([np.expand_dims(y, axis=0) for y in self.y_windows])
        self.Y = self.Y.reshape(self.Y.shape[0], self.Y.shape[2], self.Y.shape[3])
        # save the number of windows
        self.num_windows = self.X.shape[0]
        # save the number of features
        self.num_features = self.X.shape[2]
        # save the number of outputs
        self.num_outputs = self.Y.shape[2]
        # save all the variables we used to the class
        self.subjects = subjects
        self.subject_events = subject_events
        self.subject_event_windows = subject_event_windows
        self.sort_column = sort_column
    
    def __len__(self):
        return self.num_windows
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
        
def split_df_into_subjects_dask(df):
    subject_names = df['Subject'].unique().compute()
    return [df[df['Subject'] == subject] for subject in subject_names]

def split_subject_df_dask(subject_df):
    # get the time column for the subject
    time = subject_df['Time'].values.compute()
    # get the indices where the time resets to its minimum value
    time_splits = np.where(time == time.min())[0]
    # split the subject index into events
    subject_events = da.split(subject_df.index.compute(), time_splits)
    # get rid of the empty lists in the list of lists
    subject_events = [i for i in subject_events if len(i) > 0]
    # return the list of events
    return [subject_df.loc[event] for event in subject_events]

def sliding_event_df_dask(event_df, window_size, stride):
    # get the number of rows in the event
    num_rows = event_df.shape[0].compute()
    # get the number of windows in the event
    num_windows = (num_rows - window_size) // stride + 1
    # get the indices of the windows
    window_indices = [da.arange(i, i + window_size) for i in range(0, num_windows * stride, stride)]
    # return the list of windows
    return [event_df.iloc[window] for window in window_indices]


# lets write a function that will plot the prediction of the model on a event dataframe from the dataclass 
def get_event_prediction(event_df, model, window_size = 100, stride=5, xcols = ['EPB', 'EPL', 'FPL', 'APL', 'ADD', 'FCU', 'FPB', 'OPP'], ycols = ['Fx','Fy','Fz'], device='cuda'):
    # get the number of rows in the event
    num_rows = event_df.shape[0]
    window_dfs = sliding_event_df(event_df, window_size = num_rows, stride = 1)
    # get the X and Y values for the event
    X = np.array([window[xcols].values for window in window_dfs])
    Y = np.array([window[ycols].values for window in window_dfs])
    # reshape the X and Y values for the model
    X = np.expand_dims(X, axis=0)
    X = X.reshape(X.shape[0], X.shape[2], X.shape[3])
    Y = np.expand_dims(Y, axis=0)
    Y = Y.reshape(Y.shape[0], Y.shape[2], Y.shape[3])
    # get the prediction from the model
    Y_pred = model(torch.tensor(X, dtype=torch.float32).to(device))
    # get the prediction and actual values from the tensors
    Y_pred = Y_pred.detach().cpu().numpy()
    # knowing the stride we know that the prediction is every stride number of rows
    # lets make a list of indexs to keep track of which windows belond to which time points
    window_indices = [i.index for i in window_dfs]
    # go into each window indicies and append the predicitnos to each dataframe Fx_pred, Fy_pred, Fz_pred
    window_dfs = [window_dfs[i].assign(Fx_pred = Y_pred[0][i][0], Fy_pred = Y_pred[0][i][1], Fz_pred = Y_pred[0][i][2]) for i in range(len(window_dfs))]
    window_merged = pd.concat(window_dfs)
    # take the mean of each index using the window indices
    event_df = window_merged.groupby(window_merged.index).mean()
    return window_dfs

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

def TLLSTM(model, X, change_layers=1):
    new = lstm(n_linear_layers=model.n_linear_layers, n_outputs=model.output.out_features, lstm_n_outputs=model.lstm_n_outputs, layer_size=model.layer_size)
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

# makes make another TLMLSTM_newlayers function
def TLMLSTM_newlayers(model, X, change_layers=1, new_layer_size=10):
    new = lstm(n_linear_layers=model.n_linear_layers, 
               n_outputs=model.output.out_features, 
               lstm_n_outputs=model.lstm_n_outputs, 
               layer_size=model.layer_size)
    test = new(X)
    new.load_state_dict(model.state_dict())
    children = [child for child in new.children()]
    for child in children:
        for param in child.parameters():
            param.requires_grad = False
    total_layers = len(children)
    # add the new layers
    new.new_layer_size = new_layer_size
    new.layers = len(new.fcs) + change_layers
    for i in range(change_layers):
        if i == 0:
            new.fcs.append(nn.Linear(new.layer_size, new.new_layer_size))
        else:
            new.fcs.append(nn.Linear(new.new_layer_size, new.new_layer_size))
            # make sure the new layers are trainable
        new.fcs[-1].requires_grad = True
    # make the new output layer 
    new.output = nn.Linear(new.new_layer_size, new.output.out_features)
    return new
