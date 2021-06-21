# !/usr/bin/env python3.7
# -*- coding: <utf-8 -*-

"""
Created on Thu May 17 22:07:10 2020

@author: therrmann
"""
# --- IMPORT DEPENDENCIES ---------------------------------------------------------------------------------------------+

import torch
import copy
import pickle
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import KFold
from skopt.utils import use_named_args
from Step_settings import ML_dimensions, dataset_size, n_split
from helper import read_data, train_test_split
import sklearn as skl
from collections import OrderedDict

cuda = False
ftype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#######################################################################
# Implement base model via pyTorch
#######################################################################

# --- ACTIVATION FUNCTION ---------------------------------------------------------------------------------------------+

def swish(x):
    return x * torch.sigmoid(x)

# --- DEEP NEURAL NETWORK ---------------------------------------------------------------------------------------------+

class DNNModel(torch.nn.Module):
    def __init__(self, D_in, D_out, h_sizes_linear, h_sizes_hidden, dropout_rate, activation_input, activation_hidden):
        super(DNNModel, self).__init__()
        self.input = torch.nn.Linear(in_features=D_in, out_features=h_sizes_linear)
        self.hidden = torch.nn.Linear(in_features=h_sizes_linear, out_features=h_sizes_hidden)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.output = torch.nn.Linear(in_features=h_sizes_hidden, out_features=D_out)
        self.activation_input = str(activation_input)
        self.activation_hidden = str(activation_hidden)

    def forward(self, x):
        out = self.input(x)
        if self.activation_input == 'swish':
            out = swish(out)
        elif self.activation_input == 'relu':
            out = F.relu(out)
        elif self.activation_input == 'tanH':
            out = torch.tanh(out)
        out = self.hidden(out)
        if self.activation_hidden == 'swish':
            out = swish(out)
            out = self.dropout(out)
        elif self.activation_hidden == 'relu':
            out = F.relu(out)
            out = self.dropout(out)
        elif self.activation_hidden == 'tanH':
            out = torch.tanh(out)
            out = self.dropout(out)
        out = self.output(out)
        out = torch.sigmoid(out)
        return out

# --- SOFT DECISION TREE MODEL ------------------------------------------------------------------+

class SDT(torch.nn.Module):

    def __init__(self, cuda, DT_depth, DT_lamda, input_dim, output_dim):
        super(SDT, self).__init__()
        self.depth = DT_depth
        self.lamda = DT_lamda
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.inner_node_num = 2 ** self.depth - 1
        self.leaf_num = 2 ** self.depth

        # Different penalty coefficients for nodes in different layer
        self.penalty_list = [self.lamda * (2 ** (-self.depth)) for depth in range(0, self.depth)]

        # Initialize inner nodes and leaf nodes (input dimension on inner nodes is added by 1, serving as bias)
        self.inner_nodes = torch.nn.Sequential(OrderedDict([
            ('linear', torch.nn.Linear(self.input_dim + 1, self.inner_node_num, bias=False)),
            ('sigmoid', torch.nn.Sigmoid()),
        ]))
        self.leaf_nodes = torch.nn.Linear(self.leaf_num, self.output_dim, bias=False)

    def forward(self, data):
        _mu, _penalty = self._forward(data)
        output = self.leaf_nodes(_mu)
        return output, _penalty

    """ Core implementation on data forwarding in SDT """

    def _forward(self, data):
        batch_size = data.size()[0]
        data = self._data_augment_(data)
        path_prob = self.inner_nodes(data)
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)
        _mu = data.data.new(batch_size, 1, 1).fill_(1.)
        _penalty = torch.tensor(0.).to(self.device)

        begin_idx = 0
        end_idx = 1

        for layer_idx in range(0, self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]
            _penalty = _penalty + self._cal_penalty(layer_idx, _mu,
                                                    _path_prob)  # extract inner nodes in current layer to calculate regularization term
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            _mu = _mu * _path_prob
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)
        mu = _mu.view(batch_size, self.leaf_num)
        return mu, _penalty

    """ Calculate penalty term for inner-nodes in different layer """

    def _cal_penalty(self, layer_idx, _mu, _path_prob):
        penalty = torch.tensor(0.).to(self.device)
        batch_size = _mu.size()[0]
        _mu = _mu.view(batch_size, 2 ** layer_idx)
        _path_prob = _path_prob.view(batch_size, 2 ** (layer_idx + 1))
        for node in range(0, 2 ** (layer_idx + 1)):
            alpha = torch.sum(_path_prob[:, node] * _mu[:, node // 2], dim=0) / torch.sum(_mu[:, node // 2], dim=0)
            penalty -= self.penalty_list[layer_idx] * 0.5 * (torch.log(alpha) + torch.log(1 - alpha))
        return penalty

    """ Add constant 1 onto the front of each instance """

    def _data_augment_(self, input):
        batch_size = input.size()[0]
        input = input.view(batch_size, -1)
        bias = torch.ones(batch_size, 1).to(self.device)
        input = torch.cat((bias, input), 1)
        return input

# --- LINEAR REGRESSION MODEL -----------------------------------------------------------------------------------------+

class RegressionModel(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(RegressionModel, self).__init__()
        self.linear = torch.nn.Linear(D_in, D_out, bias=True)

    def forward(self, x):
        return self.linear(x)

#######################################################################
# Implement training step algorithm
#######################################################################

def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Parameter):
        m.reset_parameters()


def make_train_step(model2, model3, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        model2.train()
        model3.train()


        y_SDT, _ = model2.forward(data=x)
        y_RM = model3(x)

        yhat = torch.squeeze(y_SDT) + torch.squeeze(y_RM)

        loss = loss_fn(yhat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        return loss.item()

    return train_step

def make_DNN_train_step(model1, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_DNN_step(x, y):
        model1.train()

        y_DNN = model1(x)

        yhat = torch.squeeze(y_DNN)

        loss = loss_fn(yhat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    return train_DNN_step

def make_Hyperparameter_step(model1, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def Hyperparameter_step(x, y):
        model1.train()

        y_res = model1(x)

        yhat = torch.squeeze(y_res)

        loss = loss_fn(yhat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    return Hyperparameter_step


#######################################################################
# Implement hyperparameter training model
#######################################################################

def make_hyperparameter_model(model_1, loss_fn, choosen_opt, n_epochs, batch_train):
    # Builds function that performance train_step over all epochs and batches
    def hyperparameter_model(x_fit, y_fit, x_val, y_val):

        x_fit = x_fit.to_numpy()
        y_fit = y_fit.to_numpy()

        x_val = x_val.to_numpy()
        y_val = y_val.to_numpy()

        x_train_tensor = torch.from_numpy(x_fit).float()
        y_train_tensor = torch.from_numpy(y_fit).float()

        x_val_tensor = torch.from_numpy(x_val).float()
        y_val_tensor = torch.from_numpy(y_val).float()

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

        train_loader = DataLoader(dataset=train_dataset, batch_size=int(batch_train), shuffle=False)
        val_loader = DataLoader(dataset=val_dataset, batch_size=int(batch_train), shuffle=False)

        training_losses = []
        validation_losses = []

        Hyperparameter_step = make_Hyperparameter_step(model_1, loss_fn, choosen_opt)

        for epoch in range(n_epochs):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                loss = Hyperparameter_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            training_losses.append(training_loss)

            with torch.no_grad():
                val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    model_1.eval()
                    y_DNN = model_1.forward(x_val)
                    yhat = torch.squeeze(y_DNN)

                    val_loss = loss_fn(yhat, y_val).item()
                    val_losses.append(val_loss)
                validation_loss = np.mean(val_losses)
                validation_losses.append(validation_loss)

            print(f"[{epoch + 1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")

        return training_losses, validation_losses

    return hyperparameter_model

#######################################################################
# Implement training model
#######################################################################

def make_train_model(model_2, model_3, loss_fn, choosen_opt, n_epochs, batch_train):
    # Builds function that performance train_step over all epochs and batches
    def train_model(x_fit, y_fit):

        x_fit = x_fit.to_numpy()
        y_fit = y_fit.to_numpy()

        x_train_tensor = torch.from_numpy(x_fit).float()
        y_train_tensor = torch.from_numpy(y_fit).float()

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

        train_loader = DataLoader(dataset=train_dataset, batch_size=int(batch_train), shuffle=False)

        training_losses = []

        train_step = make_train_step(model_2, model_3, loss_fn, choosen_opt)

        for epoch in range(n_epochs):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                loss = train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            training_losses.append(training_loss)

            print(f"[{epoch + 1}] Training loss: {training_loss:.3f}")

        return training_losses

    return train_model

def make_DNN_train_model(model_1, loss_fn, choosen_opt, n_epochs, batch_train):
    # Builds function that performance train_step over all epochs and batches
    def train_DNN_model(x_fit, y_fit):

        x_fit = x_fit.to_numpy()
        y_fit = y_fit.to_numpy()

        x_train_tensor = torch.from_numpy(x_fit).float()
        y_train_tensor = torch.from_numpy(y_fit).float()

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

        train_loader = DataLoader(dataset=train_dataset, batch_size=int(batch_train), shuffle=False)

        training_losses = []

        train_DNN_step = make_DNN_train_step(model_1, loss_fn, choosen_opt)

        for epoch in range(n_epochs):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                loss = train_DNN_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            training_losses.append(training_loss)

            print(f"[{epoch + 1}] Training loss: {training_loss:.3f}")

        return training_losses

    return train_DNN_model

#######################################################################
# Implement helper functions for the training model
#######################################################################

class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

def compile_optimizer(model_2, model_3, optimizer, learning_rate, decay):
    opt = optimizer
    opt = opt.lower()

    models = list(model_2.parameters()) + list(model_3.parameters())

    if opt == 'adam':
        optimizer = torch.optim.Adam(models, lr=learning_rate, weight_decay=decay)
    elif opt == 'rms':
        optimizer = torch.optim.RMSprop(models, lr=learning_rate, weight_decay=decay)
    else:
        optimizer = torch.optim.SGD(models, lr=learning_rate, weight_decay=decay)

    return optimizer

def compile_optimizer_Step2(model_1, optimizer, learning_rate, decay):
    opt = optimizer
    opt = opt.lower()

    models = list(model_1.parameters())

    if opt == 'adam':
        optimizer = torch.optim.Adam(models, lr=learning_rate, weight_decay=decay)
    elif opt == 'rms':
        optimizer = torch.optim.RMSprop(models, lr=learning_rate, weight_decay=decay)
    else:
        optimizer = torch.optim.SGD(models, lr=learning_rate, weight_decay=decay)

    return optimizer

# get the data

all_data = read_data()
x_train, y_train, x_test, y_test = train_test_split(all_data, 0.8)

#######################################################################
# Implement Bayesian Optimization Model include K-Fold
#######################################################################

@use_named_args(dimensions=ML_dimensions)
def model_fitness(h_sizes_linear, h_sizes_hidden, dropout_rate, activation_input, activation_hidden,
                  optimizer, learning_rate, decay, num_epochs, batch_train):

    global x_train

    with open("C:/Users/user/Desktop/Daten/Universität/Masterarbeit/residual.txt", "rb") as fp:  # Pickling
        residuals = pickle.load(fp)

    global dataset_size
    global n_split

    D_in = len(x_train.columns)
    D_out = np.size(residuals, 1)

    model_1 = DNNModel(D_in, D_out, h_sizes_linear, h_sizes_hidden, dropout_rate, activation_input,
                       activation_hidden).to(device)

    init_state_1 = copy.deepcopy(model_1.state_dict())

    choosen_opt = compile_optimizer_Step2(model_1, optimizer, learning_rate, decay)

    init_state_opt = copy.deepcopy(choosen_opt.state_dict())

    n_epochs = num_epochs

    # Specify the training configuration (optimizer and compile)

    loss_fn = torch.nn.MSELoss(reduction='mean')

    hyperparameter_model = make_hyperparameter_model(model_1, loss_fn, choosen_opt, n_epochs,
                                                     batch_train)

    if dataset_size == 'small':

        # Small datasets enable the possibility to do K-fold CV without extensive computation costs

        cv_scores = []
        cv_sem = []
        for train_index, val_index in KFold(n_split, shuffle=False).split(x_train):

            model_1.load_state_dict(init_state_1)

            choosen_opt.load_state_dict(init_state_opt)

            x_fit, x_val = x_train.iloc[train_index], x_train.iloc[val_index]
            y_fit, y_val = residuals.iloc[train_index], residuals.iloc[val_index]

            training_losses, validation_losses = hyperparameter_model(x_fit, y_fit, x_val, y_val)

            last10_scores = np.array(validation_losses[-10:])
            mean_step = last10_scores.mean()
            sem_step = last10_scores.std()
            cv_scores.append(mean_step)
            cv_sem.append(sem_step)

            # If the model didn't converge then set a high loss.
            if np.isnan(any(cv_scores)):
                return 9999.0, 0.0

        mean = np.mean(cv_scores)
        sem = np.std(sem_step)

    elif dataset_size == 'large':

        cv_scores = []
        cv_sem = []

        model_1.load_state_dict(init_state_1)

        choosen_opt.load_state_dict(init_state_opt)

        x_fit, x_val, y_fit, y_val = skl.model_selection.train_test_split(x_train, residuals, test_size=0.2, shuffle=False)

        training_losses, validation_losses = hyperparameter_model(x_fit, y_fit, x_val, y_val)

        last10_scores = np.array(validation_losses[-10:])
        mean_step = last10_scores.mean()
        sem_step = last10_scores.std()
        cv_scores.append(mean_step)
        cv_sem.append(sem_step)

        # If the model didn't converge then set a high loss.
        if np.isnan(any(cv_scores)):
            return 9999.0, 0.0

        mean = np.mean(cv_scores)
        sem = np.std(sem_step)

    else:
        print('Please specify size of dataset in settings.py: small/large')
        raise SystemExit

    print()
    print("Mean MSE: {0:.2%}".format(mean))
    print("Mean sem: {0:.2%}".format(sem))
    print()


    # the optimizer aims for the lowest score

    return mean