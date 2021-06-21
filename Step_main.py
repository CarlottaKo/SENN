# !/usr/bin/env python3.7
# -*- coding: <utf-8 -*-

"""
Created on Thu May 17 22:07:10 2020

@author: therrmann
"""

# --- IMPORT DEPENDENCIES ------------------------------------------------------+

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper import read_data, train_test_split, test_results
from Step_model import model_fitness, compile_optimizer, DNNModel, SDT, RegressionModel, make_train_model, make_DNN_train_model, compile_optimizer_Step2
from Step_settings import ML_dimensions, ML_default_parameters, PSO_dimensions, PSO_default_parameters
from skopt import gp_minimize
from Step_optimizer import PSO, PSO_Hyperparameter
import pickle
import csv
from sklearn.metrics import explained_variance_score
import time


def run(current_step):

    all_data = read_data()

    x_train, y_train, x_test, y_test = train_test_split(all_data, 0.8)

    cuda = False
    ftype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x_train_num = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()
    x_train_tensor = torch.tensor(x_train_num).type(ftype)
    x_test_tensor = torch.tensor(x_test).type(ftype)

    D_in = len(x_train.columns)
    D_out = len(y_train.columns)

    #######################################################################
    # Hyperparameter tuning: bayesian optimization using gaussian processes
    #######################################################################

    if current_step == 'St_train':

        SDT_Model = SDT(cuda=False, DT_depth=2, DT_lamda=1e-3, input_dim=D_in, output_dim=D_out).to(device)
        Regression_Model = RegressionModel(D_in, D_out).to(device)

        optimizer = compile_optimizer(model_2=SDT_Model, model_3=Regression_Model, optimizer='adam',
                                      learning_rate=1e-2, decay=0)

        # Specify the training configuration (optimizer and compile)

        loss_fn = torch.nn.MSELoss(reduction='mean')

        train_model = make_train_model(model_2=SDT_Model, model_3=Regression_Model, loss_fn=loss_fn,
                                       choosen_opt=optimizer, n_epochs=12, batch_train=64)

        train_model(x_train, y_train)

        torch.save(SDT_Model, "C:/Users/user/Desktop/Daten/Universität/Masterarbeit/CP_model.h5")
        torch.save(Regression_Model, "C:/Users/user/Desktop/Daten/Universität/Masterarbeit/Regression_model.h5")

        SDT_Model.eval()
        Regression_Model.eval()

        SDT_results, _ = SDT_Model.forward(data=x_train_tensor)
        Regression_results = Regression_Model(x_train_tensor)

        y_pred = np.sum([SDT_results.data.numpy(), Regression_results.data.numpy()], axis=0)

        residual = y_train - y_pred

        for param in SDT_Model.parameters():
            print(param.data)

        with open("C:/Users/user/Desktop/Daten/Universität/Masterarbeit/residual.txt", "wb") as fp:  # Pickling
            pickle.dump(residual, fp)

        return SDT_results.detach().numpy()

    if current_step == 'ML_tune':

        gp_result = gp_minimize(func=model_fitness,
                                dimensions=ML_dimensions,
                                n_calls=50,
                                noise='gaussian',
                                n_jobs=1,
                                kappa=1.96,
                                x0=ML_default_parameters)

        print('Hyperparameters choosen by bayesian optimizer')
        print('h_sizes_linear:~', gp_result.x[0])
        print('h_sizes_hidden:~', gp_result.x[1])
        print('dropout_rate:~', gp_result.x[2])
        print('activation_input:~', gp_result.x[3])
        print('activation_hidden:~', gp_result.x[4])
        print('optimizer:~', gp_result.x[5])
        print('learning_rate:~', gp_result.x[6])
        print('decay:~', gp_result.x[7])
        print('num_epochs:~', gp_result.x[8])
        print('batch_train:~', gp_result.x[9])

        with open('C:/Users/user/Desktop/Daten/Universität/Masterarbeit/tune.sav', 'wb') as h:
            pickle.dump(gp_result, h)

        raise SystemExit

    #################################################################
    # Train Model: Train the model with the optimized hyperparameters
    #################################################################

    elif current_step == 'ML_train':
        with open('C:/Users/user/Desktop/Daten/Universität/Masterarbeit/tune.sav', 'rb') as h:
            tuner = pickle.load(h)

        with open("C:/Users/user/Desktop/Daten/Universität/Masterarbeit/residual.txt", "rb") as fp:  # Pickling
            residuals = pickle.load(fp)

        SDT_Model = torch.load("C:/Users/user/Desktop/Daten/Universität/Masterarbeit/CP_model.h5")
        Regression_Model = torch.load("C:/Users/user/Desktop/Daten/Universität/Masterarbeit/Regression_model.h5")

        DNN_Model = DNNModel(D_in, D_out, h_sizes_linear=tuner.x[0], h_sizes_hidden=tuner.x[1], dropout_rate=tuner.x[2],
                             activation_input=tuner.x[3], activation_hidden=tuner.x[4]).to(device)

        optimizer_DNN = compile_optimizer_Step2(model_1=DNN_Model, optimizer=tuner.x[5],
                                        learning_rate=tuner.x[6], decay=tuner.x[7])

        # Specify the training configuration (optimizer and compile)

        loss_fn = torch.nn.MSELoss(reduction='mean')

        train_DNN_model = make_DNN_train_model(model_1=DNN_Model, loss_fn=loss_fn,
                                       choosen_opt=optimizer_DNN, n_epochs=tuner.x[8], batch_train=tuner.x[9])

        train_DNN_model(x_train, residuals)

        DNN_Model.eval()
        SDT_Model.eval()
        Regression_Model.eval()

        start = time.process_time()
        # your code here

        DNN_results = DNN_Model(x_test_tensor)
        SDT_results, _ = SDT_Model.forward(data=x_test_tensor)
        Regression_results = Regression_Model(x_test_tensor)

        y_pred = np.sum([DNN_results.data.numpy(), SDT_results.data.numpy(), Regression_results.data.numpy()], axis=0)

        print('time:', time.process_time() - start)

        mse, mae = test_results(y_test, y_pred)

        explained_variance = explained_variance_score(y_test, y_pred, multioutput='raw_values')

        ABC = Regression_results.data.numpy()
        ABCD = ABC.sum(axis=0)
        ACA = y_pred.sum(axis=0)

        MIM = ABCD/ACA

        torch.save(DNN_Model, "C:/Users/user/Desktop/Daten/Universität/Masterarbeit/DNN_model.h5")

        print('mse:', mse.mean())
        print('mae:', mae.mean())
        print('Explained Variance:', explained_variance)

        return DNN_results, SDT_results, Regression_results, MIM

    ####################################################
    # Use Model: Load the trained model for further use
    ####################################################

    elif current_step == 'PSO_tune':
        PSO_result = gp_minimize(func=PSO_Hyperparameter,
                                dimensions=PSO_dimensions,
                                n_calls=200,
                                noise='gaussian',
                                n_jobs=1,
                                kappa=1.96,
                                x0=PSO_default_parameters)

        print('Hyperparameters choosen by bayesian optimizer')
        print('x_0:~', PSO_result.x[0])
        print('x_1:~', PSO_result.x[1])
        print('x_2:~', PSO_result.x[2])
        print('w:~', PSO_result.x[3])
        print('c1:~', PSO_result.x[4])
        print('c2:~', PSO_result.x[5])
        print('num_particles:~', PSO_result.x[6])
        print('maxiter:~', PSO_result.x[7])

        with open('C:/Users/user/Desktop/Daten/Universität/Masterarbeit/PSO_tune.sav', 'wb') as p:
            pickle.dump(PSO_result, p)

        raise SystemExit

    elif current_step == 'PSO_run':

        with open('C:/Users/user/Desktop/Daten/Universität/Masterarbeit/PSO_tune.sav', 'rb') as ps:
            PSO_tuner = pickle.load(ps)

        DNN_Model = torch.load("C:/Users/user/Desktop/Daten/Universität/Masterarbeit/DNN_model.h5")
        SDT_Model = torch.load("C:/Users/user/Desktop/Daten/Universität/Masterarbeit/CP_model.h5")
        Regression_Model = torch.load("C:/Users/user/Desktop/Daten/Universität/Masterarbeit/Regression_model.h5")

        DNN_Model.eval()
        SDT_Model.eval()
        Regression_Model.eval()

        # --- COST FUNCTION ------------------------------------------------------------+

        def PSO_costloss(x):

            x_PSO = np.delete(x_test, np.s_[13:16], axis=1)

            dim = len(x_PSO)
            col = len(x)

            for i in range(len(x)):
                new_col = np.full((dim, 1), x[i])
                x_PSO = np.hstack((x_PSO, new_col))

            x_PSO_tensor = torch.tensor(x_PSO).type(ftype)

            DNN_results = DNN_Model(x_PSO_tensor)
            SDT_results, _ = SDT_Model.forward(data=x_PSO_tensor)
            Regression_results = Regression_Model(x_PSO_tensor)

            y_pred = np.sum([DNN_results.data.numpy(), SDT_results.data.numpy(), Regression_results.data.numpy()], axis=0)

            y_target = np.full((dim, col), 1)
            error = y_pred - y_target

            loss = np.sqrt((error ** 2)).max()

            return loss

        initial = [PSO_tuner.x[0], PSO_tuner.x[1], PSO_tuner.x[2]]  # initial starting location [x1,x2...]
        bounds = [(0, 1), (0, 1), (0, 1)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
        PSO_fit = PSO(PSO_costloss, x0=initial, w=PSO_tuner.x[3], c1=PSO_tuner.x[4], c2=PSO_tuner.x[5], bounds=bounds, num_particles=PSO_tuner.x[6], maxiter=PSO_tuner.x[7])
        pos_best_g, err_best_g = PSO_fit.optimize()

        return pos_best_g, err_best_g

    elif current_step == 'test':
        DNN_Model = torch.load("C:/Users/user/Desktop/Daten/Universität/Masterarbeit/DNN_model.h5")
        SDT_Model = torch.load("C:/Users/user/Desktop/Daten/Universität/Masterarbeit/CP_model.h5")
        Regression_Model = torch.load("C:/Users/user/Desktop/Daten/Universität/Masterarbeit/Regression_model.h5")

        DNN_Model.eval()
        SDT_Model.eval()
        Regression_Model.eval()

        x = [0.4034, 0.1269, 1]

        x_PSO = np.delete(x_test, np.s_[13:16], axis=1)

        dim = len(x_PSO)
        col = len(x)

        for i in range(len(x)):
            new_col = np.full((dim, 1), x[i])
            x_PSO = np.hstack((x_PSO, new_col))

        x_PSO_tensor = torch.tensor(x_PSO).type(ftype)

        DNN_results = DNN_Model(x_PSO_tensor)
        SDT_results, _ = SDT_Model.forward(data=x_PSO_tensor)
        Regression_results = Regression_Model(x_PSO_tensor)

        y_pred = np.sum([DNN_results.data.numpy(), SDT_results.data.numpy(), Regression_results.data.numpy()], axis=0)

        DNN_results = DNN_results.data.numpy()
        SDT_results = SDT_results.data.numpy()
        Regression_results = Regression_results.data.numpy()

        np.savetxt('C:/Users/user/Desktop/Daten/Universität/Masterarbeit/DNN_results.csv', DNN_results, delimiter=",")
        np.savetxt('C:/Users/user/Desktop/Daten/Universität/Masterarbeit/SDT_results.csv', SDT_results, delimiter=",")
        np.savetxt('C:/Users/user/Desktop/Daten/Universität/Masterarbeit/Regression_results.csv', Regression_results, delimiter=",")
        np.savetxt('C:/Users/user/Desktop/Daten/Universität/Masterarbeit/y_pred.csv', y_pred, delimiter=",")
        np.savetxt('C:/Users/user/Desktop/Daten/Universität/Masterarbeit/y_test.csv', y_test, delimiter=",")

        return y_pred, DNN_results, SDT_results, Regression_results, y_test


#############################################################
# Current step in the process: Selection by decision variable
#############################################################

if __name__ == '__main__':
    y_pred, DNN_results, SDT_results, Regression_results, y_test = run(current_step='test')

    Pred_Mean = y_pred[:, 0].mean()
    Test_Mean = y_test[:, 0].mean()

    AAA = (Pred_Mean - Test_Mean)/Test_Mean


