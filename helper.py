# !/usr/bin/env python3.7
# -*- coding: <utf-8 -*-

"""
Created on Thu May 17 22:07:10 2020

@author: therrmann
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing

def read_data():
    coating_raw = pd.read_csv('C:/Users/user/Desktop/Daten/Universität/Masterarbeit/Data/coating_0.csv', sep=",")
    coating_raw.drop(coating_raw.columns[0], axis=1, inplace=True)
    norm_coating = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(coating_raw),
                             columns=coating_raw.columns,
                             index=coating_raw.index)

    printing_raw = pd.read_csv('C:/Users/user/Desktop/Daten/Universität/Masterarbeit/Data/printing_0.csv', sep=",")
    printing_raw.drop(printing_raw.columns[0], axis=1, inplace=True)
    norm_printing = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(printing_raw),
                                columns=printing_raw.columns,
                                index=printing_raw.index)

    data_all = pd.concat([norm_coating, norm_printing], axis=1, join='outer')

    return data_all


def train_test_split(data, train_percentage):
    test_percentage = 1 - train_percentage
    train_size, test_size = int(len(data) * train_percentage), int(len(data) * test_percentage)
    train, test = data[0:train_size], data[train_size:len(data)]

    x_train, y_train = train.iloc[:, :16], train.iloc[:, 16:]
    x_test, y_test = test.iloc[:, :16], test.iloc[:, 16:]
    return x_train, y_train, x_test, y_test


def test_results(actual, predicted):
    difference_array = np.subtract(actual, predicted)

    squared_array = np.square(difference_array)
    rmse = squared_array.mean()

    abs_array = np.abs(difference_array)
    mae = abs_array.mean()

    return rmse, mae

def components(y_real, y_predicted):
    upper = (y_real - y_predicted)^2
    average = np.mean(y_real)
    lower = (y_real - average)^2