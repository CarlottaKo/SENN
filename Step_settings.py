# !/usr/bin/env python3.7
# -*- coding: <utf-8 -*-

"""
Created on Thu May 17 22:07:10 2020

@author: therrmann
"""
# --- IMPORT DEPENDENCIES ---------------------------------------------------------------------------------------------+

from skopt.space import Real, Categorical, Integer

#######################################################################
# Setting for machine learning model
#######################################################################

# --- K-FOLD INFORMATION ----------------------------------------------------------------------------------------------+

dataset_size = 'large'
n_split = 5

# --- SEARCH SPACE: HYPERPARAMETER ------------------------------------------------------------------------------------+

dim_h_sizes_linear = Integer(low=5, high=100, name='h_sizes_linear')
dim_h_sizes_hidden = Integer(low=5, high=50, name='h_sizes_hidden')
dim_dropout_rate = Real(low=0.1, high=0.8, name='dropout_rate')
dim_activation_input = Categorical(categories=['swish', 'relu', 'tanH'], name='activation_input')
dim_activation_hidden = Categorical(categories=['swish', 'relu', 'tanH'], name='activation_hidden')


# --- SEARCH SPACE: COMPILER ------------------------------------------------------+

dim_optimizer = Categorical(categories=['adam', 'RMSProp', 'SGD'], name='optimizer')
dim_learning_rate = Real(low=1e-3, high=1, prior='log-uniform', name='learning_rate')
dim_decay = Real(low=0, high=0.5, name='decay')
dim_num_epochs = Integer(low=12, high=25, name='num_epochs')
dim_batch_train = Integer(low=12, high=100, name="batch_train")

# --- COMBINING MODEL & COMPILER SEARCH SPACE -------------------------------------------------------------------------+

ML_dimensions = [dim_h_sizes_linear,
              dim_h_sizes_hidden,
              dim_dropout_rate,
              dim_activation_input,
              dim_activation_hidden,
              dim_optimizer,
              dim_learning_rate,
              dim_decay,
              dim_num_epochs,
              dim_batch_train
              ]

# --- DEFAULT PARAMETER -----------------------------------------------------------------------------------------------+

ML_default_parameters = [15,  # dim_h_sizes_linear
                      15,  # dim_h_sizes_hidden
                      0.5,  # dim_dropout_rate
                      'swish',  # dim_activation_input
                      'swish',  # dim_activation_hidden
                      'adam',  # dim_optimizer,
                      1,  # dim_learning_rate,
                      0,  # dim_decay,
                      12,  # dim_num_epochs,
                      64  # dim_batch_train,
                      ]

#######################################################################
# Setting for particle swam optimization
#######################################################################

# --- SEARCH SPACE: PSO -----------------------------------------------------------------------------------------------+

dim_x_0 = Real(low=0.1, high=1, name='x_0')
dim_x_1 = Real(low=0.1, high=1, name='x_1')
dim_x_2 = Real(low=0.1, high=1, name='x_2')
dim_w = Real(low=0.01, high=1, name='w')
dim_c1 = Real(low=0.01, high=1, name='c1')
dim_c2 = Real(low=0.01, high=1, name='c2')
dim_num_particles = Integer(low=5, high=50, name='num_particles')
num_maxiter = Integer(low=5, high=150, name='maxiter')

PSO_dimensions = [dim_x_0,
              dim_x_1,
              dim_x_2,
              dim_w,
              dim_c1,
              dim_c2,
              dim_num_particles,
              num_maxiter
              ]

# --- DEFAULT PARAMETER -----------------------------------------------------------------------------------------------+

PSO_default_parameters = [0.5,  # dim_x_0
                      0.5,  # dim_x_1
                      0.5,  # dim_x_2
                      0.1,  # dim_w
                      0.1,  # dim_c1
                      0.2,  # dim_c2,
                      20,  # dim_num_particles,
                      50  # num_maxiter,
                      ]