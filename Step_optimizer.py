# --- IMPORT DEPENDENCIES ---------------------------------------------------------------------------------------------+

from __future__ import division
import random
import math
import torch
import numpy as np
from helper import read_data, train_test_split
from skopt.utils import use_named_args
from Step_settings import PSO_dimensions


#######################################################################
# Particle Swam Optimization
#######################################################################

# --- MAIN ------------------------------------------------------------------------------------------------------------+

class Particle:
    def __init__(self, x0, w, c1, c2):
        self.position_i = []  # particle position
        self.velocity_i = []  # particle velocity
        self.pos_best_i = []  # best position individual
        self.err_best_i = -1  # best error individual
        self.err_i = -1  # error individual
        self.w = w  # constant inertia weight (how much to weigh the previous velocity)
        self.c1 = c1 # cognative constant
        self.c2 = c2  # social constant

        for i in range(0, num_dimensions):
            self.velocity_i.append(random.uniform(0, 1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self, costFunc):
        self.err_i = costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i == -1:
            self.pos_best_i = self.position_i
            self.err_best_i = self.err_i

    # update new particle velocity
    def update_velocity(self, pos_best_g):


        for i in range(0, num_dimensions):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = self.c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = self.c2 * r2 * (pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = self.w * self.velocity_i[i] + vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self, bounds):
        for i in range(0, num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i] > bounds[i][1]:
                self.position_i[i] = bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0]


class PSO():
    def __init__(self, costFunc, x0, w, c1, c2, bounds, num_particles, maxiter):
        global num_dimensions

        num_dimensions = len(x0)
        self.err_best_g = -1  # best error for group
        self.pos_best_g = []  # best position for group
        self.costFunc = costFunc
        self.bounds = bounds
        self.num_particles = num_particles
        self.maxiter = maxiter
        # establish the swarm
        self.swarm = []
        for i in range(0, num_particles):
            self.swarm.append(Particle(x0, w, c1, c2))

    def optimize(self):

        # begin optimization loop
        i = 0
        while i < self.maxiter:
            # print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0, self.num_particles):
                self.swarm[j].evaluate(self.costFunc)

                # determine if current particle is the best (globally)
                if self.swarm[j].err_i < self.err_best_g or self.err_best_g == -1:
                    self.pos_best_g = list(self.swarm[j].position_i)
                    self.err_best_g = float(self.swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0, self.num_particles):
                self.swarm[j].update_velocity(self.pos_best_g)
                self.swarm[j].update_position(self.bounds)
            i += 1

        # print final results
        print()
        print('FINAL:')
        print('Feed: {0:.2%}'.format(self.pos_best_g[0]))
        print('Temperature: {0:.2%}'.format(self.pos_best_g[1]))
        print('Infrared: {0:.2%}'.format(self.pos_best_g[2]))
        print('rmse: {0:.2%}'.format(self.err_best_g))
        print()

        return self.pos_best_g, self.err_best_g

#######################################################################
# Hyperparameter Turning for PSO
#######################################################################

@use_named_args(dimensions=PSO_dimensions)
def PSO_Hyperparameter(x_0, x_1, x_2, w, c1, c2, num_particles, maxiter):

    all_data = read_data()

    x_train, y_train, x_test, y_test = train_test_split(all_data, 0.8)

    x_test = x_test.to_numpy()

    cuda = False
    ftype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x0 = [x_0, x_1, x_2]
    bounds = [(0, 1), (0, 1), (0, 1)]


    DNN_Model = torch.load("C:/Users/user/Desktop/Daten/Universität/Masterarbeit/DNN_model.h5")
    SDT_Model = torch.load("C:/Users/user/Desktop/Daten/Universität/Masterarbeit/CP_model.h5")
    Regression_Model = torch.load("C:/Users/user/Desktop/Daten/Universität/Masterarbeit/Regression_model.h5")

    DNN_Model.eval()
    SDT_Model.eval()
    Regression_Model.eval()

    # --- COST FUNCTION -----------------------------------------------------------------------------------------------+

    def PSO_costf(x=x0):
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

        loss = np.sqrt((error ** 2)).mean()

        return loss

    PSO_fit = PSO(PSO_costf, x0, w, c1, c2, bounds, num_particles, maxiter)
    pos_best_g, err_best_g = PSO_fit.optimize()

    return err_best_g
