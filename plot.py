import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, det, slogdet
from scipy.linalg import cho_factor, cho_solve, LinAlgError
import time
from scipy.stats import gamma, norm
import pickle
from copy import deepcopy
from scipy.linalg import blas, lapack
import pandas as pd


# Objective function
def f(x):
    if 0<= x < 0.333:
        return np.cos(12 * np.pi * x) * 1.35
    elif 0.333 <= x <= 0.666:
        return 1.35
    elif 0.66 < x <= 1:
        return np.cos(6 * np.pi * x) * 1.35



seed = 2
np.random.seed(seed)

n = 10
noise = 0.1
X_train = np.linspace(0, 1, n).reshape(-1, 1)
y_train = np.apply_along_axis(f, 1, X_train) + np.random.normal(0, noise, n).reshape(-1,1)

X_new = np.array([0.23232323, 0.50505051, 0.06060606, 0.38383838, 0.83838384, 0.72727273, 0.04040404, 0.18181818, 0.14141414]).reshape(-1, 1)
y_new = np.apply_along_axis(f, 1, X_new) + np.random.normal(0, noise, 9).reshape(-1,1)

X_train = np.concatenate((X_train, X_new))
y_train =  np.concatenate((y_train, y_new))

m = 100
X_test = np.linspace(0, 1, m).reshape(-1, 1)
# yy = np.apply_along_axis(f, 1, xx).reshape(-1,1)

with open('plot20.pyc', 'rb') as f:
    plot1 = pickle.load(f)

    # print('fit mean', fit['mean'], fit['mean'].shape)
    # print('fit sigma', np.diag(fit['Sigma']), np.diag(fit['Sigma']).shape)
    # print('fit alc', alc, alc.shape)
    # plot = {'mean': fit['mean'], 'sigma': np.diag(fit['Sigma']), 'alc': alc}

mean = plot1['mean']
std = plot1['sigma']


# Plot results
plt.figure(figsize=(10, 6))
plt.plot(X_train, y_train, 'ro', label='Training Data')
plt.plot(X_test, mean, 'b-', label='DGP Mean')
plt.fill_between(X_test.flatten(), mean.flatten() - 1.96 * std, 
                 mean.flatten() + 1.96 * std, color='blue', alpha=0.2, label='95% Confidence Interval')
plt.legend()
plt.title("Deep Gaussian Process on Piecewise Function")
plt.xlabel("x")
plt.ylabel("y")
plt.show()



