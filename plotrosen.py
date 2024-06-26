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
from scipy.stats import qmc


# Objective function
def f(x):
    if 0<= x < 0.333:
        return np.cos(12 * np.pi * x) * 1.35
    elif 0.333 <= x <= 0.666:
        return 1.35
    elif 0.66 < x <= 1:
        return np.cos(6 * np.pi * x) * 1.35

def rosenbrock_5d(x):
    """
    Calculate the 5-dimensional Rosenbrock function.

    Parameters:
    x (np.ndarray): A 1D array of length 5.

    Returns:
    float: The value of the 5-dimensional Rosenbrock function at x.
    """
    
    return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1))



seed = 2
np.random.seed(seed)

n = 80
noise = 0.1
n_dim = 5


sampler = qmc.LatinHypercube(d=n_dim)
X_train = sampler.random(n=n)
y_train = np.apply_along_axis(rosenbrock_5d, 1, X_train).reshape(-1,1) + np.random.normal(0, noise, n).reshape(-1,1)

# X_train = np.linspace(0, 1, n).reshape(-1, 1)
# y_train = np.apply_along_axis(rosenbrock_5d, 1, X_train) + np.random.normal(0, noise, n).reshape(-1,1)

# X_new = np.array([0.23232323, 0.50505051, 0.06060606, 0.38383838, 0.83838384, 0.72727273, 0.04040404, 0.18181818, 0.14141414]).reshape(-1, 1)
# y_new = np.apply_along_axis(f, 1, X_new) + np.random.normal(0, noise, 9).reshape(-1,1)

# X_train = np.concatenate((X_train, X_new))
# y_train =  np.concatenate((y_train, y_new))

m = 1000


# X_test = np.linspace(0, 1, m).reshape(-1, 1)
# yy = np.apply_along_axis(f, 1, xx).reshape(-1,1)

with open('rosen5d_80-70_m100_plot80.pyc', 'rb') as f:
# with open('rosen2d_n10-70_m200_plot32.pyc', 'rb') as f:

    
    plot1 = pickle.load(f)


   
    # print('fit mean', fit['mean'], fit['mean'].shape)
    # print('fit sigma', np.diag(fit['Sigma']), np.diag(fit['Sigma']).shape)
    # print('fit alc', alc, alc.shape)
    # plot = {'mean': fit['mean'], 'sigma': np.diag(fit['Sigma']), 'alc': alc}

rmse = plot1['rmse']
mean = plot1['mean'][0:100]
# mean = plot1['mean']
std = plot1['sigma'][0:100]


# print('mean is',  mean.shape, X_test.shape, X_test[:,0].reshape(-1).shape, mean.reshape(-1).shape)
print('std is', std, std.shape)
print(f'rmse: {rmse}')


X_test = plot1['xx']
X_test = X_test[0:100]
print('X_test shape', X_test.shape)
y_test = np.apply_along_axis(rosenbrock_5d, 1, X_test).reshape(-1,1) 



sort_idx = np.argsort(X_test[:,0])


# Plot results
plt.figure(figsize=(10, 6))
# plt.plot(X_train[:,0], y_train, 'ro', label='Training Data')
plt.plot(X_test[:,0][sort_idx], mean[sort_idx], 'b-', label='DGP predict mean')
plt.plot(X_test[:,0][sort_idx], y_test[sort_idx], 'r-', label='DGP actual')

# plt.scatter(X_test[:,0], y_test, label='DGP actual')
# plt.scatter(X_test[:,0], mean,  label='DGP predict mean')


plt.fill_between(X_test[:,0][sort_idx], mean[sort_idx] - 0.0005 * std[sort_idx], 
                 mean[sort_idx] + 0.0005 * std[sort_idx], color='blue', alpha=0.2, label='95% Confidence Interval')
# plt.fill_between(X_test[:,0][sort_idx], mean[sort_idx] - 0.05 * std[sort_idx], 
#                  mean[sort_idx] + 0.05 * std[sort_idx], color='blue', alpha=0.2, label='95% Confidence Interval')
plt.legend()
plt.title("Deep Gaussian Process on Piecewise Function")
plt.xlabel("x")
plt.ylabel("y")
plt.show()



