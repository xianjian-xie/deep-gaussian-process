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

exp_name = 'rosen5d_10-50_m100'


fig, ax = plt.subplots(1,3,figsize=(18,6))
g_list = []
for t in range(10,41):
    with open(f'{exp_name}_plot{t}.pyc', 'rb') as f:
        plot = pickle.load(f)
        g_list.append(plot['g'])
        # g_list.append(plot['theta_y'])
        # g_list.append(plot['theta_w'][:,0])

x = np.arange(1,1001)
y = g_list[0]
ax[0].plot(x,y, color='blue', linestyle='-', label='g')

ax[0].set_xlabel('MCMC Round')
ax[0].set_ylabel('g')
ax[0].set_title('trace plot of g')

# plt.show()




# fig, ax = plt.subplots(1,1,figsize=(8,6))
g_list = []
for t in range(10,41):
    with open(f'{exp_name}_plot{t}.pyc', 'rb') as f:
        plot = pickle.load(f)
        # g_list.append(plot['g'])
        g_list.append(plot['theta_y'])
        # g_list.append(plot['theta_w'][:,0])

x = np.arange(1,1001)
y = g_list[0]
ax[1].plot(x,y, color='blue', linestyle='-', label='theta_y')

ax[1].set_xlabel('MCMC Round')
ax[1].set_ylabel('theta_y')
ax[1].set_title('trace plot of theta_y')

# plt.show()

# fig, ax = plt.subplots(1,1,figsize=(8,6))
g_list = []
for t in range(10,41):
    with open(f'{exp_name}_plot{t}.pyc', 'rb') as f:
        plot = pickle.load(f)
        # g_list.append(plot['g'])
        # g_list.append(plot['theta_y'])
        g_list.append(plot['theta_w'][:,0])

x = np.arange(1,1001)
y = g_list[0]
ax[2].plot(x,y, color='blue', linestyle='-', label='theta_w')

ax[2].set_xlabel('MCMC Round')
ax[2].set_ylabel('theta_w')
ax[2].set_title('trace plot of theta_w')

plt.show()


    
