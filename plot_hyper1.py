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
fig, ax = plt.subplots(1,1,figsize=(8,6))
g_list = []
for t in range(10,41):
    with open(f'{exp_name}_plot{t}.pyc', 'rb') as f:
        plot = pickle.load(f)
        # g_list.append(plot['g'])
        g_list.append(plot['theta_y'])
        # g_list.append(plot['theta_w'][:,0])s

x = np.arange(10,41)
y = [np.mean(g) for g in g_list]
ax.plot(x,y, color='blue', marker='o',linestyle='-', label='hyperparam val')

ax.set_xlabel('Round')
ax.set_ylabel('Hyper')
ax.set_title('MCMC Posterior Sampling Values')

plt.show()


    
