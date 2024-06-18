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

exp_name = 'rosen5d_n10-30_m100'
fig, ax = plt.subplots(1,1,figsize=(8,6))
g_list = []
for t in range(10,41):
    with open(f'{exp_name}_plot{t}.pyc', 'rb') as f:
        plot = pickle.load(f)
        # g_list.append(plot['g'])
        g_list.append(plot['theta_y'])
        # g_list.append(plot['theta_w'][:,0])
for i, g in enumerate(g_list, start=10):
    print('gshape', g.shape)
    x = np.full(g.shape, i)
    y = g
    ax.scatter(x,y, color='blue', alpha=0.5, s=10)

ax.set_xlabel('Round')
ax.set_ylabel('Hyper')
ax.set_title('MCMC Posterior Sampling Values')

plt.show()


    
