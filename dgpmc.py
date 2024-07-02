import numpy as np
from numpy.linalg import inv, det, slogdet
from scipy.linalg import cho_factor, cho_solve, LinAlgError
import time
from scipy.stats import gamma, norm
import pickle
from copy import deepcopy
from scipy.linalg import blas, lapack
import pandas as pd
import os



# Objective function
def f(x):
    if 0<= x < 0.333:
        return np.cos(12 * np.pi * x) * 1.35
    elif 0.333 <= x <= 0.666:
        return 1.35
    elif 0.66 < x <= 1:
        return np.cos(6 * np.pi * x) * 1.35
    
# logl: evaluates MVN log likelihood with zero mean, formula 2 and formula 8
def logl(out_vec, in_dmat, g, theta, outer=True, v=None, tau2=False, mu=0, scale=1):
    # print('theta,g,scale', theta, g, scale)
    n = len(out_vec)
    K = scale * Exp2(in_dmat, 1, theta, g)
    # print('K', K.shape, K)
    # print('in_dmat', in_dmat)
    inv_det = inv_det_py(K)
    
    Mi = inv_det['Mi']
    ldet = inv_det['ldet']
    # print('Mi', Mi[:3,:3])
    # print('ldet', ldet)
    # print('out_vec shape', out_vec.shape)
    quadterm = (out_vec - mu).T @ Mi @ (out_vec - mu)
    print('quadterm', quadterm, quadterm.shape)
    if (outer):
        logl_val = (-n * 0.5) * np.log(quadterm) - 0.5 * ldet
    else:
        logl_val = -0.5 * quadterm -0.5 * ldet
    
    print('logl_val', logl_val)

    if tau2:
        tau2_val = quadterm/n
        print('tau2_val', tau2_val, tau2_val.shape)
    else:
        tau2_val = None

    print('tau2_val', tau2_val)

    return {'logl': logl_val, 'tau2': tau2_val}

# formula (1)
def Exp2(distmat, tau2, theta, g):
    n1, n2 = distmat.shape
    covmat = np.zeros((n1,n2))

    for i in range(n1):
        for j in range(n2):
            r = distmat[i,j] / theta
            covmat[i,j] = tau2 * np.exp(-r)

    if n1 == n2:
        for i in range(n1):
            covmat[i,i] += tau2 * g

    return covmat

def inv_det_py(M):
    try:
        c, lower = cho_factor(M)
        Mi = cho_solve((c, lower), np.eye(M.shape[0]))
        ldet = 2 * np.sum(np.log(np.diag(c)))
        # print('Mi, ldet', Mi,ldet)
        return {'Mi': Mi, 'ldet': ldet}
    except LinAlgError as e:
        print("Matrix is not positive definite")

    # M = np.array([[4,2],[2,3]])
    # Mi, ldet = inv_det_py(M)
    # print(Mi, ldet)

    # Mi1 = inv(M)
    # sign, ldet1 = slogdet(M)
    # print(Mi1, ldet1)

def distance_py(X1_in, X2_in):
    """
    Compute the pairwise squared distances between rows of X1_in and X2_in.
    
    Parameters:
    X1_in : numpy.ndarray
        Matrix of size (n1, m).
    X2_in : numpy.ndarray
        Matrix of size (n2, m).
    
    Returns:
    D_out : numpy.ndarray
        Matrix of size (n1, n2) containing the pairwise squared distances.
    """
    n1, m = X1_in.shape
    n2 = X2_in.shape[0]
    
    # Initialize the output distance matrix
    D_out = np.zeros((n1, n2))
    
    # Compute pairwise squared distances
    for i in range(n1):
        for j in range(n2):
            D_out[i, j] = np.sum((X1_in[i, :] - X2_in[j, :])**2)
    
    return D_out   

def distance_symm_py(X_in):
    """
    Compute the symmetric pairwise squared distances between rows of X_in.
    
    Parameters:
    X_in : numpy.ndarray
        Matrix of size (n, m).
    
    Returns:
    D_out : numpy.ndarray
        Symmetric matrix of size (n, n) containing the pairwise squared distances.
    """
    n, m = X_in.shape
    
    # Initialize the output distance matrix
    D_out = np.zeros((n, n))
    
    # Compute symmetric pairwise squared distances
    for i in range(n):
        D_out[i, i] = 0.0
        for j in range(i + 1, n):
            D_out[i, j] = np.sum((X_in[i, :] - X_in[j, :])**2)
            D_out[j, i] = D_out[i, j]
    
    return D_out


def sq_dist(X1, X2=None):
    """
    Compute the pairwise squared distances between rows of X1 and optionally X2.
    
    Parameters:
    X1 : numpy.ndarray
        Matrix of size (n1, m).
    X2 : numpy.ndarray, optional
        Matrix of size (n2, m).
    
    Returns:
    D : numpy.ndarray
        Matrix of pairwise squared distances.
        If X2 is None, returns a symmetric matrix of size (n1, n1).
        Otherwise, returns a matrix of size (n1, n2).
    """
    X1 = np.asarray(X1)
    n1, m = X1.shape
    
    if X2 is None:
        D = distance_symm_py(X1)
    else:
        X2 = np.asarray(X2)
        D = distance_py(X1, X2)
    
    return D

def diag_quad_mat(A, B):
    """
    Compute the diagonal of A @ B @ A.T for s2 calculations.
    
    Parameters:
    A : numpy.ndarray
        Matrix of size (Arow, n).
    B : numpy.ndarray
        Matrix of size (n, n).
    
    Returns:
    s2 : numpy.ndarray
        Vector of size (Arow) containing the diagonal elements.
    """
    Arow = A.shape[0]
    Brow = B.shape[0]
    s2 = np.zeros(Arow)
    
    for i in range(Arow):
        s2[i] = 0.0
        for j in range(Brow):
            temp_sum = 0.0
            for n in range(Brow):
                temp_sum += A[i, n] * B[n, j]
            s2[i] += temp_sum * A[i, j]
    
    return s2

#This computes the negative log-likelihood per data point, adjusted by the length of y.
#The proper noun for this score is log marginal likelihood
def score(y, mu, sigma):
    """
    Calculates the score based on the response vector, predicted mean, and covariance matrix.

    Parameters:
    y (np.ndarray): Response vector.
    mu (np.ndarray): Predicted mean vector.
    sigma (np.ndarray): Covariance matrix.

    Returns:
    float: The calculated score.
    """
    id = inv_det_py(sigma)
    diff = y - mu
    score_value = (- id['ldet'] - diff.T @ id['Mi'] @ diff) / len(y)
    print('cal_score', score_value, score_value.shape)
    return score_value

def rmse(y, mu):
    """
    Calculates the root mean square error (RMSE).

    Parameters:
    y (np.ndarray): Response vector.
    mu (np.ndarray): Predicted mean vector.

    Returns:
    float: The RMSE value.
    """
    
    print('y_in is', y.reshape(-1), y.reshape(-1).shape)
    print('mu_in is', mu, mu.shape)
    return np.sqrt(np.mean((y.reshape(-1) - mu) ** 2))


def krig(y, dx, d_new=None, d_cross=None, theta=1, g=1.5e-8, tau2=1, s2=False, sigma=False, f_min=False, v=2.5, prior_mean=0, prior_mean_new=0):
    """
    kriging interpolation, predict the values at new points based on a set of observed data points, predictive posterior.
    
    Prameters:
    y: The observed target values.
    dx: Distance matrix for the observed data points.
    d_new: Distance matrix for new data points (optional).
    d_cross: Cross-distance matrix between observed and new data points.
    theta: Hyperparameter for the kernel function (length scale).
    g: Noise parameter (nugget effect).
    tau2: Scale parameter for the kernel.
    s2: Boolean indicating whether to return the predictive variance.
    sigma: Boolean indicating whether to return the predictive covariance matrix.
    f_min: Boolean indicating whether to return the minimum expected value.
    v: Smoothness parameter (specific to the Matern kernel, but not used in this code).
    prior_mean: Prior mean for the observed data.
    prior_mean_new: Prior mean for the new data points.
    """
    out = {}
    
    C = Exp2(dx, tau2, theta, g)
    C_cross = Exp2(d_cross, tau2, theta, 0)  
    # no g in rectangular matrix, nugget effect represents noise specific to the observations. 
    # When predicting at new points, you are not directly measuring these new points but instead using the model to infer their values.
    
    C_inv = inv_det_py(C)['Mi']
    out['mean'] = prior_mean_new + C_cross @ C_inv @ (y - prior_mean)
    
    if f_min:  # predict at observed locations, return min expected value
        C_cross_observed_only = Exp2(dx, tau2, theta, 0)
        out['f_min'] = np.min(C_cross_observed_only @ C_inv @ y)
    
    if s2:
        C_new = np.full(d_cross.shape[0], 1 + g)
        out['s2'] = tau2 * (C_new - diag_quad_mat(C_cross, C_inv))
    
    if sigma:
        quad_term = C_cross @ C_inv @ C_cross.T
        C_new = Exp2(d_new, tau2, theta, g)
        out['sigma'] = tau2 * (C_new - quad_term)
    
    return out

    
def fill_final_row(x, w_0, D, theta_w_0, v):
    """
    Fill the final row of the matrix w_0 using kriging interpolation.
    
    Parameters:
    x : numpy.ndarray
        Input matrix of size (n, m).
    w_0 : numpy.ndarray
        Initial matrix of size (n-1, D).
    D : int
        Number of dimensions.
    theta_w_0 : numpy.ndarray
        Array of length scales for each dimension.
    v : float
        Smoothness parameter for the Matern kernel, not used in this code
    
    Returns:
    w_0_new : numpy.ndarray
        Updated matrix with the final row filled.
    """
    n = x.shape[0]
    dx = sq_dist(x)
    new_w = np.zeros(D)
    old_x = x[:n-1, :]
    new_x = x[n-1, :].reshape(1, -1)
    
    for i in range(D):
        krig_result = krig(w_0[:, i], dx[:n-1, :n-1], d_cross=sq_dist(new_x, old_x), theta=theta_w_0[i], g=1.5e-8, v=v)
        new_w[i] = krig_result['mean']
    
    return np.vstack([w_0, new_w])


def check_initialization(initial, layers=2, x=None, D=None, vecchia=None, v=None, m=None):
    if 'tau2' not in initial:
        initial['tau2'] = 1
            
    if layers == 2:
        if 'w' not in initial or initial['w'] is None:
            initial['w'] = np.zeros((x.shape[0], D))
        
        if not isinstance(initial['w'], np.ndarray):
            initial['w'] = np.array(initial['w'])
        
        if initial['w'].shape[1] != D:
            raise ValueError("Dimension of initial['w'] does not match D")
        
        if isinstance(initial['theta_w'], int) or len(initial['theta_w']) == 1:
            initial['theta_w'] = np.repeat(initial['theta_w'], D)
        
        if initial['w'].shape[0] == x.shape[0] - 1:
            initial['w'] = fill_final_row(x, initial['w'], D, initial['theta_w'], v)
    
    return initial

def check_settings(settings, layers=1, D=None):
    if settings is None:
        settings = {}
    if 'l' not in settings or settings['l'] is None:
        settings['l'] = 1
    if 'u' not in settings or settings['u'] is None:
        settings['u'] = 2

    if 'alpha' not in settings:
        settings['alpha'] = {}
    if 'beta' not in settings:
        settings['beta'] = {}

    if 'g' not in settings['alpha'] or settings['alpha']['g'] is None:
        settings['alpha']['g'] = 1.5
    if 'g' not in settings['beta'] or settings['beta']['g'] is None:
        settings['beta']['g'] = 3.9

    if layers == 1:
        if 'theta' not in settings['alpha'] or settings['alpha']['theta'] is None:
            settings['alpha']['theta'] = 1.5
        if 'theta' not in settings['beta'] or settings['beta']['theta'] is None:
            settings['beta']['theta'] = 3.9 / 1.5

    elif layers == 2:
        if 'theta_w' not in settings['alpha'] or settings['alpha']['theta_w'] is None:
            settings['alpha']['theta_w'] = 1.5
        if 'theta_y' not in settings['alpha'] or settings['alpha']['theta_y'] is None:
            settings['alpha']['theta_y'] = 1.5
        if 'theta_w' not in settings['beta'] or settings['beta']['theta_w'] is None:
            settings['beta']['theta_w'] = 3.9 / 4
        if 'theta_y' not in settings['beta'] or settings['beta']['theta_y'] is None:
            settings['beta']['theta_y'] = 3.9 / 6

    elif layers == 3:
        if 'theta_z' not in settings['alpha'] or settings['alpha']['theta_z'] is None:
            settings['alpha']['theta_z'] = 1.5
        if 'theta_w' not in settings['alpha'] or settings['alpha']['theta_w'] is None:
            settings['alpha']['theta_w'] = 1.5
        if 'theta_y' not in settings['alpha'] or settings['alpha']['theta_y'] is None:
            settings['alpha']['theta_y'] = 1.5
        if 'theta_z' not in settings['beta'] or settings['beta']['theta_z'] is None:
            settings['beta']['theta_z'] = 3.9 / 4
        if 'theta_w' not in settings['beta'] or settings['beta']['theta_w'] is None:
            settings['beta']['theta_w'] = 3.9 / 12
        if 'theta_y' not in settings['beta'] or settings['beta']['theta_y'] is None:
            settings['beta']['theta_y'] = 3.9 / 6

    return settings



eps = 1.5e-8  # Define a small value for eps

def sample_g(out_vec, in_dmat, g_t, theta, alpha, beta, l, u, ll_prev=None, v=None):
    # Propose value
    g_star = np.random.uniform(low=l * g_t / u, high=u * g_t / l)

    # Compute acceptance threshold
    ru = np.random.uniform(low=0, high=1)

    # g_star = 0.0112553
    # ru = 0.720324
    print('g_star is', g_star)
    print('ru is', ru)
    if ll_prev is None:
        ll_prev = logl(out_vec, in_dmat, g_t, theta, outer=True, v=v)['logl']
        print('ll_prev is', ll_prev)
    # print('all param', g_t, alpha, beta, ru, g_star, gamma.logpdf(g_t - eps, a=alpha, scale=1/beta))
    lpost_threshold = (ll_prev + gamma.logpdf(g_t - eps, a=alpha, scale=1/beta) + 
                       np.log(ru) - np.log(g_t) + np.log(g_star))
    print('lpost_threshold is', lpost_threshold)


    ll_new = logl(out_vec, in_dmat, g_star, theta, outer=True, v=v)['logl']
    
    print('ll_new is', ll_new)


    # Accept or reject (lower bound of eps)
    new = ll_new + gamma.logpdf(g_star - eps, a=alpha, scale=1/beta)
    
    print('new is', new)
    
    if new > lpost_threshold:
        return {'g': g_star, 'll': ll_new}
    else:
        return {'g': g_t, 'll': ll_prev}


def sample_theta(out_vec, in_dmat, g, theta_t, alpha, beta, l, u, outer, ll_prev=None, v=None, tau2=False, prior_mean=0, scale=1):
    # print('enter sample_theta', ll_prev, g)
    # Propose value

    theta_star = np.random.uniform(low=l * theta_t / u, high=u * theta_t / l)
    # theta_star = 0.6
    print('theta theta_star is', theta_star)

    # Compute acceptance threshold
    ru = np.random.uniform(low=0, high=1)
    # ru = 0.7
    print('theta ru is', ru)
    if ll_prev is None:
        ll_prev = logl(out_vec, in_dmat, g, theta_t, outer, v, mu=prior_mean, scale=scale)['logl']
        print('theta ll_prev is', ll_prev)
    
    lpost_threshold = (ll_prev + gamma.logpdf(theta_t - eps, a=alpha, scale=1/beta) + 
                       np.log(ru) - np.log(theta_t) + np.log(theta_star))
    
    print('theta lpost_threshold is', lpost_threshold)

    ll_new = logl(out_vec, in_dmat, g, theta_star, outer, v, tau2=tau2, mu=prior_mean, scale=scale)

    print('theta ll_new is', ll_new)

    # Accept or reject (lower bound of eps)
    new = ll_new['logl'] + gamma.logpdf(theta_star - eps, a=alpha, scale=1/beta)
    print('theta new is', new)

    if new > lpost_threshold:
        return {'theta': theta_star, 'll': ll_new['logl'], 'tau2': ll_new.get('tau2')}
    else:
        return {'theta': theta_t, 'll': ll_prev, 'tau2': None}


def sample_w(out_vec, w_t, w_t_dmat, in_dmat, g, theta_y, theta_w, ll_prev=None, v=None, prior_mean=None, scale=1):
    """
    Sample latent variables w using elliptical slice sampling (ESS).
    
    Parameters:
    out_vec (np.ndarray): Observed response vector or matrix y, shape (n_samples, y_dim).
    w_t (np.ndarray): Current values of the latent variables (hidden layer), shape (n_samples, D).
    w_t_dmat (np.ndarray): Distance matrix for the current values of the latent variables w_t, shape (n_samples, n_samples).
    in_dmat (np.ndarray): Input distance matrix computed from the input data x, shape (n_samples, n_samples).
    g (float): Noise parameter (nugget effect) added to the diagonal of the covariance matrix.
    theta_y (float): Length-scale parameter for the covariance function related to the observed data y.
    theta_w (np.ndarray or float): Length-scale parameter for the covariance function related to the latent variables w, length D.
    ll_prev (float, optional): Previous log-likelihood value. If not provided, it will be computed from scratch.
    v (float, optional): Smoothness parameter for the Matern kernel. When v=999, it indicates the use of the squared exponential kernel.
    prior_mean (np.ndarray, optional): Prior mean for the latent variables w. If not provided, defaults to a zero array with the same shape as w_t.
    scale (float, default=1): Scaling factor for the covariance function of the prior distribution.
    
    Returns:
    dict: A dictionary containing:
        'w': Updated latent variables w_t.
        'll': Updated log-likelihood value ll_prev.
        'dw': Updated distance matrix for the latent variables.
    """
    
    if prior_mean is None:
        prior_mean = np.zeros_like(w_t)
    
    D = w_t.shape[1]  # dimension of hidden layer

    if ll_prev is None:
        ll_prev = logl(out_vec, w_t_dmat, g, theta_y, outer=True, v=v)['logl']
    
    for i in range(D):  # separate sampling for each dimension of hidden layer
        
        # Draw from prior distribution
        if v == 999:
            w_prior = np.random.multivariate_normal(prior_mean[:, i], scale * Exp2(in_dmat, 1, theta_w[i], 0))
        
        # Initialize a and bounds on a
        a = np.random.uniform(0, 2 * np.pi)
        amin = a - 2 * np.pi
        amax = a
        
        # Compute acceptance threshold - based on all dimensions of previous w
        ru = np.random.uniform(0, 1)
        ll_threshold = ll_prev + np.log(ru)
        
        # Calculate proposed values, accept or reject, repeat if necessary
        accept = False
        count = 0
        w_prev = w_t[:, i].copy()  # store for re-proposal
        
        while not accept:
            count += 1
            # print('x1 is',x,'y1 is',y)
            # Calculate proposed values and new likelihood
            w_t[:, i] = w_prev * np.cos(a) + w_prior * np.sin(a)
            # print('x2 is',x,'y2 is',y)
            dw = sq_dist(w_t)
            
            new_logl = logl(out_vec, dw, g, theta_y, outer=True, v=v)['logl']
            
            # Accept or reject
            if new_logl > ll_threshold:
                ll_prev = new_logl
                accept = True
            else:
                # Update the bounds on a and repeat
                if a < 0:
                    amin = a
                else:
                    amax = a
                a = np.random.uniform(amin, amax)
                if count > 100:
                    raise RuntimeError('Reached maximum iterations of ESS')
    
    return {'w': w_t, 'll': ll_prev, 'dw': dw}


def gibbs_one_layer(x, y, nmcmc, verb, initial, true_g, settings, v):
    dx = sq_dist(x)
    g = np.zeros(nmcmc)
    if true_g is None:
        g[0] = initial['g']
    else:
        g[0] = true_g
    theta = np.zeros(nmcmc)
    theta[0] = initial['theta']
    tau2 = np.zeros(nmcmc)
    tau2[0] = initial['tau2']
    ll_store = np.zeros(nmcmc)
    ll_store[0] = np.nan
    ll = None
    
    for j in range(1, nmcmc):
        if verb and (j % 500 == 0):
            print(j)
        
        # Sample nugget (g)
        if true_g is None:
            samp = sample_g(y, dx, g[j - 1], theta[j - 1], alpha=settings['alpha']['g'], 
                            beta=settings['beta']['g'], l=settings['l'], u=settings['u'], 
                            ll_prev=ll, v=v)
            g[j] = samp['g']
            ll = samp['ll']
            print(f'g{j}, ll is {g[j]}, {ll}')
        else:
            g[j] = true_g
        
        # Sample lengthscale (theta)
        samp = sample_theta(y, dx, g[j], theta[j - 1], 
                            alpha=settings['alpha']['theta'],
                            beta=settings['beta']['theta'], l=settings['l'], 
                            u=settings['u'], outer=True, ll_prev=ll, v=v, 
                            tau2=True)
        theta[j] = samp['theta']
        ll = samp['ll']
        ll_store[j] = ll
        print(f'theta{j}, ll is {theta[j]}, {ll}')

        if samp['tau2'] is None:
            tau2[j] = tau2[j - 1]
        else:
            tau2[j] = samp['tau2']

        print(f'tau2{j}, ll is {tau2[j]}')
        print()

    return {'g': g, 'theta': theta, 'tau2': tau2, 'll': ll_store}


def fit_one_layer(x, y, nmcmc=10000, sep=False, verb=True, g_0=0.01, 
                  theta_0=0.1, true_g=None, settings=None, cov="matern", v=2.5, 
                  vecchia=False, m=None, ordering=None):
    start_time = time.time()
    
    if m is None:
        m = min(25, len(y) - 1)
            
    if cov == "exp2":
        v = 999  # solely used as an indicator
        
    if not vecchia and len(y) > 300:
        print("'vecchia = TRUE' is recommended for faster computation.")
        
    if nmcmc <= 1:
        raise ValueError("nmcmc must be greater than 1")

    # Check inputs
    if isinstance(x, (list, np.ndarray)) and np.isscalar(x[0]):
        x = np.reshape(x, (-1, 1))
        
    if sep and x.shape[1] == 1:
        sep = False  # no need for separable theta in one dimension
        
    settings = check_settings(settings, layers=1)
    initial = {'theta': theta_0, 'g': g_0, 'tau2': 1}
    
    if m >= len(y):
        raise ValueError("m must be less than the length of y")
        
    if cov == "matern" and v not in [0.5, 1.5, 2.5]:
        raise ValueError("v must be one of 0.5, 1.5, or 2.5")
        
    if ordering is not None:
        if not vecchia:
            print("ordering only used when vecchia = TRUE")

    # Create output object
    out = {'x': x, 'y': y, 'nmcmc': nmcmc, 'settings': settings, 'v': v}
    
    if vecchia:
        out['m'] = m
        out['ordering'] = ordering

    # Conduct MCMC
     
    samples = gibbs_one_layer(x, y, nmcmc, verb, initial, true_g, settings, v)
            
    out.update(samples)
    out['time'] = time.time() - start_time
    
    if vecchia:
        out['class'] = 'gpvec'
    else:
        out['class'] = 'gp'
        
    return out


def ifel(logical, yes, no):

    if logical:
        return yes
    else:
        return no
    
def exp_improv(mu, sig2, f_min):
    """
    Expected improvement calculation.

    Parameters:
    mu (np.ndarray): Mean values.
    sig2 (np.ndarray): Variance values.
    f_min (float): Minimum function value.

    Returns:
    np.ndarray: Expected improvement values.
    """
    ei_store = np.zeros(len(mu))
    i = np.where(sig2 > eps)[0]
    mu_i = mu[i]
    sig_i = np.sqrt(sig2[i])
    ei = (f_min - mu_i) * norm.cdf(f_min, loc=mu_i, scale=sig_i) + \
         sig_i * norm.pdf(f_min, loc=mu_i, scale=sig_i)
    ei_store[i] = ei
    return ei_store

def calc_entropy(mu, sig2, limit):
    """
    Entropy calculation.

    Parameters:
    mu (np.ndarray): Mean values.
    sig2 (np.ndarray): Variance values.
    limit (float): Entropy limit.

    Returns:
    np.ndarray: Entropy values.
    """
    fail_prob = norm.cdf((mu - limit) / np.sqrt(sig2))
    ent = -(1 - fail_prob) * np.log(1 - fail_prob) - fail_prob * np.log(fail_prob)
    ent[np.isnan(ent)] = 0
    return ent

def predict_nonvec(object, x_new, settings, layers):
    """
    parameters:
    (lite=False, store_latent=False, mean_map=True,
    return_all=False, EI=False, entropy_limit=None) in 1d use case
    x_new: matrix of predictive input locations
    settings:
        lite: logical indicating whether to calculate only point-wise 
                variances (lite = TRUE) or full covariance (lite = FALSE)
        store_latent: logical indicating whether to store and return mapped 
                values of latent layers (two or three layer models only)
        mean_map: logical indicating whether to map hidden layers using 
                conditional mean (mean_map = TRUE)
        return_all: logical indicating whether to return mean and point-wise
                variance prediction for ALL samples 
        EI: logical indicating whether to calculate expected improvement 
        entropy_limit: optional limit state for entropy calculations 
    """
    tic = time.process_time()
    x_new = np.asarray(x_new)
    object['x_new'] = x_new
    n_new = x_new.shape[0]
    if layers >= 2:
        D = object['w'][0].shape[1]  # dimension of latent layer(s)
        if settings['lite'] and not settings['mean_map']:
            raise ValueError("mean_map = FALSE requires lite = FALSE")
        if not settings['mean_map']:
            print("mean_map = FALSE may cause numerical instability in latent layer mapping")

    if layers == 1:
        sep = isinstance(object['theta'], np.ndarray)
    else:
        sep = False  # two and three layers never use separable lengthscales

    if not sep:
        dx = sq_dist(object['x'])
        dx_cross = sq_dist(x_new, object['x'])
        if not settings['lite']:
            dx_new = sq_dist(x_new)
        else:
            dx_new = None

    if settings['return_all'] and not settings['lite']:
        raise ValueError("return_all only offered when lite = TRUE")
    if settings['entropy_limit'] is not None and not isinstance(settings['entropy_limit'], (int, float)):
        raise ValueError("entropy_limit must be numeric")
    if settings['EI']:
        if np.all(object['g'] <= 1e-6):
            f_min = False
            y_min = np.min(object['y'])
        else:
            f_min = True
    else:
        f_min = False

    prior_mean_new = 0
    prior_mean = np.zeros(len(object['y']))
    prior_tau2 = 1

    #### seperation
    mu_t = np.zeros((n_new, object['nmcmc']))
    if settings['lite']:
        s2_sum = np.zeros(n_new)
        if settings['return_all']:
            s2_t = np.zeros((n_new, object['nmcmc']))
    else:  
        sigma_sum = np.zeros((n_new, n_new))
    if settings['EI']:  
        ei_sum = np.zeros(n_new)
    if settings['entropy_limit'] is not None:  
        ent_sum = np.zeros(n_new)
    if layers >= 2:
        if settings['store_latent']:
            w_new_list = []
            

    for t in range(object['nmcmc']):

        if layers >= 2:
            w_t = object['w'][t]
            w_new = np.zeros((n_new, D))
            for i in range(D):
                if layers == 2:
                    if object['settings']['pmx']:   # pmx = False
                        prior_mean_new = x_new[:, i]
                        prior_mean = object['x'][:, i]
                        prior_tau2 = object['settings']['inner_tau2']
                k = krig(w_t[:, i], dx, dx_new, dx_cross, object['theta_w'][t, i], g=1e-10, tau2=prior_tau2, sigma=not settings['mean_map'], v=object['v'], prior_mean=prior_mean, prior_mean_new=prior_mean_new)
                if settings['mean_map']:
                    w_new[:, i] = k['mean']
                else:
                    w_new[:, i] = np.random.multivariate_normal(k['mean'], k['sigma'])
            if settings['store_latent']:
                w_new_list.append(w_new)
            dw = sq_dist(w_t)
            dw_cross = sq_dist(w_new, w_t)
            if not settings['lite']:
                dw_new = sq_dist(w_new)
            else:
                dw_new = None

        
        k = krig(object['y'], ifel(layers == 1, dx, dw), ifel(layers == 1, dx_new, dw_new), ifel(layers == 1, dx_cross, dw_cross), object['theta_y'][t], object['g'][t], object['tau2'][t], s2=settings['lite'], sigma=not settings['lite'], f_min=f_min, v=object['v'])
        # print(mu_t[:, t].shape, k['mean'].shape)
        mu_t[:, t] = k['mean'].reshape(-1)
        
        if settings['lite']:
            s2_sum += k['s2']
            if settings['return_all']:
                s2_t[:, t] = k['s2']
        else:
            sigma_sum += k['sigma']
        if settings['EI']:
            if settings['lite']:
                sig2 = k['s2'] - (object['tau2'][t] * object['g'][t])
            else:
                sig2 = np.diag(k['sigma']) - (object['tau2'][t] * object['g'][t])
            ei_sum += exp_improv(k['mean'], sig2, f_min if f_min else y_min)
        if settings['entropy_limit'] is not None:
            if settings['lite']:
                sig2 = k['s2'] - (object['tau2'][t] * object['g'][t])
            else:
                sig2 = np.diag(k['sigma']) - (object['tau2'][t] * object['g'][t])
            ent_sum += calc_entropy(k['mean'], sig2, settings['entropy_limit'])

# add variables to output list
    # mean val
    object['mean'] = np.mean(mu_t, axis=1)
    if layers >= 2:
        if settings['store_latent']:
            object['w_new'] = w_new_list
    if settings['lite']:
        object['s2'] = s2_sum / object['nmcmc'] + np.var(mu_t, axis=1)
        if settings['return_all']:
            object['mean_all'] = mu_t
            object['s2_all'] = s2_t
    else:
        # print('sigma_sum', sigma_sum.shape, object['nmcmc'], mu_t.shape)
        # sigma value
        object['Sigma'] = sigma_sum / object['nmcmc'] + np.cov(mu_t)
    if settings['EI']:
        object['EI'] = ei_sum / object['nmcmc']
    if settings['entropy_limit'] is not None:
        object['entropy'] = ent_sum / object['nmcmc']
    toc = time.process_time()
    object['time'] += toc - tic

    return object


def predict_dgp2(object, x_new, lite=True, store_latent=False, mean_map=True,
                 return_all=False, EI=False, entropy_limit=None, cores=1, **kwargs):
    """
    Predict function for dgp2 objects.

    Parameters:
    object (object): The dgp2 object.
    x_new (np.ndarray): New input data for prediction.
    lite (bool): Lite mode.
    store_latent (bool): Whether to store latent variables.
    mean_map (bool): Whether to use mean map.
    return_all (bool): Whether to return all results.
    EI (bool): Expected improvement.
    entropy_limit (float): Entropy limit.
    cores (int): Number of cores for parallel processing.
    kwargs: Additional arguments.

    Returns:
    object: The updated dgp2 object with predictions.
    """
    settings = {
        'lite': lite,
        'store_latent': store_latent,
        'mean_map': mean_map,
        'return_all': return_all,
        'EI': EI,
        'entropy_limit': entropy_limit,
        'cores': cores
    }
    object = predict_nonvec(object, x_new, settings, layers=2)
    return object

def new_vector(n):
   
    if n == 0:
        return None
    v = np.zeros(n)
    return v

def new_matrix(n1, n2):
   
    if n1 == 0 or n2 == 0:
        return None
    m = np.zeros((n1, n2))
    return m

def sq(x):
  
    return x * x

def covar(col, X1, n1, X2, n2, d):
    """
        calculates the covariance matrix K between the points in X1 and X2 using the specified exponential kernel with length scale d.
    """
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            for k in range(col):
                K[i][j] += sq(X1[i][k] - X2[j][k])
            K[i][j] = np.exp(-K[i][j] / d)
    return K

def calc_g_mui_kxy(col, x, X, n, Ki, Xref, m, d, g, gvec, mui, kx, kxy):
    """
    # calc_g_mui_kxy(col, Xcand[i], X, n, Ki, Xref, nref, d, g, gvec, mui, kx, kxy)
    
    Calculates the g vector, mui, and kxy for the given inputs.

    Parameters:
    col (int): Number of columns in x, X, and Xref. (n_w_old)
    x (np.ndarray): Input vector.   (w_new)
    X (np.ndarray): Matrix of observed inputs.  (w_old)
    n (int): Number of rows in X.   (n_w_old)
    Ki (np.ndarray): Inverse covariance matrix of X.    inv(K(w_old))
    Xref (np.ndarray): Reference input matrix.      (w_new)
    m (int): Number of rows in Xref.    (n_w_new)
    d (float): Distance parameter.  (theta_y)
    g (float): Gaussian parameter.  (g)
    gvec (np.ndarray): Output g vector. 
    mui (np.ndarray): Output mui scalar.
    kx (np.ndarray): Output kx vector.
    kxy (np.ndarray): Output kxy vector.
    """
    # Sanity check
    if m == 0:
        assert kxy is None and Xref is None

    kx[:] = covar(col, x.reshape(1, -1), 1, X, n, d).flatten()
    if m > 0:
        kxy[:] = covar(col, x.reshape(1, -1), 1, Xref, m, d).flatten()
    # y=αAx+βy  Ki * kx
    # A (Ki) is a symmetric matrix.
    # x (kx) is a vector.
    # y (gvec) is the result vector.
    # α (1.0) and β (0.0) are scalars.
    gvec[:] = blas.dsymv(1.0, Ki, kx, 0.0)

    mui[0] = 1.0 + g - np.dot(kx, gvec)

    mu_neg = -1.0 / mui[0]
    gvec *= mu_neg
    
def calc_ktKikx(ktKik, m, k, n, g, mui, kxy, Gmui, ktGmui, ktKikx):
    """
    Calculates the ktKikx values for the given inputs.

    Parameters:
    ktKik (np.ndarray): Placeholder array (not used in this implementation).
    m (int): Number of candidates.
    k (np.ndarray): Matrix of covariances.
    n (int): Number of rows in k.
    g (np.ndarray): g vector.
    mui (float): Scalar value.
    kxy (np.ndarray): kxy vector.
    Gmui (np.ndarray): Placeholder array (not used in this implementation).
    ktGmui (np.ndarray): Placeholder array (not used in this implementation).
    ktKikx (np.ndarray): Output ktKikx vector.
    """
    def sq(x):
        return x * x

    # Loop over all of the m candidates
    for i in range(m):
        ktKikx[i] = sq(np.dot(k[i], g)) * mui
        ktKikx[i] += 2.0 * np.dot(k[i], g) * kxy[i]
        ktKikx[i] += sq(kxy[i]) / mui
        
def calc_alc(m, ktKik, s2p, phi, badj, tdf, w):
    """
    Calculates the ALC value for the given inputs.

    Parameters:
    m (int): Number of candidates.
    ktKik (np.ndarray): ktKik array.
    s2p (np.ndarray): s2p array (not used in this implementation).
    phi (float): Scalar value.
    badj (np.ndarray): Placeholder array (not used in this implementation).
    tdf (float): Scalar value (not used in this implementation).
    w (np.ndarray): Placeholder array (not used in this implementation).

    Returns:
    float: Calculated ALC value.
    """
    alc = 0.0
    for i in range(m):
        alc += phi * ktKik[i]
    
    return alc / m

def alcGP(col, X, Ki, d, g, n, phi, ncand, Xcand, nref, Xref, verb, alc):
    """
    #alcGP(n_w_col, w_old, inv(K(w_old)), theta_y, g, n_w_row, tau2, n_w_new_row, w_new, n_w_new_row, w_new, verb_in, alc_out)
    alcGP(col, X, Ki, d, g, n, phi, ncand, Xcand, nref, Xref, verb, alc):
    col: n_w_col
    X: w_old
    Ki: inv(K(w_old))
    d: theta_y
    g: g
    n: n_w_row
    phi: tau2
    ncand: n_w_new_row
    Xcand: w_new
    nref: n_w_new_row
    Xref: w_new
    """
    df = float(n)
    
    gvec = new_vector(n)
    kxy = new_vector(nref)
    kx = new_vector(n)
    ktKikx = new_vector(nref)
    
    k = new_matrix(nref, n)
    k = covar(col, Xref, nref, X, n, d)
    
    for i in range(ncand):
        mui = np.zeros(1)
        calc_g_mui_kxy(col, Xcand[i], X, n, Ki, Xref, nref, d, g, gvec, mui, kx, kxy)
        calc_ktKikx(None, nref, k, n, gvec, mui[0], kxy, None, None, ktKikx)
        alc[i] = calc_alc(nref, ktKikx, None, phi, None, df, None)
    
def new_matrix_bones(v, n1, n2):
    """
    Creates a 2D array from a 1D array.
    """
    if len(v) != n1 * n2:
        raise ValueError("The length of the input array does not match the specified dimensions.")
    
    return np.array(v).reshape(n1, n2)


def alcGP_py(X_in, n_in, col_in, Ki_in, d_in, g_in, ncand_in, Xcand_in, nref_in, Xref_in, phi_in, verb_in, alc_out):
    """
    n, col = X.shape
    ncand = Xcand.shape[0]
    nref = Xref.shape[0]
    alc_out = np.zeros(ncand)
    #alcGP_py(w_old, n_w_row, n_w_col, inv(K(w_old)), theta_y, g, n_w_new_row, w_new, n_w_new_row,  w_new, tau2, verb, alc_out)
    alcGP_py(X.T.flatten(), n, col, Ki.T.flatten(), theta, g, ncand, Xcand.T.flatten(), nref, Xref.T.flatten(), tau2, verb, alc_out)

    Parameters:
    X_in (array-like): Flattened array representing the input matrix X. (w_old)
    n_in (int): Number of rows in the input matrix X. (n_w_row)
    col_in (int): Number of columns in the input matrix X.  (n_w_col)
    Ki_in (array-like): Flattened array representing the Ki matrix. inv(K(w_old))
    d_in (float): Distance parameter.   (theta_y)
    g_in (float): Gaussian parameter.   (g)
    ncand_in (int): Number of candidate points. (n_w_new_row)
    Xcand_in (array-like): Flattened array representing the candidate points matrix.    (w_new)
    nref_in (int): Number of reference points.  (n_w_new_row)
    Xref_in (array-like): Flattened array representing the reference points matrix. (w_new)
    phi_in (float): Hyperparameter. (tau2)
    verb_in (int): Verbosity flag.
    alc_out (array-like): Array to store the output ALC values.
    """
    # Create matrices from flattened arrays
    Xref = new_matrix_bones(Xref_in, nref_in, col_in)
    Xcand = new_matrix_bones(Xcand_in, ncand_in, col_in)
    Ki = new_matrix_bones(Ki_in, n_in, n_in)
    X = new_matrix_bones(X_in, n_in, col_in)
    
    # Call the ALC function 
    #alcGP(n_w_col, w_old, inv(K(w_old)), theta_y, g, n_w_row, tau2, n_w_new_row, w_new, n_w_new_row, w_new, verb_in, alc_out)
    alcGP(col_in, X, Ki, d_in, g_in, n_in, phi_in, ncand_in, Xcand, nref_in, Xref, verb_in, alc_out)
    

def alc_f(X, Ki, theta, g, Xcand, Xref, tau2, verb=0):
    """
    K = Exp2(sq_dist(w), 1, object['theta_y'][t], object['g'][t])
    Ki = inv_det_py(K)['Mi']
    alc += alc_f(w, Ki, object['theta_y'][t], object['g'][t], w_new, ref, object['tau2'][t])


    Parameters:
    X (np.ndarray): Input matrix X. (w)
    Ki (np.ndarray): Ki matrix. inv(K(w_old))
    theta (float): Distance parameter.  (theta_y)
    g (float): Gaussian parameter.
    Xcand (np.ndarray): Candidate points matrix.    (w_new)
    Xref (np.ndarray): Reference points matrix. (w_new)
    tau2 (float): Hyperparameter.
    verb (int): Verbosity flag.

    Returns:
    np.ndarray: ALC values.
    """
    n, col = X.shape
    ncand = Xcand.shape[0]
    nref = Xref.shape[0]
    
    alc_out = np.zeros(ncand)
    
    #alcGP_py(w, n_w_row, n_w_col, inv(K(w_old)), theta_y, g, n_w_new_row, w_new, n_w_new_row,w_new, tau2, verb, alc_out)
    alcGP_py(X.T.flatten(), n, col, Ki.T.flatten(), theta, g, ncand, Xcand.T.flatten(), nref, Xref.T.flatten(), tau2, verb, alc_out)
    
    return alc_out

def alc_dgp2(object, x_new=None, ref=None, cores=1):
    tic = time.process_time()
    
    if object['v'] != 999:
        raise ValueError("ALC is only implemented for the un-approximated squared exponential kernel. Re-fit model with 'vecchia = FALSE' and 'cov = 'exp2' in order to use ALC.")
    
    if x_new is None:
        if 'x_new' not in object:
            raise ValueError("x_new has not been specified")
        else:
            x_new = object['x_new']
            if 'w_new' not in object:
                predicted = False
            else:
                predicted = True
    else:
        predicted = False

    if isinstance(x_new, np.ndarray) and len(x_new.shape) == 1:
        x_new = x_new.reshape(-1, 1)
    if ref is not None and not isinstance(ref, np.ndarray):
        raise ValueError('ref must be a matrix')
    
    n_new = x_new.shape[0]
    if not predicted:
        D = object['w'][0].shape[1]
        dx = sq_dist(object['x'])
        d_cross = sq_dist(x_new, object['x'])
    
    alc = np.zeros(n_new)
    for t in range(object['nmcmc']):
        w = object['w'][t]
        
        if predicted:
            w_new = object['w_new'][t]
        else:
            w_new = np.zeros((n_new, D))
            for i in range(D):
                w_new[:, i] = krig(w[:, i], dx, d_cross=d_cross, theta=object['theta_w'][t, i], g=eps, v=999)['mean']
        
        if ref is None:
            ref = w_new
        
        K = Exp2(sq_dist(w), 1, object['theta_y'][t], object['g'][t])
        Ki = inv_det_py(K)['Mi']
        alc += alc_f(w, Ki, object['theta_y'][t], object['g'][t], w_new, ref, object['tau2'][t])

    
    toc = time.process_time()
    return {'value': alc / object['nmcmc'], 'time': toc - tic}


#######1d usecase
np.random.seed(1980)
print('random is', np.random.rand(1))



root = os.getcwd() 
log_path =  os.path.join(root, 'log')
print('log_path', log_path)

seed = 1
layers = 1

print(f"seed is {seed}")
np.random.seed(seed)
print(f"layers is {layers}")

# Generate original data and reference grid
n = 10
new_n = 0
# new_n = 10
m = 100
noise = 0.1


# x = np.linspace(0, 1, n).reshape(-1, 1)

# Evaluate the objective function at these points and add noise
# y = np.apply_along_axis(f, 1, x) + np.random.normal(0, noise, n).reshape(-1,1)

x = np.array([1.36981,    -1.38577,
0.587107,    0.708629,
1.11078 ,   -1.35198,
0.604678,   -0.113224,
-1.09662,     0.86328,
-1.93585,    -2.09702,
-0.796596,    0.589165,
-0.172759 ,   0.317555,
0.414028 ,   0.952823,
-1.8498,   -0.231656,
1.11212 ,     0.4542,
0.711986 ,   -1.96284,
-0.0535862,   0.0855292,
1.35658 ,    1.17783,
-1.29087 ,    0.13915,
0.765691  ,  -1.04417,
-0.13442   , 0.297845,
-1.0086  ,  0.890088,
0.311205  ,  0.482937,
-0.00486998,     1.22764]).reshape((-1,2))

# x = np.array([1,2,3,4]).reshape((-1,2))
# print('x is', x)

y = np.array([-0.766594,
-0.522244,
-0.722181,
-0.562775,
 0.529142,
  2.80303,
 0.166224,
-0.266232,
-0.460391,
  2.62303,
-0.655071,
-0.667461,
 -0.33331,
-0.675061,
 0.760724,
-0.641081,
-0.285415,
 0.416961,
-0.449543,
-0.291749]).reshape((-1,1))

# y = np.array([3,7]).reshape((-1,1))


# x = lhs(1, samples=n)
# y = np.apply_along_axis(f, 1, x) + np.random.normal(0, noise, n)

# Set initial values for MCMC
g_0 = 0.01
if layers == 1:
    theta_0 = 0.5
elif layers == 2:
    theta_y_0 = 0.5
    theta_w_0 = 1
    w_0 = x
elif layers == 3:
    theta_y_0 = 0.5
    theta_w_0 = 1
    theta_z_0 = 1
    w_0 = x
    z_0 = x

# Set locations to store results
time_store = np.full(n + new_n + 1, np.nan)
rmse_store = np.full(n + new_n + 1, np.nan)
score_store = np.full(n + new_n + 1, np.nan)

x_new_list = []

# Conduct sequential design
for t in range(n, n + new_n + 1):
    print(f'Running {t} out of {n + new_n}')
    
    # Select new random set of candidate/testing points
    # xx = lhs(1, samples=m)
    # yy = np.apply_along_axis(f, 1, xx)
    xx = np.linspace(0, 1, m).reshape(-1, 1)
    yy = np.apply_along_axis(f, 1, xx).reshape(-1,1)

    
    if t == n:
        nmcmc = 100
        burn = 8000
        thin = 2
    else:
        nmcmc = 3000
        burn = 1000
        thin = 2
    
    # Fit Model
    fit = fit_one_layer(x, y, nmcmc=nmcmc, g_0=g_0, theta_0=theta_0)
    print('param theta is', fit['theta'][nmcmc-1])
    print('param tau2 is', fit['tau2'][nmcmc-1])

#     # Trim, predict, and calculate ALC
#     fit = trim_dgp2(fit, burn=burn, thin=thin)
#     fit = predict_dgp2(fit, xx, lite=False)
#     alc_list = alc_dgp2(fit)
#     alc = alc_list['value']
#     fit_time = fit['time'] + alc_list['time']
    
#     # Store metrics
#     time_store[t] = fit_time
#     rmse_store[t] = rmse(yy, fit['mean'])
#     score_store[t] = score(yy, fit['mean'].reshape((-1,1)), fit['Sigma'])
#     print('yy', yy, yy.shape)
#     print('fit mean', fit['mean'], fit['mean'].shape)
#     print('rmse_score', rmse_store[t], rmse_store[t].shape)
#     with open(log_path  + '.txt', 'a') as file:
#         # print('yy{}: {}, {}'.format(t, yy, yy.shape), file=file)
#         # print('fit mean{}: {}, {}'.format(t, fit['mean'], fit['mean'].shape), file=file)
#         print('rmse{}: {}, {}'.format(t, rmse_store[t], rmse_store[t].shape), file=file)
#         print('score{}: {}, {}'.format(t, score_store[t], rmse_store[t].shape), file=file)
#         print('x_new_list{}: {}'.format(t, x_new_list),file=file)
#         file.flush()
    
#     print('fit sigma', np.diag(fit['Sigma']), np.diag(fit['Sigma']).shape)
#     print('fit alc', alc, alc.shape)
#     plot = {'mean': fit['mean'], 'sigma': np.diag(fit['Sigma']), 'alc': alc, 'rmse': rmse_store[t], 'score': score_store[t]}
    
    
#     with open(f'plot{t}.pyc', 'wb') as file:
#         pickle.dump(plot, file)
#         file.close()
    
#     # with open('plot.pkl', 'rb') as f:
#     #     plot = pickle.load(f)

#     if t == n + new_n:
#         break
    
#     # Select next design point
#     x_new = xx[np.argmax(alc)]
#     x_new_list.append(x_new)
#     print('x_new', x_new)
#     x = np.vstack((x, x_new))
#     y = np.append(y, f(x_new) + np.random.normal(0, noise))
    
#     # Adjust starting locations for the next iteration
#     g_0 = fit['g'][fit['nmcmc'] - 1]
#     theta_y_0 = fit['theta_y'][fit['nmcmc'] - 1]
#     theta_w_0 = fit['theta_w'][fit['nmcmc'] - 1]
#     w_0 = fit['w'][fit['nmcmc'] - 1]
         
        
#         # Save results
#     results_df = pd.DataFrame({
#         'time': time_store,
#         'rmse': rmse_store,
#         'score': score_store
#     })
#     # results_df.to_csv(f"1D_layer{layers}_seed{seed}.csv", index=False)

# # Save final results
# results_df = pd.DataFrame({
#     'time': time_store,
#     'rmse': rmse_store,
#     'score': score_store
# })
# # results_df.to_csv(f"1D_layer{layers}_seed{seed}.csv", index=False)

# print('x_new_list', x_new_list)

# #######
