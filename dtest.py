import numpy as np



def squared_exponential_covariance(x1, x2, tau2, theta, g):
    """
    Compute the Squared Exponential covariance between two vectors.

    Parameters:
    x1 : numpy.ndarray
        Input vector x.
    x2 : numpy.ndarray
        Input vector x'.
    tau2 : float
        Variance term (sigma_f^2).
    theta : numpy.ndarray
        Length scale parameter vector (l).
    g : float
        Noise term (sigma_n^2).

    Returns:
    k : float
        Covariance between x and x'.
    """
    # Compute the scaled distance r_l(x, x')
    r_l = np.sqrt(np.sum(((x1 - x2) / theta) ** 2))
    
    print('rl', r_l)
    
    # Compute the covariance k(x, x')
    k = tau2 * np.exp(-0.5 * r_l**2) + g * (x1 == x2).all()
    
    return k

# Example usage:
x1 = np.array([1.0, 2.0, 3.0])
x2 = np.array([2, 4, 6])
tau2 = 1.0
theta = np.array([1.0, 2.0, 3.0])
g = 0.1

covariance = squared_exponential_covariance(x1, x2, tau2, theta, g)
print("Covariance:", covariance)


# a = np.zeros((2,2))
# b = np.full((2,1),1)
# a[:,0] = b
# print(a.shape, a[0].shape, a[0,:].shape, a[:,0].shape, b.shape)