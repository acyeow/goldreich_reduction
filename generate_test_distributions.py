import numpy as np

def generate_far_exact_distribution(n):
    """
    Generate a distribution that is far from uniform.

    Parameters:
    n (int): The size of the domain.

    Returns:
    np.ndarray: The far exact distribution.
    """
    distribution = np.zeros(n)
    distribution[0] = 0.7
    distribution[1:] = 0.3 / (n - 1)
    return distribution

def generate_bimodal_distribution(n, peak1=0.3, peak2=0.7, std=0.1):
    """
    Generate a bimodal distribution over a domain of size n.

    Parameters:
    n (int): The size of the domain.
    peak1 (float): The first peak location.
    peak2 (float): The second peak location.
    std (float): The standard deviation of the peaks.

    Returns:
    np.ndarray: The bimodal distribution.
    """
    x = np.linspace(0, 1, n)
    dist1 = np.exp(-0.5 * ((x - peak1) / std)**2)
    dist2 = np.exp(-0.5 * ((x - peak2) / std)**2)
    distribution = dist1 + dist2
    return distribution / np.sum(distribution)

def generate_normal_distribution(n, mean=None, std=None):
    """
    Generate a Gaussian distribution over a domain of size n.

    Parameters:
    n (int): The size of the domain.
    mean (float): The mean of the Gaussian distribution.
    std (float): The standard deviation of the Gaussian distribution.

    Returns:
    np.ndarray: The Gaussian distribution.
    """
    if mean is None:
        mean = n / 2
    if std is None:
        std = n / 10  # Default spread
    x = np.arange(n)
    distribution = np.exp(-0.5 * ((x - mean) / std)**2)
    return distribution / np.sum(distribution)

def generate_exponential_distribution(n, scale=None):
    """
    Generate an exponential distribution over a domain of size n.

    Parameters:
    n (int): The size of the domain.
    scale (float): The scale parameter of the exponential distribution.

    Returns:
    np.ndarray: The exponential distribution.
    """
    if scale is None:
        scale = n / 10
    x = np.arange(n)
    distribution = np.exp(-x / scale)
    return distribution / np.sum(distribution)