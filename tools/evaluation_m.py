import numpy as np
import pandas as pd
from scipy.stats import kendalltau, ks_2samp, energy_distance, wasserstein_distance
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
import torch

def calculate_energy_distances(dataset1, dataset2, num_samples=1000):
    """
    Calculate the energy distances between two dataset.

    """
    dataset1 = np.array(dataset1).flatten()
    dataset2 = np.array(dataset2).flatten()
    
    return energy_distance(dataset1, dataset2)

def calculate_w_distances(dataset1, dataset2, num_samples=1000):
    """
    Calculate the energy distances between two dataset.

    """
    dataset1 = np.array(dataset1).reshape(-1)
    dataset2 = np.array(dataset2).reshape(-1)
    return wasserstein_distance(dataset1, dataset2)
    
    
def kendalltau_corr(data):
    """compute the Kendall Tau correlation between one time series data"""
    data = np.array(data)
    correlation_matrix = np.zeros((data.shape[1], data.shape[1]))
    
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            correlation_matrix[i, j] = kendalltau(data[:, i], data[:, j])[0]
            
    return correlation_matrix


def calculate_autocorrelation_mse(dataset1, dataset2):
    """
    Calculate the correlation and mean square error between two dataset.

    """
    dataset1 = np.array(dataset1)
    dataset2 = np.array(dataset2)
    
    # compute the correlation matrix of data1
    correlation_matrix1 = kendalltau_corr(dataset1)
    
    # compute the correlation matrix of data2
    correlation_matrix2 = kendalltau_corr(dataset2)
    
    # compute the mean square error
    mse = mean_squared_error(correlation_matrix1, correlation_matrix2)
    
    return mse
    
    
# ks distance
def ks_distance(data1, data2):
    """compute the Kolmogorov-Smirnov distance between two time series data"""
    data1 = np.array(data1)
    data2 = np.array(data2)
    data1 = data1.reshape(-1)
    data2 = data2.reshape(-1)
    
    data1 = data1[np.random.choice(data1.shape[0], data1.shape[0], replace=False)]
    data2 = data2[np.random.choice(data2.shape[0], data2.shape[0], replace=False)]
    return ks_2samp(data1, data2)[0]


# Gaussian Kernel function with bandwidth
def gaussian_kernel(matrix1, matrix2, bandwidth):
    dists = euclidean_distances(matrix1, matrix2, squared=True)  # squared Euclidean distance
    return np.exp(-dists / bandwidth)

def MMD_kernel(a, b, bandwidths = [0.5, 1, 5, 10]):
    kernels_a = np.mean([gaussian_kernel(a, a, bandwidth) for bandwidth in bandwidths], axis=0)
    kernels_b = np.mean([gaussian_kernel(b, b, bandwidth) for bandwidth in bandwidths], axis=0)
    kernels_ab = np.mean([gaussian_kernel(a, b, bandwidth) for bandwidth in bandwidths], axis=0)
    return np.mean(kernels_a) + np.mean(kernels_b) - 2*np.mean(kernels_ab)

def compute_quantile(pre_data):
    q = np.arange(0.01, 1, 0.01)
    quantile_value = np.quantile(pre_data, q, axis=0)
    return quantile_value 

def pinball_loss_compute(y, f, q):
    if y >= f:
        return (y - f) * q
    else:
        return (f - y) * (1 - q)
    
def plloss(pre_data, true_data):

    quantile_values = compute_quantile(pre_data)
    q = np.arange(0.01, 1, 0.01)
    
    loss = []
    for i in range(len(q)):
        ind_loss = [pinball_loss_compute(y, f, q[i]) for y, f in zip(true_data, quantile_values[i,:])]
        loss.append(np.mean(ind_loss))
    pinball_loss = np.mean(loss)

    return pinball_loss

def compute_mse(pre_data, true_data):
    mean_pre = np.mean(pre_data, axis=0)
    mean_true = np.mean(true_data, axis=0)
    return mean_squared_error(mean_pre, mean_true)


def pinball_loss(y_true, y_pred, quantile):
    delta = y_true - y_pred
    return torch.mean(torch.max(quantile * delta, (quantile - 1) * delta))

def crps(y_true, y_pred_samples):
    """
    Calculate the CRPS (Continuous Ranked Probability Score).
    
    Args:
        y_true (ndarray): The true values.
        y_pred_samples (ndarray): Samples from the predictive distribution.

    Returns:
        float: The CRPS score.
    """
    # Sort predictions and calculate differences
    sorted_pred_samples = np.sort(y_pred_samples, axis=0)
    diff = sorted_pred_samples - y_true
    n = y_pred_samples.shape[0]
    
    crps_score = np.mean((2 * np.arange(1, n+1) - 1) * diff, axis=0) / n
    return np.mean(crps_score)
