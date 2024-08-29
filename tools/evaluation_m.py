import numpy as np
import pandas as pd
from scipy.stats import kendalltau, ks_2samp, energy_distance, wasserstein_distance
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances

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
