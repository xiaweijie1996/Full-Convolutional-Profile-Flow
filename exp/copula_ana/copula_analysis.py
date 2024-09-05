# this script is used to answer the review questions of the why copula fails to generate the peak samples

import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, gaussian_kde
from scipy.interpolate import interp1d
from multicopula_model import EllipticalCopula

# data path
copula_path = r'C:\Users\weijiexia\OneDrive - Delft University of Technology\Align4Energy\paper2\test\uk_ind\copula_samples_uk.csv'
fctflow_path = r'C:\Users\weijiexia\OneDrive - Delft University of Technology\Align4Energy\paper2\test\uk_ind\fctflow_samples_uk_2.csv'
original_path = r'C:\Users\weijiexia\OneDrive - Delft University of Technology\Align4Energy\paper2\data\uk_data_cleaned_ind_test.csv'
original_path_train = r'C:\Users\weijiexia\OneDrive - Delft University of Technology\Align4Energy\paper2\data\uk_data_cleaned_ind_train.csv'
# read the data
copula_data = pd.read_csv(copula_path, index_col=0).values[:,:-2]
fctflow_data = pd.read_csv(fctflow_path, index_col=0)
original_data = pd.read_csv(original_path, index_col=0).values[:,:-2]
fctflow_data = fctflow_data.sample(frac=1).reset_index(drop=True)
fctflow_data = np.array(fctflow_data.iloc[:4000, :])[:,:-2]

original_data_train = pd.read_csv(original_path_train, index_col=0).values[:,:-2]

#--------------------------------------------------------------------------------


print('copula_data shape:', copula_data.shape, 'fctflow_data shape:', fctflow_data.shape, 'original_data shape:', original_data.shape)

# reduce the dimensionality of the data to 2 using PCA
pca = PCA(n_components=2)
original_data_pca = pca.fit_transform(original_data)
copula_data_pca = pca.transform(copula_data)
fctflow_data_pca = pca.transform(fctflow_data)
print('copula_data_pca shape:', copula_data_pca.shape, 'fctflow_data_pca shape:', fctflow_data_pca.shape, 'original_data_pca shape:', original_data_pca.shape)

# Plot the data
plt.figure()
plt.scatter(original_data_pca[:, 0], original_data_pca[:, 1], label='Original', alpha=0.7)
plt.scatter(copula_data_pca[:, 0], copula_data_pca[:, 1], label='Copula', alpha=0.7)
plt.scatter(fctflow_data_pca[:, 0], fctflow_data_pca[:, 1], label='FCPflow', alpha=0.7)

# Add labels to the axes
plt.xlabel('PCA Dim 1')
plt.ylabel('PCA Dim 2')

# Adjust the legend's position to the top (outside of the plot)
plt.legend(bbox_to_anchor=(0.5, 1.3), loc='upper center', ncol=3)

# Show the plot
plt.title('PCA of the original, copula and FCPflow data')
plt.show()

#--------------------------------------------------------------------------------

# Function to compute empirical CDF for each dimension
def compute_empirical_marginal_cdf(a):
    sorted_dim1 = np.sort(a[:, 0])
    sorted_dim2 = np.sort(a[:, 1])

    cdf_dim1 = np.arange(1, len(sorted_dim1) + 1) / len(sorted_dim1)
    cdf_dim2 = np.arange(1, len(sorted_dim2) + 1) / len(sorted_dim2)

    cdf_func_dim1 = interp1d(sorted_dim1, cdf_dim1, kind='linear', bounds_error=False, fill_value=(0, 1))
    cdf_func_dim2 = interp1d(sorted_dim2, cdf_dim2, kind='linear', bounds_error=False, fill_value=(0, 1))

    return cdf_func_dim1, cdf_func_dim2

# Convert the data to the empirical CDF
original_cdf1, original_cdf2 = compute_empirical_marginal_cdf(original_data_pca)

# Evaluate the CDF of all original data
original_cdf_values1 = original_cdf1(original_data_pca[:, 0])
original_cdf_values2 = original_cdf2(original_data_pca[:, 1])

# Combine the CDF values (using the mean in this case)
combined_cdf_values = np.vstack((original_cdf_values1, original_cdf_values2)).T

# fit t distribution to the combined CDF values
copula_model = EllipticalCopula(original_data_pca.T)
copula_model.fit()
samples, inv_collection = copula_model.sample(n_samples=4000)
inv_collection = np.array(inv_collection).T
samples = samples.T

samples_cdf1 = original_cdf1(samples[:, 0])
samples_cdf2 = original_cdf2(samples[:, 1])
combined_cdf_values_copula = np.vstack((samples_cdf1, samples_cdf2))

# Plot the CDF values
plt.figure()
plt.scatter(original_cdf_values1 , original_cdf_values2, alpha=0.5, label='Original')
plt.scatter(samples_cdf1, samples_cdf2, alpha=0.3, label='t-Copula')
plt.xlabel('Value of u1')
plt.ylabel('Value of u2')
plt.title('(u1,u2) distribution of the original and Copula')
plt.legend(bbox_to_anchor=(0.5, 1.3), loc='upper center', ncol=2)
plt.show()

#--------------------------------------------------------------------------------

# Plot the data
plt.figure()
plt.scatter(original_data_pca[:, 0], original_data_pca[:, 1], label='Original', alpha=0.7)
plt.scatter(copula_data_pca[:, 0], copula_data_pca[:, 1], label='Copula', alpha=0.7)
plt.scatter(fctflow_data_pca[:, 0], fctflow_data_pca[:, 1], label='FCPflow', alpha=0.7)
plt.scatter(samples[:, 0], samples[:, 1], label='t-Distribution', alpha=0.7)
# Add labels to the axes
plt.xlabel('PCA Dim 1')
plt.ylabel('PCA Dim 2')

# Adjust the legend's position to the top (outside of the plot)
plt.legend(bbox_to_anchor=(0.5, 1.3), loc='upper center', ncol=3)

# Show the plot
plt.title('PCA of the original, copula and FCPflow data')
plt.show()