import os
import sys
_parent_path =os.path.join(os.path.dirname(__file__), '..','..','..')
sys.path.append(_parent_path)

import pandas as pd
import time
import matplotlib.pyplot as plt

from alg.copula_model.multicopula_model import EllipticalCopula
from tools.evaluation_m import MMD_kernel, calculate_w_distances

# define the data path and read 
path = os.path.join(_parent_path, 'data/nl_data_1household.csv')
data = pd.read_csv(path).iloc[:,3:-2].values
# drop nan
data = data[~pd.isna(data).any(axis=1)]

# define the copula model
start = time.time()
copula = EllipticalCopula(data.T)
copula.fit()
# sample the data
samples = copula.sample(data.shape[0])
  

# plot the data
plt.figure()
plt.plot(samples, label='Real data', alpha=0.3)
save_path = os.path.join(_parent_path, 'exp/computational_cost/copula/copula_time_cost.png')
plt.savefig(save_path)

# calculate the MMD
mmd = MMD_kernel(data, samples.T)
end = time.time() 

print('mmd: ', mmd)
print('Data shape: ', data.shape)
print('Samples shape: ', samples.shape)
print('Time cost: ', end-start)