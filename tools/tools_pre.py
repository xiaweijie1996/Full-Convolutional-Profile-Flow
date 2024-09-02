import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_pre(pre, re_data, scaler, con_dim, _sample_index=0, path='Generated_Data_Comparison.png'):
    # Inverse transform to get the original scale of the data
    orig_data_pre = scaler.inverse_transform(pre.cpu().detach().numpy())
    orig_data_re = scaler.inverse_transform(re_data.cpu().detach().numpy())
    
    _real_pre = orig_data_pre[_sample_index, con_dim:]
    _cond = orig_data_pre[_sample_index, :con_dim]
    predict_pre = orig_data_re[:, con_dim:]
    
    # Calculate 95% prediction interval
    lower_bound = np.percentile(predict_pre, 2.5, axis=0)
    upper_bound = np.percentile(predict_pre, 97.5, axis=0)

    # Calculate average prediction
    avg_prediction = np.mean(predict_pre, axis=0)

    # Plot the real condition and data
    _len_con, _len_pre = len(_cond), len(_real_pre)
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, _len_con), _cond, color='blue', label='Real condition')
    
    # Plot the 95% prediction interval (light blue)
    plt.fill_between(range(_len_con, _len_con + _len_pre), lower_bound, upper_bound, color='lightblue', alpha=0.5, label='95% Prediction Interval')
    
    # Plot the average prediction
    plt.plot(range(_len_con, _len_con + _len_pre), avg_prediction, color='red', label='Average Prediction')
    
    # Plot the real data
    plt.plot(range(_len_con, _len_con + _len_pre), _real_pre, color='yellow', label='Real data')
    
    # Add grid
    plt.grid(True)

    # Add legend and save the figure
    plt.legend()
    plt.savefig(path)
    plt.close()


