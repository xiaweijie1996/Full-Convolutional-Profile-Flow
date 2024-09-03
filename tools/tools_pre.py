import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

font_path = 'tools/TIMES.TTF'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
front_size = 15


def plot_pre(pre, re_data, scaler, con_dim, re_data_list=None, _sample_index=0, path='Generated_Data_Comparison.png'):
    # Inverse transform to get the original scale of the data
    orig_data_pre = scaler.inverse_transform(pre.cpu().detach().numpy())
    orig_data_re = scaler.inverse_transform(re_data.cpu().detach().numpy())
    
    _real_pre = orig_data_pre[_sample_index, con_dim:]
    _cond = orig_data_pre[_sample_index, :con_dim]
    predict_pre = orig_data_re[:, con_dim:]
    
    # Calculate 95% prediction interval for the main re_data
    lower_bound = np.percentile(predict_pre, 2.5, axis=0)
    upper_bound = np.percentile(predict_pre, 97.5, axis=0)

    # Calculate average prediction for the main re_data
    avg_prediction = np.mean(predict_pre, axis=0)

    # Plot the real condition and data
    _len_con, _len_pre = len(_cond), len(_real_pre)
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, _len_con), _cond, color='blue', label='Real condition')
    
    # Plot the 95% prediction interval (light blue)
    plt.fill_between(range(_len_con, _len_con + _len_pre), lower_bound, upper_bound, color='lightblue', alpha=0.5, label='95% Prediction Interval')
    
    # Plot the average prediction for the main re_data
    plt.plot(range(_len_con, _len_con + _len_pre), avg_prediction, color='red', label='Average Prediction')
    
    # Plot the real data
    plt.plot(range(_len_con, _len_con + _len_pre), _real_pre, color='yellow', label='Real data')

    # Plot the average curves for each dataset in re_data_list, if provided
    if re_data_list is not None:
        for idx, re_data_item in enumerate(re_data_list):
            orig_data_re_item = scaler.inverse_transform(re_data_item.cpu().detach().numpy())
            avg_prediction_item = np.mean(orig_data_re_item[:, con_dim:], axis=0)
            plt.plot(range(_len_con, _len_con + _len_pre), avg_prediction_item, label=f'Average Prediction Set {idx + 1}')
    
    # Add grid
    plt.grid(True)
    plt.xlim(0, _len_con + _len_pre)
    plt.ylim(0, max(_real_pre) * 1.1)
    plt.xticks(fontsize=front_size, fontproperties=font_prop)
    plt.yticks(fontsize=front_size, fontproperties=font_prop)

    # Add legend and save the figure
    plt.xlabel('Time [Hours]', fontsize=front_size, fontproperties=font_prop)
    plt.ylabel('Consumption [kWh]', fontsize=front_size, fontproperties=font_prop)
    plt.legend(fontsize=front_size, prop=font_prop)
    plt.savefig(path)
    plt.close()
