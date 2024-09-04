import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

font_path = 'tools/TIMES.TTF'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
front_size = 12

def plot_pre(pre, re_data_fcpflow, scaler, con_dim, re_data_list=None, _sample_index=0, path='Generated_Data_Comparison.png'):
    # Inverse transform to get the original scale of the data for FCPflow and other models
    orig_data_pre = scaler.inverse_transform(pre.cpu().detach().numpy())
    orig_data_re_fcpflow = scaler.inverse_transform(re_data_fcpflow.cpu().detach().numpy())
    
    _real_pre = orig_data_pre[_sample_index, con_dim:]
    _cond = orig_data_pre[_sample_index, :con_dim]
    predict_pre_fcpflow = orig_data_re_fcpflow[:, con_dim:]
    
    # Calculate 95% prediction interval for FCPflow model
    lower_bound = np.percentile(predict_pre_fcpflow, 5, axis=0)
    upper_bound = np.percentile(predict_pre_fcpflow, 95, axis=0)

    # Calculate average prediction for FCPflow
    avg_prediction_fcpflow = np.mean(predict_pre_fcpflow, axis=0)

    # Plot the real condition and data
    _len_con, _len_pre = len(_cond), len(_real_pre)
    plt.figure(figsize=(12, 4))

    # Plot the real condition in black
    plt.plot(range(0, _len_con), _cond, color='black', label='Real condition', linewidth=1.5)

    # Plot the 95% prediction interval for FCPflow (light blue)
    plt.fill_between(range(_len_con, _len_con + _len_pre), lower_bound, upper_bound, color='lightblue', alpha=0.5, label='90% Prediction Interval (FCPflow)')

    # Plot the average prediction for FCPflow using a thicker purple line
    plt.plot(range(_len_con, _len_con + _len_pre), avg_prediction_fcpflow, color='purple', label='Mean Prediction (FCPflow)', linewidth=2.5)

    # Plot the real data in blue, labeled as "True"
    plt.plot(range(_len_con, _len_con + _len_pre), _real_pre, color='blue', label='True', linewidth=1.5)

    # Plot the average predictions for each dataset in re_data_list with requested colors and line styles
    model_names = ['cWGAN-GP', 'cNICE', 'cVAE']
    colors = ['green', 'red', 'orange']
    linestyles = ['--', '--', '--']
    
    if re_data_list is not None:
        for idx, re_data_item in enumerate(re_data_list):
            orig_data_re_item = scaler.inverse_transform(re_data_item.cpu().detach().numpy())
            avg_prediction_item = np.mean(orig_data_re_item[:, con_dim:], axis=0)
            plt.plot(range(_len_con, _len_con + _len_pre), avg_prediction_item, label=f'Mean Prediction ({model_names[idx]})', color=colors[idx], linestyle=linestyles[idx], linewidth=1)

    # Add grid, labels, legend, and limits
    plt.grid(True)
    plt.xlim(0, _len_con + _len_pre)
    plt.ylim(min(_real_pre)-5, max(_real_pre) * 1.3)
    plt.xticks(fontsize=front_size, fontproperties=font_prop)
    plt.yticks(fontsize=front_size, fontproperties=font_prop)

    # Add labels and legend
    plt.xlabel('Time [Hours]', fontsize=front_size, fontproperties=font_prop)
    plt.ylabel('Consumption [kWh]', fontsize=front_size, fontproperties=font_prop)
    plt.legend(fontsize=front_size, prop=font_prop, loc='lower left', bbox_to_anchor=(0, 0))

    # Save the figure
    plt.savefig(path)
    plt.close()
