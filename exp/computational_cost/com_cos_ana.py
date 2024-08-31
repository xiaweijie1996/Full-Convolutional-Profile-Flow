import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = 'tools/TIMES.TTF'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
front_size = 15

# ------------MMD Comparison IN Steps------------
# Read the data
path_mmd = 'exp/computational_cost/step_mmd_com.csv'
data_mmd_org = pd.read_csv(path_mmd, sep=';')

# Convert relevant columns to numeric
for col in ['WGAN-GP - MMD', 'NICE - MMD', 'VAE - MMD', 'FCPFLOW - MMD']:
    data_mmd_org [col] = pd.to_numeric(data_mmd_org [col], errors='coerce')

# Allow user to define the maximum step to be plotted
max_step = 260000  # Default to the maximum step in the data

# Filter data based on the maximum step
data_mmd = data_mmd_org[data_mmd_org ['Step'] <= max_step]

# Plot the data with transparency and smoothed curves
plt.figure(figsize=(9, 4.6))

# List of models
models = ['NICE - MMD', 'VAE - MMD', 'WGAN-GP - MMD', 'FCPFLOW - MMD']
colors = ['blue', 'orange', 'green', 'red']  # Different colors for each model

for i, model in enumerate(models):
    # Plot the original curve with transparency
    plt.plot(data_mmd['Step'], data_mmd[model], color=colors[i], alpha=0.3)
    
    # Calculate and plot the smoothed curve using a rolling mean
    smoothed_curve = data_mmd[model].rolling(window=100, min_periods=1).mean()
    plt.plot(data_mmd['Step'], smoothed_curve, label=f'{model} (Smoothed)', color=colors[i])

    # Find and print the minimum MMD value and its corresponding step
    min_mmd = data_mmd[model].min()
    min_step = data_mmd['Step'][data_mmd[model].idxmin()]
    print(f'Minimum {model}: {min_mmd} at Step {min_step}')
    # tick size
    plt.xticks(fontsize=front_size, fontproperties=font_prop)
    plt.yticks(fontsize=front_size, fontproperties=font_prop)
    # limit the x-axis
    plt.xlim(0, max_step)
    # limit the y-axis
    plt.ylim(0, 0.8)

# Add legend
plt.legend(fontsize=front_size, prop=font_prop)

# Add labels and title
plt.xlabel('Step [-]', fontsize=front_size, fontproperties=font_prop)
plt.ylabel('MMD [-]', fontsize=front_size, fontproperties=font_prop)
# plt.title('MMD of Different Models over Steps', fontsize=front_size, fontproperties=font_prop)
# Show plot
plt.savefig('exp/computational_cost/mmd_comparison_in_steps.png')
# ------------MMD Comparison IN Steps------------


# ------------MMD Comparison IN Time------------
path_time = 'exp/computational_cost/step_time_com.csv'
data_time_org = pd.read_csv(path_time, sep=',')
saved_columns = ['Step', 'NICE - time', 'VAE - time', 'WGAN-GP - time', 'FCPFLOW - time']
data_time_org = data_time_org[saved_columns]

# List of models and their corresponding 'time' columns
models = ['NICE', 'VAE', 'WGAN-GP', 'FCPFLOW']

# Loop through each model and find the max time and corresponding step
for model in models:
    time_column = f'{model} - time'
    # Find the max time value
    max_time = data_time_org[time_column].max()
    # Find the step corresponding to the max time value
    max_step = data_time_org['Step'][data_time_org[time_column].idxmax()]
    print(f'Maximum {model} time: {max_time} at Step {max_step}')
    # print the average time per step time/step
    print(f'Average {model} time per step: {max_time/max_step}')
        
    # based on the average time per step, we can estimate the time for a specific step
    # add a new column to the mmd data frame for the estimated time for each model
    data_mmd[f'{model} - estimated time'] = data_mmd['Step'] * (max_time/max_step)


# ------------MMD Comparison IN Time------------
# Use time as the x-axis to plot the MMD values
max_time = data_mmd['VAE - estimated time'].max()  # Set this to the maximum time you want to consider

plt.figure(figsize=(9, 4.6))
for i, model in enumerate(models):
    # Filter the data based on max_time
    filtered_data = data_mmd[data_mmd[f'{model} - estimated time'] <= max_time]
    
    # Plot the original curve with transparency
    _colum_name = f'{model} - MMD'
    plt.plot(filtered_data[f'{model} - estimated time'], filtered_data[_colum_name], color=colors[i], alpha=0.3)
    
    # Calculate and plot the smoothed curve using a rolling mean
    smoothed_curve = filtered_data[_colum_name].rolling(window=100, min_periods=1).mean()
    plt.plot(filtered_data[f'{model} - estimated time'], smoothed_curve, label=f'{model} (Smoothed)', color=colors[i])
    
    # Find and print the minimum MMD value and its corresponding time within the max_time range
    min_mmd = filtered_data[_colum_name].min()
    min_time = filtered_data[f'{model} - estimated time'][filtered_data[_colum_name].idxmin()]
    print(f'Minimum {model}: {min_mmd} at Time {min_time}')
    
    # Set tick size
    plt.xticks(fontsize=front_size, fontproperties=font_prop)
    plt.yticks(fontsize=front_size, fontproperties=font_prop)
    
    # Limit the x-axis to max_time
    plt.xlim(0, max_time)
    
    # Limit the y-axis
    plt.ylim(0, 0.8)

# Add legend
plt.legend(fontsize=front_size, prop=font_prop)

# Add labels and title
plt.xlabel('Time [Second]', fontsize=front_size, fontproperties=font_prop)
plt.ylabel('MMD [-]', fontsize=front_size, fontproperties=font_prop)
# plt.title('MMD of Different Models over Time', fontsize=front_size, fontproperties=font_prop)

# Show plot
plt.savefig('exp/computational_cost/mmd_comparison_in_time.png')

    