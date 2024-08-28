import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_max_and_index(*args):
    results = []
    for array in args:
        # Find the maximum value in each row
        max_values = np.max(array, axis=1, keepdims=True)
        # Find the index of the maximum value in each row
        max_indices = np.argmax(array, axis=1, keepdims=True)
        # Append the results as a tuple
        results.append((max_values, max_indices))
    return results

def plot_peak_times(results, resolution, country, models):
    all_peak_times = []
    colors = plt.cm.get_cmap('tab10', len(models))  # Use a colormap to get distinct colors
    plt.figure(figsize=(10, 5))
    _size = 18
    # ave_points = []
    for i, (max_values, max_indices) in enumerate(results):
        peak_times = (max_indices * resolution).flatten()
        all_peak_times.extend(peak_times)
        peak_times_in_hours = peak_times / 60.0
        plt.scatter(peak_times_in_hours, max_values.flatten(), alpha=0.1, color=colors(i))
        avg_peak_time_model = np.mean(peak_times)
        avg_peak_value = np.mean(max_values)
        avg_peak_time_hours = avg_peak_time_model // 60
        avg_peak_time_minutes = avg_peak_time_model % 60
        avg_peak_time_in_hours = avg_peak_time_model / 60.0
        time_variance = np.std(peak_times_in_hours)
        value_variance = np.std(max_values)
        plt.scatter(avg_peak_time_in_hours, avg_peak_value, color=colors(i), edgecolor='black', s=100, label=f'Avg {models[i]} ({int(avg_peak_time_hours)}:{int(avg_peak_time_minutes):02d})')
        # plt.plot([avg_peak_time_in_hours - time_variance, avg_peak_time_in_hours + time_variance], [avg_peak_value, avg_peak_value], 
        #          color=colors(i), linestyle='--',linewidth=3.0, alpha=1)
        # plt.plot([avg_peak_time_in_hours, avg_peak_time_in_hours], [avg_peak_value - value_variance, avg_peak_value + value_variance], 
        #          color=colors(i), linestyle='--',linewidth=3.0, alpha=1)
        plt.xlim(0, 24)
        plt.ylim(0, None)
        plt.xticks(fontsize=_size)
        plt.yticks(fontsize=_size)
        # ave_points.append((avg_peak_time_in_hours, avg_peak_value))
    plt.xlabel('Time of Day (Hours)', fontsize=_size)
    plt.ylabel('Peak Value', fontsize=_size)
    plt.title(f'Peak Values and Peak Times for {country.upper()} Data', fontsize=_size)
    plt.legend(loc='upper right', fontsize=_size)
    plt.grid(True)
    plt.savefig(f'exp/peak_analysis/{country}_peak_times.png')
    plt.show()
    # compute the distance between the average peak times
    # avg_peak_times = [point[0] for point in ave_points]
    # i = 0
    # print(avg_peak_times )
    # for _time, _value in ave_points:
    #     distances_time = (_time-avg_peak_times[0][0])**2
    #     distances_value = (_value-ave_points[0][1])**2
    #     edu_distance = np.sqrt(distances_time + distances_value)
    #     print(f'{models[i]} vs. Original: {edu_distance:.2f}')
    #     i += 1


# ---------- analysis the peak of the of ge data ----------
path_original = 'data/ge_data_ind.csv'
path_copula = 'exp/peak_analysis/data/ge/copula_samples_ge.csv'
path_fcpflow = 'exp/peak_analysis/data/ge/fctflow_samples_ge.csv'
path_gmm = 'exp/peak_analysis/data/ge/gmm_samples_ge.csv'
path_wgan = 'exp/peak_analysis/data/ge/wgan_samples_ge.csv'

data_original = pd.read_csv(path_original).values
copula_data = pd.read_csv(path_copula, index_col=0).values
fcpflow_data = pd.read_csv(path_fcpflow, index_col=0).values
gmm_data = pd.read_csv(path_gmm, index_col=0).values

original_max_vi = get_max_and_index(data_original, copula_data, fcpflow_data, gmm_data)
models = ['Original', 'Copula', 'FCPflow', 'GMM']
plot_peak_times(original_max_vi, 15, 'ge', models)

# ---------- analysis the peak of the of ge data ----------
