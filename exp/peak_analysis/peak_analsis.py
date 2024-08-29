import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = 'tools/TIMES.TTF'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

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

def sampler(*args, num_samples=1000):
    # give np arraies as input and randomly sample num_samples from each array and return the samples
    samples = []
    for array in args:
        indices = np.random.choice(array.shape[0], num_samples, replace=False)
        samples.append(array[indices])
    return samples

def plot_peak_times(results, resolution, country, models):
    all_peak_times = []
    colors = plt.cm.get_cmap('coolwarm', len(models))  # Use a colormap to get distinct colors
    plt.figure(figsize=(10, 3.5))
    _size = 12
    ave_points = []
    for i, (max_values, max_indices) in enumerate(results):
        peak_times = (max_indices * resolution).flatten()
        all_peak_times.extend(peak_times)
        peak_times_in_hours = peak_times / 60.0
        plt.scatter(peak_times_in_hours, max_values.flatten(), alpha=0.2, color=colors(i))
        avg_peak_time_model = np.mean(peak_times)
        avg_peak_value = np.mean(max_values)
        avg_peak_time_hours = avg_peak_time_model // 60
        avg_peak_time_minutes = avg_peak_time_model % 60
        avg_peak_time_in_hours = avg_peak_time_model / 60.0
        plt.scatter(avg_peak_time_in_hours, avg_peak_value, color=colors(i), edgecolor='black', s=150, label=f'Avg {models[i]} ({int(avg_peak_time_hours)}:{int(avg_peak_time_minutes):02d})')
        # print(f'Average peak time for {models[i]}: {avg_peak_time_hours:.2f} hours, {avg_peak_value} minutes')
        plt.xlim(0, 24)
        plt.ylim(0, None)
        plt.xticks(fontsize=_size-4)
        plt.yticks(fontsize=_size-4)
        ave_points.append((avg_peak_time_in_hours, avg_peak_value))
    plt.xlabel('Time of Day (Hours)', fontproperties=font_prop, fontsize=_size)
    plt.ylabel('Peak Value',  fontproperties=font_prop, fontsize=_size, labelpad=5)
    # plt.title(f'Peak Values and Peak Times for {country.upper()} Data', fontproperties=font_prop, fontsize=_size)
    plt.legend(loc='upper right', prop=font_prop, fontsize=_size)
    plt.grid(True)
    plt.savefig(f'exp/peak_analysis/{country}_peak_times.png')
    plt.show()
    return ave_points
    
def compute_dis(args, models):
    original_x, original_y = args[0][0], args[0][1]
    i = 1
    for arg in args[1:]:
        # compute eud distance between original and other models
        eud = np.sqrt((original_x - arg[0])**2 + (original_y - arg[1])**2)
        print(f'Euclidean distance between Original and {models[i]}: {np.mean(eud):.2f}')
        i += 1

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
wgan_data = pd.read_csv(path_wgan, index_col=0).values
data_original, copula_data, fcpflow_data, gmm_data, wgan_data = sampler(data_original, copula_data, fcpflow_data, gmm_data, wgan_data, num_samples=1000)
original_max_vi = get_max_and_index(data_original, copula_data, fcpflow_data, gmm_data, wgan_data)
models = ['Original', 'Copula', 'FCPflow', 'GMM', 'WGAN-gp']
ave_points = plot_peak_times(original_max_vi, 15, 'ge', models)
compute_dis(ave_points,  models=models)
# ---------- analysis the peak of the of ge data ----------


# ---------- analysis the peak of the of nl data ----------
path_original = 'data/nl_data_cleaned_annual_test.csv'
path_copula = 'exp/peak_analysis/data/nl/copula_samples_nl.csv'
path_fcpflow = 'exp/peak_analysis/data/nl/fctflow_samples_nl.csv'
data_original = pd.read_csv(path_original, index_col=0).iloc[:, 2:-2].values
data_original = data_original[~pd.isna(data_original).any(axis=1)]
data_copula = pd.read_csv(path_copula, index_col=0).iloc[:, :-2].values
data_fcpflow = pd.read_csv(path_fcpflow, index_col=0).iloc[:, :-2].values
data_original, data_copula, data_fcpflow = sampler(data_original, data_copula, data_fcpflow, num_samples=1000)
print(data_original.shape, data_copula.shape, data_fcpflow.shape)
original_max_vi = get_max_and_index(data_original, data_copula, data_fcpflow)
models = ['Original', 'Copula', 'FCPflow']
ave_points = plot_peak_times(original_max_vi, 60, 'nl', models)
compute_dis(ave_points, models)
# ---------- analysis the peak of the of nl data ----------

# ---------- analysis the peak of the of uk data ----------
path_original = 'data/uk_data_cleaned_ind_test.csv'
path_copula = 'exp/peak_analysis/data/uk/copula_samples_uk.csv'
path_fcpflow = 'exp/peak_analysis/data/uk/fctflow_samples_uk.csv'
data_original = pd.read_csv(path_original, index_col=0).iloc[:, :-2].values
data_original = data_original[~pd.isna(data_original).any(axis=1)]
data_copula = pd.read_csv(path_copula, index_col=0).iloc[:, :-2].values
data_fcpflow = pd.read_csv(path_fcpflow, index_col=0).iloc[:, :-2].values
data_original, data_copula, data_fcpflow = sampler(data_original, data_copula, data_fcpflow, num_samples=1000)
print(data_original.shape, data_copula.shape, data_fcpflow.shape)
original_max_vi = get_max_and_index(data_original, data_copula, data_fcpflow)
models = ['Original', 'Copula', 'FCPflow']
ave_points = plot_peak_times(original_max_vi, 30, 'uk', models)
compute_dis(ave_points, models)
# ---------- analysis the peak of the of uk data ----------

# ---------- analysis the peak of the of aus data ----------
path_original = 'data/aus_data_cleaned_annual_test.csv'
path_copula = 'exp/peak_analysis/data/aus/copula_samples_aus.csv'
path_fcpflow = 'exp/peak_analysis/data/aus/fctflow_samples_aus.csv'
data_original = pd.read_csv(path_original, index_col=0).iloc[:, 2:-2].values
data_copula = pd.read_csv(path_copula, index_col=0).iloc[:, :-2].values
data_fcpflow = pd.read_csv(path_fcpflow, index_col=0).iloc[:, :-2].values
data_original, data_copula, data_fcpflow = sampler(data_original, data_copula, data_fcpflow, num_samples=1000)
print(data_original.shape, data_copula.shape, data_fcpflow.shape)
original_max_vi = get_max_and_index(data_original, data_copula, data_fcpflow)
models = ['Original', 'Copula', 'FCPflow']
ave_points = plot_peak_times(original_max_vi, 30, 'aus', models)
compute_dis(ave_points, models)
# ---------- analysis the peak of the of aus data ----------

# ---------- analysis the peak of the of usa data ----------
path_original = 'data/usa_data_cleaned_annual_test.csv'
path_copula = 'exp/peak_analysis/data/usa/copula_samples_usa.csv'
path_fcpflow = 'exp/peak_analysis/data/usa/fctflow_samples_usa.csv'
data_original = pd.read_csv(path_original, index_col=0).iloc[:, :-2].values
data_copula = pd.read_csv(path_copula, index_col=0).iloc[:, :-2].values
data_fcpflow = pd.read_csv(path_fcpflow, index_col=0).iloc[:, :-2].values
data_original = data_original[~pd.isna(data_original).any(axis=1)]
data_original, data_copula, data_fcpflow = sampler(data_original, data_copula, data_fcpflow, num_samples=1000)
print(data_original.shape, data_copula.shape, data_fcpflow.shape)
original_max_vi = get_max_and_index(data_original, data_copula, data_fcpflow)
models = ['Original', 'Copula', 'FCPflow']
ave_points = plot_peak_times(original_max_vi, 15, 'usa', models)
compute_dis(ave_points, models)
# ---------- analysis the peak of the of usa data ----------

# ---------- analysis the peak of the of uk weather data ----------
path_original = 'data/uk_data_cleaned_ind_weather_test.csv'
path_copula = 'exp/peak_analysis/data/uk_weather/copula_samples_uk_weather.csv'
path_fcpflow = 'exp/peak_analysis/data/uk_weather/fctflow_samples_uk_weather.csv'
data_original = pd.read_csv(path_original, index_col=0).iloc[:, 4:4+48].values
data_copula = pd.read_csv(path_copula, index_col=0).iloc[:, :48].values
data_fcpflow = pd.read_csv(path_fcpflow, index_col=0).iloc[:, :48].values
data_original = data_original[~pd.isna(data_original).any(axis=1)]
data_original, data_copula, data_fcpflow = sampler(data_original, data_copula, data_fcpflow, num_samples=1000)
print(data_original.shape, data_copula.shape, data_fcpflow.shape)
original_max_vi = get_max_and_index(data_original, data_copula, data_fcpflow)
models = ['Original', 'Copula', 'FCPflow']
ave_points = plot_peak_times(original_max_vi, 30, 'uk_weather', models)
compute_dis(ave_points, models)
# ---------- analysis the peak of the of uk weather data ----------