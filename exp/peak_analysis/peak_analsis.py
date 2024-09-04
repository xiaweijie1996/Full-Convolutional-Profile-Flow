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
        np.random.seed(0)
        indices = np.random.choice(array.shape[0], num_samples, replace=False)
        samples.append(array[indices])
    return samples

def plot_peak_times(results, resolution, country, models):
    all_peak_times = []
    
    # Change the colormap to a blue-tinted one
    colors = plt.cm.get_cmap('PuBu', len(models))  # Use 'coolwarm' for a balanced color map
    
    # Adjust figure size for a long and narrow layout
    plt.figure(figsize=(16, 3))  # Even longer and narrower
    _size = 20  # Set a reasonable size for the ticks
    ave_points = []
    
    for i, (max_values, max_indices) in enumerate(results):
        peak_times = (max_indices * resolution).flatten()
        all_peak_times.extend(peak_times)
        peak_times_in_hours = peak_times / 60.0
        
        # Plot individual points for each model with transparency
        plt.scatter(peak_times_in_hours, max_values.flatten(), alpha=0.1, color=colors(i), s=40)  # Slightly larger points
        
        # Calculate and plot the average peak time and value for each model
        avg_peak_time_model = np.mean(peak_times)
        avg_peak_value = np.mean(max_values)
        avg_peak_time_hours = avg_peak_time_model // 60
        avg_peak_time_minutes = avg_peak_time_model % 60
        avg_peak_time_in_hours = avg_peak_time_model / 60.0
        
        # Plot the average peak time with a black edge for better visibility
        plt.scatter(avg_peak_time_in_hours, avg_peak_value, color=colors(i), edgecolor='black', s=150, 
                    label=f'Center {models[i]} ({int(avg_peak_time_hours)}:{int(avg_peak_time_minutes):02d})')
        
        plt.xlim(0, 24)  # Time of day in hours (0 to 24)
        plt.ylim(0, None)  # Automatically scale the y-axis
        
        # Append points for further analysis
        ave_points.append((avg_peak_time_in_hours, avg_peak_value))
    
    # Set labels with custom font properties
    plt.xlabel('Time of Day [Hours]', fontsize=_size, labelpad=10, fontproperties=font_prop)
    plt.ylabel('Peak Value [kWh]', fontsize=_size, labelpad=5, fontproperties=font_prop)
    
    # Move legend to the top of the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(models), fontsize=_size, prop=font_prop)
    
    # Add grid for better visual guidance
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Manually set the font size and properties for x and y ticks
    plt.gca().tick_params(axis='x', labelsize=_size)  # Explicitly set x-tick size
    plt.gca().tick_params(axis='y', labelsize=_size)  # Explicitly set y-tick size

    # Apply the custom font properties to the x and y tick labels
    for label in plt.gca().get_xticklabels():
        label.set_fontproperties(font_prop)  # Apply font properties to x-ticks
        label.set_fontsize(_size-6)  # Ensure the size is applied

    for label in plt.gca().get_yticklabels():
        label.set_fontproperties(font_prop)  # Apply font properties to y-ticks
        label.set_fontsize(_size-6)  # Ensure the size is applied
        
    # also apply this to legend's font
    for text in plt.gca().get_legend().get_texts():
        text.set_fontproperties(font_prop)
        text.set_fontsize(_size-6)

    # Save the figure with the country name in the file name
    plt.savefig(f'exp/peak_analysis/{country}_peak_times.png', bbox_inches='tight', dpi=300)
    
    # Show the figure
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
models = ['Original', 'Copula', 'FCPFlow', 'GMM', 'WGAN-GP']
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
models = ['Original', 'Copula', 'FCPFlow']
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
models = ['Original', 'Copula', 'FCPFlow']
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
models = ['Original', 'Copula', 'FCPFlow']
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
models = ['Original', 'Copula', 'FCPFlow']
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
models = ['Original', 'Copula', 'FCPFlow']
ave_points = plot_peak_times(original_max_vi, 30, 'uk_weather', models)
compute_dis(ave_points, models)
# ---------- analysis the peak of the of uk weather data ----------

