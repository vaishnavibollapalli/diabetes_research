import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from xgboost import XGBRegressor
import shap
import warnings
from pandas.errors import PerformanceWarning

# Suppressing the noise
warnings.simplefilter(action='ignore', category=PerformanceWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

print("1/4: Loading data and mapping brain regions...")
df_fs = pd.read_csv("OASIS3_Freesurfer_output (1).csv").copy()
df_demo = pd.read_csv("OASIS3_demographics.csv").copy()
df_cdr = pd.read_csv("OASIS3_UDSb4_cdr.csv").copy()

df_fs['OASISID'] = df_fs['Subject']
df_fs['days_mri'] = df_fs['MR_session'].str.extract(r'_d(\d+)').astype(float)
df_cdr['days_cdr'] = df_cdr['OASIS_session_label'].str.extract(r'_d(\d+)').astype(float)

df_mri_demo = pd.merge(df_fs, df_demo, on='OASISID', how='inner')

# Define the list of regions available in your Freesurfer file
regions = {
    'Hippocampus': ('Left-Hippocampus_volume', 'Right-Hippocampus_volume'),
    'Amygdala': ('Left-Amygdala_volume', 'Right-Amygdala_volume'),
    'Caudate': ('Left-Caudate_volume', 'Right-Caudate_volume'),
    'Putamen': ('Left-Putamen_volume', 'Right-Putamen_volume'),
    'Pallidum': ('Left-Pallidum_volume', 'Right-Pallidum_volume'),
    'Thalamus': ('Left-Thalamus-Proper_volume', 'Right-Thalamus-Proper_volume'),
    'Lateral Ventricle': ('Left-Lateral-Ventricle_volume', 'Right-Lateral-Ventricle_volume')
}

master_rows = []
for idx, mri_row in df_mri_demo.iterrows():
    subj = mri_row['OASISID']
    mri_day = mri_row['days_mri']
    if pd.isna(mri_day): continue
    subj_cdr = df_cdr[df_cdr['OASISID'] == subj]
    if subj_cdr.empty: continue
    
    day_deltas = (subj_cdr['days_cdr'] - mri_day).abs().dropna()
    if day_deltas.empty: continue
    
    closest_cdr_row = df_cdr.loc[day_deltas.idxmin()]
    
    row_data = {
        'Age': closest_cdr_row['age at visit'],
        'Gender': mri_row['GENDER'], 
        'eTIV': mri_row['IntraCranialVol'],
        'CDRSUM': closest_cdr_row['CDRSUM']
    }
    # Extract and combine Left/Right for each region
    for name, (left, right) in regions.items():
        row_data[name] = mri_row[left] + mri_row[right]
        
    master_rows.append(row_data)

df_master = pd.DataFrame(master_rows).dropna()

print("2/4: Running GPR Normative Models across all regions...")
df_healthy = df_master[df_master['CDRSUM'] == 0]
df_m = df_healthy[df_healthy['Gender'] == 1]
df_f = df_healthy[df_healthy['Gender'] == 2]

kernel = RBF(length_scale=10.0) + WhiteKernel(noise_level=1e-4)

# Dictionary to hold computed W-scores for XGBoost
X_features = pd.DataFrame({'Age': df_master['Age'], 'Gender': df_master['Gender']})
gp_models = {}

for name in regions.keys():
    # Normalize by head size (eTIV)
    norm_m = df_m[name] / df_m['eTIV']
    norm_f = df_f[name] / df_f['eTIV']
    norm_all = df_master[name] / df_master['eTIV']
    
    # Fit independent curves for this region
    gp_m = GaussianProcessRegressor(kernel=kernel).fit(df_m[['Age']], norm_m)
    gp_f = GaussianProcessRegressor(kernel=kernel).fit(df_f[['Age']], norm_f)
    gp_models[name] = (gp_m, gp_f)
    
    # Generate W-scores
    w_scores = []
    for idx, row in df_master.iterrows():
        age = np.array([[row['Age']]])
        gp = gp_m if row['Gender'] == 1 else gp_f
        mean, std = gp.predict(age, return_std=True)
        w_scores.append((row[name]/row['eTIV'] - mean[0]) / max(std[0], 1e-6))
        
    X_features[f'{name}_Deviation'] = w_scores

print("3/4: Quantifying predictive significance via XGBoost & SHAP...")
y = df_master['CDRSUM']
xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05).fit(X_features, y)

explainer = shap.TreeExplainer(xgb, feature_perturbation="tree_path_dependent")
shap_values = explainer(X_features)

# Calculate mean absolute SHAP values for the regions
feature_names = X_features.columns
mean_shap = np.abs(shap_values.values).mean(axis=0)
shap_df = pd.DataFrame({'Feature': feature_names, 'Importance': mean_shap})
# Filter out baseline Age/Gender to see just the brain structural rankings
region_shap = shap_df[shap_df['Feature'].str.contains('_Deviation')].copy()
region_shap['Feature'] = region_shap['Feature'].str.replace('_Deviation', '')
region_shap = region_shap.sort_values(by='Importance', ascending=False)

print("4/4: Generating Dashboards & Atlases...")
# Plot 1: Brain Region Significance Ranking
plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=region_shap, palette='viridis')
plt.title("Predictive Significance of Structural Brain Regions for Tracking Cognitive Decline")
plt.xlabel("Mean Absolute SHAP Value (Predictive Weight)")
plt.ylabel("Brain Region")
plt.tight_layout()
plt.savefig("multi_region_significance.png", dpi=300)
plt.close()

# Plot 2: Top 4 Region Trajectory Atlas Grid
top_4_regions = region_shap['Feature'].head(4).tolist()
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
age_range = np.linspace(df_master['Age'].min(), df_master['Age'].max(), 100).reshape(-1, 1)

for i, name in enumerate(top_4_regions):
    gp_m, gp_f = gp_models[name]
    m_mean, m_std = gp_m.predict(age_range, return_std=True)
    f_mean, f_std = gp_f.predict(age_range, return_std=True)
    
    ax = axes[i]
    ax.plot(age_range, m_mean, 'b-', label='Male', linewidth=2)
    ax.fill_between(age_range.flatten(), m_mean - 1.96*m_std, m_mean + 1.96*m_std, color='blue', alpha=0.1)
    
    ax.plot(age_range, f_mean, 'r-', label='Female', linewidth=2)
    ax.fill_between(age_range.flatten(), f_mean - 1.96*f_std, f_mean + 1.96*f_std, color='red', alpha=0.1)
    
    ax.set_title(f"{name} Lifespan Trajectory")
    ax.set_xlabel("Age")
    ax.set_ylabel("Normalized Volume")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

plt.suptitle("Normative Structural Aging Trajectories for Top Significant Regions", fontsize=16)
plt.tight_layout()
plt.savefig("trajectory_atlas.png", dpi=300)
plt.close()

print("\n=== SUCCESS ===")
print("Generated 'multi_region_significance.png' - A visual ranking of region importance.")
print("Generated 'trajectory_atlas.png' - A 4-panel visual mapping of top aging curves.")