import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from xgboost import XGBRegressor
import shap
import warnings
from pandas.errors import PerformanceWarning

# suppressing all warning noise
warnings.simplefilter(action='ignore', category=PerformanceWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

print("1/5: Loading and aligning longitudinal spreadsheets...")
df_fs = pd.read_csv("OASIS3_Freesurfer_output (1).csv").copy()
df_demo = pd.read_csv("OASIS3_demographics.csv").copy()
df_cdr = pd.read_csv("OASIS3_UDSb4_cdr.csv").copy()

df_fs['OASISID'] = df_fs['Subject']
df_fs['days_mri'] = df_fs['MR_session'].str.extract(r'_d(\d+)').astype(float)
df_cdr['days_cdr'] = df_cdr['OASIS_session_label'].str.extract(r'_d(\d+)').astype(float)
df_cdr = df_cdr.dropna(subset=['days_cdr'])

df_mri_demo = pd.merge(df_fs, df_demo, on='OASISID', how='inner')

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
    master_rows.append({
        'Age': closest_cdr_row['age at visit'],
        'Gender': mri_row['GENDER'], 
        'eTIV': mri_row['IntraCranialVol'],
        'L_Hipp': mri_row['Left-Hippocampus_volume'],
        'R_Hipp': mri_row['Right-Hippocampus_volume'],
        'CDRSUM': closest_cdr_row['CDRSUM']
    })

df_master = pd.DataFrame(master_rows).dropna()
df_master['Norm_Hippocampus'] = (df_master['L_Hipp'] + df_master['R_Hipp']) / df_master['eTIV']

print("2/5: Fitting Gaussian Process Regressions (Layer 1)...")
df_healthy = df_master[df_master['CDRSUM'] == 0]
df_m = df_healthy[df_healthy['Gender'] == 1]
df_f = df_healthy[df_healthy['Gender'] == 2]

kernel = RBF(length_scale=10.0) + WhiteKernel(noise_level=1e-4)
gp_male = GaussianProcessRegressor(kernel=kernel).fit(df_m[['Age']], df_m['Norm_Hippocampus'])
gp_female = GaussianProcessRegressor(kernel=kernel).fit(df_f[['Age']], df_f['Norm_Hippocampus'])

# Calculate W-scores
deviations = []
for idx, row in df_master.iterrows():
    age = np.array([[row['Age']]])
    mean, std = gp_male.predict(age, return_std=True) if row['Gender'] == 1 else gp_female.predict(age, return_std=True)
    deviations.append((row['Norm_Hippocampus'] - mean[0]) / max(std[0], 1e-6))
df_master['Hippocampus_Deviation_W_Score'] = deviations

print("3/5: Training Risk Predictor & Explainability Engine (Layer 2)...")
X = df_master[['Age', 'Gender', 'Hippocampus_Deviation_W_Score']]
y = df_master['CDRSUM']
xgb_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05).fit(X, y)

explainer = shap.TreeExplainer(xgb_model, feature_perturbation="tree_path_dependent")
shap_values = explainer(X)

print("4/5: Generating Plot 1: Trajectory Divergence Atlas...")
plt.figure(figsize=(10, 6))
age_range = np.linspace(df_master['Age'].min(), df_master['Age'].max(), 100).reshape(-1, 1)

m_mean, m_std = gp_male.predict(age_range, return_std=True)
f_mean, f_std = gp_female.predict(age_range, return_std=True)

plt.plot(age_range, m_mean, 'b-', label='Healthy Male Trajectory', linewidth=2)
plt.fill_between(age_range.flatten(), m_mean - 1.96*m_std, m_mean + 1.96*m_std, color='blue', alpha=0.15)

plt.plot(age_range, f_mean, 'r-', label='Healthy Female Trajectory', linewidth=2)
plt.fill_between(age_range.flatten(), f_mean - 1.96*f_std, f_mean + 1.96*f_std, color='red', alpha=0.15)

plt.title("Sex-Stratified Hippocampal Volumetric Trajectories Across the Lifespan")
plt.xlabel("Chronological Age")
plt.ylabel("Normalized Hippocampal Volume")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("trajectory_divergence.png", dpi=300)
plt.close()

print("5/5: Generating Plot 2: SHAP Feature Importance...")
plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title("Machine Learning Feature Importance Weights (Predicting Cognitive Decline)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.close()

print("\n=== SUCCESS ===")
print("Both diagnostic plots have been generated and saved to your directory!")
print("1. 'trajectory_divergence.png' -> Visualizes where male/female brain trends separate.")
print("2. 'feature_importance.png' -> Visualizes the predictive weight of your metrics.")