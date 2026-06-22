import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import warnings
from pandas.errors import PerformanceWarning

# Suppress processing warnings
warnings.simplefilter(action='ignore', category=PerformanceWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# CHOOSE REGION TO ANIMATE

# Options: 'Lateral Ventricle', 'Amygdala', 'Hippocampus'
TARGET_REGION = 'Lateral Ventricle'  #chnge the region here to get animate for a different brain region

regions_map = {
    'Hippocampus': ('Left-Hippocampus_volume', 'Right-Hippocampus_volume'),
    'Amygdala': ('Left-Amygdala_volume', 'Right-Amygdala_volume'),
    'Lateral Ventricle': ('Left-Lateral-Ventricle_volume', 'Right-Lateral-Ventricle_volume')
}

print(# Establish separate data channels via standard pipelines
      f"Configuring dynamic data streams for: {TARGET_REGION}...")


# DATA PIPELINE & MODEL ALIGNMENT

df_fs = pd.read_csv("OASIS3_Freesurfer_output (1).csv").copy()
df_demo = pd.read_csv("OASIS3_demographics.csv").copy()
df_cdr = pd.read_csv("OASIS3_UDSb4_cdr.csv").copy()

df_fs['OASISID'] = df_fs['Subject']
df_fs['days_mri'] = df_fs['MR_session'].str.extract(r'_d(\d+)').astype(float)
df_cdr['days_cdr'] = df_cdr['OASIS_session_label'].str.extract(r'_d(\d+)').astype(float)

df_mri_demo = pd.merge(df_fs, df_demo, on='OASISID', how='inner')

master_rows = []
left_col, right_col = regions_map[TARGET_REGION]

for idx, mri_row in df_mri_demo.iterrows():
    subj = mri_row['OASISID']
    mri_day = mri_row['days_mri']
    if pd.isna(mri_day): continue
    subj_cdr = df_cdr[df_cdr['OASISID'] == subj]
    if subj_cdr.empty: continue
    
    day_deltas = (subj_cdr['days_cdr'] - mri_day).abs().dropna()
    if day_deltas.empty: continue
    
    closest_cdr_row = df_cdr.loc[day_deltas.idxmin()]
    if closest_cdr_row['CDRSUM'] == 0:  # Isolating healthy normative baseline
        master_rows.append({
            'Age': closest_cdr_row['age at visit'],
            'Gender': mri_row['GENDER'], 
            'Volume': mri_row[left_col] + mri_row[right_col],
            'eTIV': mri_row['IntraCranialVol']
        })

df_healthy = pd.DataFrame(master_rows).dropna()
df_healthy['Norm_Volume'] = df_healthy['Volume'] / df_healthy['eTIV']

# Train Gaussian Process Regressors
df_m = df_healthy[df_healthy['Gender'] == 1]
df_f = df_healthy[df_healthy['Gender'] == 2]

kernel = RBF(length_scale=10.0) + WhiteKernel(noise_level=1e-4)
gp_male = GaussianProcessRegressor(kernel=kernel).fit(df_m[['Age']], df_m['Norm_Volume'])
gp_female = GaussianProcessRegressor(kernel=kernel).fit(df_f[['Age']], df_f['Norm_Volume'])


# ANIMATION RENDERING ENGINE

print("Initializing animation canvas...")
fig, ax = plt.subplots(figsize=(10, 6))

min_age, max_age = int(df_healthy['Age'].min()), int(df_healthy['Age'].max())
age_axis = np.linspace(min_age, max_age, 200).reshape(-1, 1)

# Generate background baseline profiles
m_mean, m_std = gp_male.predict(age_axis, return_std=True)
f_mean, f_std = gp_female.predict(age_axis, return_std=True)

# Static background elements (Confidence Interval Shading)
ax.fill_between(age_axis.flatten(), m_mean - 1.96*m_std, m_mean + 1.96*m_std, color='blue', alpha=0.08, label='Male 95% CI')
ax.fill_between(age_axis.flatten(), f_mean - 1.96*f_std, f_mean + 1.96*f_std, color='red', alpha=0.08, label='Female 95% CI')
ax.plot(age_axis, m_mean, color='blue', linestyle='--', alpha=0.3)
ax.plot(age_axis, f_mean, color='red', linestyle='--', alpha=0.3)

# Animated dynamic elements
male_dot, = ax.plot([], [], 'bo', markersize=10, label='Male Current Age Profile')
female_dot, = ax.plot([], [], 'ro', markersize=10, label='Female Current Age Profile')
timeline_bar = ax.axvline(x=min_age, color='purple', linestyle=':', alpha=0.7)

# Text configurations
tracking_text = ax.text(0.02, 0.92, '', transform=ax.transAxes, fontsize=11, 
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

ax.set_xlim(min_age - 2, max_age + 2)
ax.set_ylim(df_healthy['Norm_Volume'].min() * 0.9, df_healthy['Norm_Volume'].max() * 1.1)
ax.set_xlabel("Chronological Lifespan Age", fontsize=12)
ax.set_ylabel(f"Normalized {TARGET_REGION} Volume", fontsize=12)
ax.set_title(f"Dynamic Structural Divergence Tracking: {TARGET_REGION}", fontsize=14, fontweight='bold')
ax.legend(loc='lower left')
ax.grid(True, linestyle=':', alpha=0.6)

# Frame generation update loop
def update_frame(frame_idx):
    current_age = min_age + frame_idx
    age_query = np.array([[current_age]])
    
    # Extract prediction milestones
    m_val = gp_male.predict(age_query)[0]
    f_val = gp_female.predict(age_query)[0]
    
    # Shift positional graphics
    male_dot.set_data([current_age], [m_val])
    female_dot.set_data([current_age], [f_val])
    timeline_bar.set_xdata([current_age])
    
    # Compute instantaneous variance metrics
    delta_percentage = abs(m_val - f_val) / ((m_val + f_val) / 2) * 100
    
    tracking_text.set_text(
        f"Timeline Step: Age {current_age}\n"
        f"Male Norm: {m_val:.5f}\n"
        f"Female Norm: {f_val:.5f}\n"
        f"Trajectory Variance: {delta_percentage:.2f}%"
    )
    return male_dot, female_dot, timeline_bar, tracking_text

# Total frame frames equal to total aging lifespan delta spans
total_frames = max_age - min_age

ani = animation.FuncAnimation(fig, update_frame, frames=total_frames, interval=150, blit=True)

# Save the animation asset
output_filename = f"{TARGET_REGION.lower().replace(' ', '_')}_divergence.gif"
print(f"Compiling tracking frames into video format. Saving as {output_filename}...")
ani.save(output_filename, writer='pillow')
plt.close()

print("\n=== SUCCESS ===")
print(f"Animation loop completed successfully! Open '{output_filename}' to watch the live sex-stratified structural divergence.")