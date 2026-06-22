import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import urllib.request
import io
import warnings
from pandas.errors import PerformanceWarning

# Suppress warnings
warnings.simplefilter(action='ignore', category=PerformanceWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

TARGET_REGION = 'Lateral Ventricle' #chnage the region here to get animate for a different brain region
regions_map = {
    'Lateral Ventricle': ('Left-Lateral-Ventricle_volume', 'Right-Lateral-Ventricle_volume')
}

print("1/4: Fetching neuroanatomy reference graphic...")
# Fetching an open-source coronal reference image highlighting ventricles
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Brain_chrischan_coronal.jpg/320px-Brain_chrischan_coronal.jpg"
try:
    with urllib.request.urlopen(url) as response:
        img_data = response.read()
    anatomy_img = plt.imread(io.BytesIO(img_data), format='jpg')
except Exception:
    # Fallback to dummy placeholder matrix if connection fails
    anatomy_img = np.zeros((100, 100, 3))

print("2/4: Loading data and aligning longitudinal profiles...")
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
    if closest_cdr_row['CDRSUM'] == 0: 
        master_rows.append({
            'Age': closest_cdr_row['age at visit'],
            'Gender': mri_row['GENDER'], 
            'Volume': mri_row[left_col] + mri_row[right_col],
            'eTIV': mri_row['IntraCranialVol']
        })

df_healthy = pd.DataFrame(master_rows).dropna()
df_healthy['Norm_Volume'] = df_healthy['Volume'] / df_healthy['eTIV']

df_m = df_healthy[df_healthy['Gender'] == 1]
df_f = df_healthy[df_healthy['Gender'] == 2]

kernel = RBF(length_scale=10.0) + WhiteKernel(noise_level=1e-4)
gp_male = GaussianProcessRegressor(kernel=kernel).fit(df_m[['Age']], df_m['Norm_Volume'])
gp_female = GaussianProcessRegressor(kernel=kernel).fit(df_f[['Age']], df_f['Norm_Volume'])


# SIDE-BY-SIDE ANIMATION SETUP

print("3/4: Rendering dual-panel animation canvas...")
fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.3]})

min_age, max_age = int(df_healthy['Age'].min()), int(df_healthy['Age'].max())
age_axis = np.linspace(min_age, max_age, 200).reshape(-1, 1)

m_mean, m_std = gp_male.predict(age_axis, return_std=True)
f_mean, f_std = gp_female.predict(age_axis, return_std=True)

# Panel 1: Structural Anatomy Reference Focus
ax_img.imshow(anatomy_img)
ax_img.set_title("Anatomical Cross-Section\n(Target: Central Lateral Ventricles)", fontsize=11, fontweight='bold')
ax_img.axis('off')

# Panel 2: Live Trajectory Monitor
ax_plot.fill_between(age_axis.flatten(), m_mean - 1.96*m_std, m_mean + 1.96*m_std, color='blue', alpha=0.08, label='Male 95% CI')
ax_plot.fill_between(age_axis.flatten(), f_mean - 1.96*f_std, f_mean + 1.96*f_std, color='red', alpha=0.08, label='Female 95% CI')
ax_plot.plot(age_axis, m_mean, color='blue', linestyle='--', alpha=0.3)
ax_plot.plot(age_axis, f_mean, color='red', linestyle='--', alpha=0.3)

male_dot, = ax_plot.plot([], [], 'bo', markersize=10, label='Male Baseline')
female_dot, = ax_plot.plot([], [], 'ro', markersize=10, label='Female Baseline')
timeline_bar = ax_plot.axvline(x=min_age, color='purple', linestyle=':', alpha=0.7)

tracking_text = ax_plot.text(0.02, 0.88, '', transform=ax_plot.transAxes, fontsize=10, 
                             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

ax_plot.set_xlim(min_age - 2, max_age + 2)
ax_plot.set_ylim(df_healthy['Norm_Volume'].min() * 0.9, df_healthy['Norm_Volume'].max() * 1.1)
ax_plot.set_xlabel("Chronological Lifespan Age", fontsize=11)
ax_plot.set_ylabel(f"Normalized Volumetric Scale", fontsize=11)
ax_plot.legend(loc='lower left')
ax_plot.grid(True, linestyle=':', alpha=0.6)

plt.suptitle(f"Longitudinal Trajectory Mapping & Divergence Atlas: {TARGET_REGION}", fontsize=14, fontweight='bold')
plt.tight_layout()

def update_frame(frame_idx):
    current_age = min_age + frame_idx
    age_query = np.array([[current_age]])
    m_val = gp_male.predict(age_query)[0]
    f_val = gp_female.predict(age_query)[0]
    
    male_dot.set_data([current_age], [m_val])
    female_dot.set_data([current_age], [f_val])
    timeline_bar.set_xdata([current_age])
    
    delta_percentage = abs(m_val - f_val) / ((m_val + f_val) / 2) * 100
    tracking_text.set_text(
        f"Timeline Track: Age {current_age}\n"
        f"Male Atrophy Baseline: {m_val:.5f}\n"
        f"Female Atrophy Baseline: {f_val:.5f}\n"
        f"Divergence Magnitude: {delta_percentage:.2f}%"
    )
    return male_dot, female_dot, timeline_bar, tracking_text

total_frames = max_age - min_age
ani = animation.FuncAnimation(fig, update_frame, frames=total_frames, interval=150, blit=True)

output_filename = f"{TARGET_REGION.lower().replace(' ', '_')}_with_anatomy.gif"
print("4/4: Compiling synchronized frame profiles...")
ani.save(output_filename, writer='pillow')
print("Opening interactive player window...")
plt.show() 

print(f"\n=== SUCCESS ===\nSaved unified presentation dashboard as: {output_filename}")
