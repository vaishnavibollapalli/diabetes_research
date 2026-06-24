import pandas as pd

# Load your aligned master data or raw Freesurfer file
print("Scanning Freesurfer dataset for extreme structural contrasts...")
df = pd.read_csv("OASIS3_Freesurfer_output (1).csv")

# Combine left and right sides and normalize by head size (eTIV)
df['Ventricles_Norm'] = (df['Left-Lateral-Ventricle_volume'] + df['Right-Lateral-Ventricle_volume']) / df['IntraCranialVol']
df['Amygdala_Norm'] = (df['Left-Amygdala_volume'] + df['Right-Amygdala_volume']) / df['IntraCranialVol']

# 1: Small Ventricles & Large Amygdala (Highly Robust / Healthy Profile) 
# Sort by Ventricles ascending, Amygdala descending
profile_1 = df.sort_values(by=['Ventricles_Norm', 'Amygdala_Norm'], ascending=[True, False]).head(3)

#  2: Large Ventricles & Small Amygdala (High Neurodegeneration Profile) 
# Sort by Ventricles descending, Amygdala ascending
profile_2 = df.sort_values(by=['Ventricles_Norm', 'Amygdala_Norm'], ascending=[False, True]).head(3)

print("\n==================================================")
print("PROFILE 1: Small Ventricles + Large Amygdala (Healthy Baseline)")
print("==================================================")
for idx, row in profile_1.iterrows():
    print(f"Subject ID: {row['Subject']} | Session: {row['MR_session']} | Ventricles (Norm): {row['Ventricles_Norm']:.5f} | Amygdala (Norm): {row['Amygdala_Norm']:.5f}")

print("\n==================================================")
print("PROFILE 2: Large Ventricles + Small Amygdala (Advanced Atrophy)")
print("==================================================")
for idx, row in profile_2.iterrows():
    print(f"Subject ID: {row['Subject']} | Session: {row['MR_session']} | Ventricles (Norm): {row['Ventricles_Norm']:.5f} | Amygdala (Norm): {row['Amygdala_Norm']:.5f}")
print("==================================================")