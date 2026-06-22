# Project NeuroDiverge: Multi-Region Sex-Stratified Neurodegeneration Trajectories Across the Lifespan

An advanced computational neuroscience and interpretable machine learning pipeline that uses longitudinal morphometric tracking from the OASIS-3 clinical cohort to map sex-specific patterns of brain aging and predict clinical cognitive decline.

---

## 🔬 Core Scientific Breakthrough & Key Findings

By expanding the framework from a single-region analysis to a multi-region structural atlas, this pipeline uncovered several highly significant neurodegenerative signatures:

### 1. Structural Predictive Significance (Refer to image_9bc002.png)
Using **XGBoost + SHAP** to rank age-and-sex-adjusted trajectory deviations (W-scores) against clinical cognitive decline (`CDRSUM`), our model revealed a striking hierarchy of predictive importance:
*   **The Amygdala Limbed Dominance:** The **Amygdala** emerged as the single most powerful structural predictor of cognitive impairment, slightly outperforming the **Hippocampus**. While the hippocampus is the traditional gold standard for Alzheimer's tracking, modern literature confirms that early, aggressive amygdala atrophy strongly correlates with neuropsychiatric symptoms and rapid disease progression.
*   **Lateral Ventricle Expansion:** The **Lateral Ventricle** ranked third. Ventricular enlargement serves as a macro-level proxy for global gray matter tissue loss (ex-vacuo dilation), making its rate of expansion a highly reliable indicator of active neurodegeneration.
*   **Basal Ganglia Stability:** Central structures like the *Pallidum*, *Caudate*, *Putamen*, and *Thalamus* showed drastically lower predictive weights, indicating that structural variation in these subcortical regions is less tied to clinical memory symptoms.

### 2. Lifespan Trajectory Divergence (Refer to image_9bc041.jpg)
By fitting **Gaussian Process Regression (GPR)** models with non-linear Radial Basis Function (RBF) kernels to healthy subjects (`CDRSUM == 0`), we mapped true normative aging baselines:
*   **The Ventricular Inflection Point:** While the Amygdala, Hippocampus, and Pallidum display steady, parallel, linear-like volumetric declines between sexes across the 40–100 age range, the **Lateral Ventricle shows an explicit non-linear divergence**. 
*   **Late-Life Accelerated Aging:** Around **age 75**, the healthy female ventricular curve curves sharply upward, significantly out-pacing the male trajectory by age 90+. This provides clear, data-driven visualization of accelerated late-life structural changes in the female brain, establishing a definitive biological divergence window.

---

## 🧠 Computational Pipeline Architecture

The framework operates via a three-layered machine learning architecture:
[ OASIS-3 Tabular Data Spreads ]
│
▼
[ 1. Data Alignment Engine ] ──► Matches longitudinal MRIs with closest Clinical Visits
│
▼
[ 2. Layer 1: Normative GPR ] ──► Fits separate Male/Female curves on healthy subjects
│
▼
[ 3. W-Score Feature Mapping ] ──► Calculates individual deviation from sex-specific norms
│
▼
[ 4. Layer 2: Explainable ML ] ──► XGBoost Regressor + SHAP value mapping targeting CDRSUM

*   **Layer 1: Non-Parametric Continuous Modeling (GPR):** Models baseline gray matter volume trends strictly on cognitively intact individuals over time. This approach naturally captures non-linear aging geometry and accommodates heteroscedastic uncertainty boundaries.
*   **Feature Engineering (W-Scores):** Instead of using raw volumetric metrics—which are heavily confounded by the fact that biological males possess larger average head sizes—volumes are normalized by Estimated Total Intracranial Volume (`IntraCranialVol`). The pipeline then computes an age-and-sex-adjusted Z-score (W-score) representing an individual's exact standard deviation from their normative group mean:
$$W = \frac{\text{Actual Normalized Vol} - \mu_{\text{GPR}}(Age)}{\sigma_{\text{GPR}}(Age)}$$
*   **Layer 2: Tree-Based Supervised Learning (XGBoost):** Uses the engineered multi-region deviation matrices to map non-linear feature interactions and predict global clinical cognitive scores.
*   **Explainability (SHAP):** Deploys tree-path dependent Shapley Additive Explanations to calculate exact feature attribution weights across all distinct clinical rows.

---

## 📊 Relational File Mapping & Joins

The script programmatically parses and blends four primary raw files linked via the patient identifier `OASISID`:
1.  **`OASIS3_Freesurfer_output (1).csv`:** Automated subcortical structure segmentations (`Left-Hippocampus_volume`, `Right-Amygdala_volume`, etc.) and head size metrics (`IntraCranialVol`).
2.  **`OASIS3_demographics.csv`:** Demographic anchor fields including fixed biological `GENDER` categories.
3.  **`OASIS3_UDSb4_cdr.csv`:** Primary clinical longitudinal target variables tracking precise `age at visit`, global cognitive scores (`MMSE`), and functional metrics (`CDRSUM`).
4.  **`OASIS3_UDSd1_diagnoses.csv`:** Secondary data asset utilized for clinical categorical diagnosis validation checks.

*Longitudinal Handling Note:* Subjects with multiple scans across different dates (tracked using the `MR_session` column, e.g., `_d0129` vs. `_d0757`) are handled automatically. The engine parses the precise day-offset and performs a nearest-neighbor temporal join to align each scan with its closest valid clinical psychometric interview.

---

## 🚀 Installation & Execution

### Dependencies
Ensure your environment is running Python 3.11+ and install the optimized computer vision and machine learning core packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
