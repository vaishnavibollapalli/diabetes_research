# Sex-Stratified Neurodegeneration Trajectories Across the Lifespan

An advanced machine learning framework using longitudinal structural neuroimaging data from the OASIS-3 cohort to map sex-specific patterns of brain aging and track cognitive decline risk.

## Workflow Overview
1. **Normative Modeling (Layer 1):** Uses Gaussian Process Regression (GPR) with RBF kernels to map non-linear volumetric trajectories for cognitively normal males and females (`CDRSUM == 0`).
2. **Feature Engineering:** Calculates age-and-sex-adjusted "W-Scores" measuring individual structural deviation away from the healthy trajectory baseline.
3. **Cognitive Risk Mapping (Layer 2):** Trains an XGBoost Regressor using trajectory deviation features to predict functional cognitive outcomes (`CDRSUM`/`MMSE`), paired with SHAP value generation for feature impact explainability.

## Dataset Structure
The framework dynamically joins four primary relational spreadsheets from the OASIS-3 study linked by `OASISID`:
* `OASIS3_Freesurfer_output (1).csv`: Contains morphometric brain tracking and calculated intracranial volumes (`IntraCranialVol`).
* `OASIS3_demographics.csv`: Supplies fixed demographic attributes (`GENDER`, `APOE` status).
* `OASIS3_UDSb4_cdr.csv`: Provides psychometric scoring profiles (`CDRSUM`, `MMSE`) along with physiological `age at visit`.
* `OASIS3_UDSd1_diagnoses.csv`: Secondary verification resource for structural clinical diagnosis labels.

## Installation & Environment Setup
Ensure you have Python 3.9+ installed along with the required analytical packages:
```bash
pip install pandas numpy scikit-learn xgboost shap gpytorch
