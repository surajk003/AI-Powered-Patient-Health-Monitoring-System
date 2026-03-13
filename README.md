# Patient Health Monitoring - Train Random Forest Model

This repository contains a script to train a Random Forest classifier to predict the `Predicted Disease` column from the provided synthetic patient monitoring CSV.

Files:
- `train_patient_monitoring_model.py`: Main script. Loads data, preprocesses, trains, evaluates, plots, and saves the model as `patient_monitoring_rf_model.pkl` by default.
- `requirements.txt`: Python dependencies.

Usage
1. Create and activate a Python environment in VS Code.
2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Run training (default reads `Synthetic_patient-HealthCare-Monitoring_dataset.csv` in this folder):

```powershell
python train_patient_monitoring_model.py --data-path "./Synthetic_patient-HealthCare-Monitoring_dataset.csv"
```

Outputs
- `patient_monitoring_rf_model.pkl`: saved pipeline and label encoder
- `confusion_matrix.png` and `feature_importances.png`: evaluation plots

Notes
- The script auto-detects numeric and categorical columns and uses appropriate imputation and encoding.
- If your dataset CSV is elsewhere, pass its path via `--data-path`.
