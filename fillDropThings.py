import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
# Load your dataset
data = pd.read_csv(r'H:\data\data.csv',dtype={"tg":float, "ldl": float},encoding='ISO-8859-1',low_memory=False)
# Identify features with missing values
features_with_missing_values = ['plt','wbc','rbc','hba1c','crp','tg','ldl','hdl','ast','alt','bilirubin','albumin','urea','creatinine','bua','pt','aptt','tt','inr','d_dimer','fibrinogen','ck','ck_mb','ldh','hbdh','ima','na','k','cl','ca','p','lactate','anion_gap','tco2','nihss',
]  

# Split the data into two parts: rows with and without missing values
data_with_missing = data[data[features_with_missing_values].isnull().any(axis=1)]
data_without_missing = data.dropna(subset=features_with_missing_values)

# Create training and test sets from the data without missing values
X_train = data_without_missing.drop(features_with_missing_values, axis=1)
y_train = data_without_missing[features_with_missing_values]

# Train a random forest model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Predict missing values
X_missing = data_with_missing.drop(features_with_missing_values, axis=1)
y_pred = rf_model.predict(X_missing)
y_pred =np.around(y_pred, decimals=2)
# Fill missing values in the original dataset
data.loc[data_with_missing.index, features_with_missing_values] = y_pred

# 存储为CSV文件
data.to_csv(r'H:\data\filled.csv', index=False)