import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier

# Load the dataset
file_path = 'data/ReTen_Final_Organized_Dataset.xlsx'
df = pd.read_excel(file_path)

# Fill missing values as in the app
df['Daily_Calorie_intake'] = df['Daily_Calorie_intake'].fillna(df['Daily_Calorie_intake'].median())
df['Supplements_Taken'] = df['Supplements_Taken'].fillna(df['Supplements_Taken'].mode()[0])
df['performance_score'] = df['performance_score'].fillna(df['performance_score'].median())

# Encode categorical variables
categorical_cols = [col for col in df.select_dtypes(include='object').columns.tolist() if col != 'user_id']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

# Prepare features for regression (performance_score prediction)
X_reg = df_encoded.drop(columns=['user_id', 'performance_score'])
y_reg = df_encoded['performance_score']

# Identify numerical columns for scaling
numerical_columns = X_reg.select_dtypes(include=np.number).columns.tolist()
binary_columns = []
for col in numerical_columns:
    unique_values = X_reg[col].unique()
    if len(unique_values) <= 2 and all(val in [0, 1] for val in unique_values):
        binary_columns.append(col)
numerical_cols_to_scale = [col for col in numerical_columns if col not in binary_columns]

# Fit scaler on numerical features
scaler = StandardScaler()
scaler.fit(X_reg[numerical_cols_to_scale])

# Scale the features
X_reg_scaled = X_reg.copy()
X_reg_scaled[numerical_cols_to_scale] = scaler.transform(X_reg[numerical_cols_to_scale])

# Train Random Forest Regressor
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_reg_scaled, y_reg)

# For classification, we need a target. Assuming we create a binary target based on performance_score
# For example, high performance vs low
mean_performance = y_reg.mean()
y_clf = (y_reg > mean_performance).astype(int)

# Train XGBoost Classifier
xgb_clf = XGBClassifier(random_state=42)
xgb_clf.fit(X_reg_scaled, y_clf)

# Train Logistic Regression for probabilities
lr_clf = LogisticRegression(random_state=42)
lr_clf.fit(X_reg_scaled, y_clf)

# Save models and scaler
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(rf_reg, 'models/model_rf_reg.joblib')
joblib.dump(xgb_clf, 'models/model_xgb_clf.joblib')
joblib.dump(lr_clf, 'models/model_clf.joblib')

print("Models retrained and saved successfully!")
print(f"Scaler fitted on {len(numerical_cols_to_scale)} features")
print(f"Regression target: performance_score (mean: {y_reg.mean():.2f})")
print(f"Classification target: performance_score > {mean_performance:.2f} (class distribution: {y_clf.value_counts().to_dict()})")