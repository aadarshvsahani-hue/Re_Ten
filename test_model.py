import joblib
import pandas as pd
import numpy as np

# Load models
scaler = joblib.load('models/scaler.joblib')
model = joblib.load('models/model_rf_reg.joblib')

# Load data
df = pd.read_excel('data/ReTen_Final_Organized_Dataset.xlsx')
df['Daily_Calorie_intake'] = df['Daily_Calorie_intake'].fillna(df['Daily_Calorie_intake'].median())
df['Supplements_Taken'] = df['Supplements_Taken'].fillna(df['Supplements_Taken'].mode()[0])
df['performance_score'] = df['performance_score'].fillna(df['performance_score'].median())

categorical_cols = [col for col in df.select_dtypes(include='object').columns.tolist() if col != 'user_id']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

X = df_encoded.drop(columns=['user_id', 'performance_score'])
y = df_encoded['performance_score']

numerical_columns = X.select_dtypes(include=np.number).columns.tolist()
binary_columns = []
for col in numerical_columns:
    unique_values = X[col].unique()
    if len(unique_values) <= 2 and all(val in [0, 1] for val in unique_values):
        binary_columns.append(col)
numerical_cols_to_scale = [col for col in numerical_columns if col not in binary_columns]

X_scaled = X.copy()
X_scaled[numerical_cols_to_scale] = scaler.transform(X[numerical_cols_to_scale])

# Test prediction on training data
pred = model.predict(X_scaled.iloc[:1])
print("Prediction on training data:", pred)
print("Actual:", y.iloc[0])