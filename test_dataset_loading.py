import pandas as pd

file_path_preprocessed = 'data/ReTen_Final_Organized_Dataset.xlsx'
df_preprocessed = pd.read_excel(file_path_preprocessed)

df_preprocessed['Daily_Calorie_intake'] = df_preprocessed['Daily_Calorie_intake'].fillna(df_preprocessed['Daily_Calorie_intake'].median())
df_preprocessed['Supplements_Taken'] = df_preprocessed['Supplements_Taken'].fillna(df_preprocessed['Supplements_Taken'].mode()[0])

categorical_cols_to_encode = [col for col in df_preprocessed.select_dtypes(include='object').columns.tolist() if col != 'user_id']
print("Categorical cols to encode:", categorical_cols_to_encode)

df_encoded_temp = pd.get_dummies(df_preprocessed, columns=categorical_cols_to_encode, drop_first=True, dtype=int)

print("Columns in df_encoded_temp:", len(df_encoded_temp.columns))
print("Has performance_score:", 'performance_score' in df_encoded_temp.columns)

y_reg_dummy = df_encoded_temp['performance_score']
print("y_reg_dummy shape:", y_reg_dummy.shape)

X_reg_dummy = df_encoded_temp.drop(columns=['user_id', 'performance_score'])
print("X_reg_dummy shape:", X_reg_dummy.shape)
print("X_reg_dummy columns:", len(X_reg_dummy.columns))