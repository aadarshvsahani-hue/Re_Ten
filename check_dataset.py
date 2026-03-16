import pandas as pd

df = pd.read_excel('data/ReTen_Final_Organized_Dataset.xlsx')
print("Columns in dataset:")
print(df.columns.tolist())
print(f"\nNumber of columns: {len(df.columns)}")
print("\nFirst few rows:")
print(df.head())