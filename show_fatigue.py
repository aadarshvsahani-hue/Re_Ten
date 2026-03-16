import pandas as pd

df = pd.read_excel('data/ReTen_Final_Organized_Dataset.xlsx')
print('Fatigue scores for all users:')
for idx, row in df.iterrows():
    print(f'User {row["user_id"]}: Fatigue Score = {row["training_fatigue_score"]:.2f}')