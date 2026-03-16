from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier

app = Flask(__name__)

# Load the scaler
scaler_filename = 'models/scaler.joblib'
loaded_scaler = joblib.load(scaler_filename)

# Load the trained models
model_rf_reg = joblib.load('models/model_rf_reg.joblib')
model_xgb_clf = joblib.load('models/model_xgb_clf.joblib')
model_clf = joblib.load('models/model_clf.joblib')

# Define the preprocessing function (must match the one used during training)
def preprocess_input_data(data, scaler, X_columns, categorical_cols, numerical_cols):
    # Convert input JSON data to a pandas DataFrame
    input_df = pd.DataFrame([data])

    # Convert 'user_id' to string if it exists in input_df, just in case
    if 'user_id' in input_df.columns:
        input_df['user_id'] = input_df['user_id'].astype(str)

    # Manually encode categorical features to match training encoding
    if 'gender' in input_df.columns:
        input_df['gender_Male'] = (input_df['gender'] == 'Male').astype(int)
        input_df = input_df.drop(columns=['gender'])

    if 'Supplements_Taken' in input_df.columns:
        input_df['Supplements_Taken_Multivitamin'] = (input_df['Supplements_Taken'] == 'Multivitamin').astype(int)
        input_df['Supplements_Taken_Protein'] = (input_df['Supplements_Taken'] == 'Protein').astype(int)
        input_df = input_df.drop(columns=['Supplements_Taken'])

    # Drop target columns if present in input (shouldn't be, but just in case)
    if 'performance_score' in input_df.columns:
        input_df = input_df.drop(columns=['performance_score'])
    if 'Is_Regular_User' in input_df.columns:
        input_df = input_df.drop(columns=['Is_Regular_User'])

    # Align columns with the training data
    processed_input = pd.DataFrame(0, index=input_df.index, columns=X_columns)
    for col in input_df.columns:
        if col in processed_input.columns:
            processed_input[col] = input_df[col]

    # Drop 'user_id_numeric' if it was part of the original X_columns but not intended for prediction
    if 'user_id_numeric' in processed_input.columns: # Assuming user_id_numeric is not a feature for the model
        processed_input = processed_input.drop(columns=['user_id_numeric'])

    # Apply scaling to numerical features
    processed_input[numerical_cols_to_scale] = scaler.transform(processed_input[numerical_cols_to_scale])

    return processed_input

    return processed_input

# --- Regression Scenario (Predicting performance_score) ---
# These variables need to be globally available or re-derived for the API to work
# Re-deriving them here based on known structure from notebook execution
file_path_preprocessed = 'data/ReTen_Final_Organized_Dataset.xlsx'
df_preprocessed = pd.read_excel(file_path_preprocessed)

df_preprocessed['Daily_Calorie_intake'] = df_preprocessed['Daily_Calorie_intake'].fillna(df_preprocessed['Daily_Calorie_intake'].median())
df_preprocessed['Supplements_Taken'] = df_preprocessed['Supplements_Taken'].fillna(df_preprocessed['Supplements_Taken'].mode()[0])
df_preprocessed['performance_score'] = df_preprocessed['performance_score'].fillna(df_preprocessed['performance_score'].median())

# Define categorical and numerical columns from the training data for preprocessing
categorical_cols = [col for col in df_preprocessed.select_dtypes(include='object').columns.tolist() if col != 'user_id']

categorical_cols_to_encode = [col for col in df_preprocessed.select_dtypes(include='object').columns.tolist() if col != 'user_id']
print("Categorical cols to encode:", categorical_cols_to_encode)
df_encoded_temp = pd.get_dummies(df_preprocessed, columns=categorical_cols_to_encode, drop_first=True, dtype=int)
print("Encoded df shape:", df_encoded_temp.shape)
print("Has performance_score:", 'performance_score' in df_encoded_temp.columns)

numerical_columns = df_encoded_temp.select_dtypes(include=np.number).columns.tolist()
binary_columns = []
for col in numerical_columns:
    unique_values = df_encoded_temp[col].unique()
    if len(unique_values) <= 2 and all(val in [0, 1] for val in unique_values):
        binary_columns.append(col)
numerical_cols_to_scale = [col for col in numerical_columns if col not in binary_columns]

y_reg_dummy = df_encoded_temp['performance_score']
X_reg_dummy = df_encoded_temp.drop(columns=['user_id', 'performance_score'])
X_columns_reg = X_reg_dummy.columns.tolist()

# Fix: numerical_cols_to_scale should be from X_reg_dummy
numerical_cols_to_scale = [col for col in X_reg_dummy.select_dtypes(include=np.number).columns.tolist() if col not in binary_columns]
print("numerical_cols_to_scale length:", len(numerical_cols_to_scale))
print("numerical_cols_to_scale[:5]:", numerical_cols_to_scale[:5])

# Assuming df_encoded from notebook history had 'user_id_numeric' and 'Is_Regular_User' added later
# To get X_columns_clf correctly, we need to rebuild it or ensure the current X_reg_dummy is aligned.
# For simplicity, we'll assume the final X_clf from notebook was derived similarly without specific columns.
# However, `user_id_numeric` is created in `c7945970` and dropped from `X_clf`. Need to ensure it's not in `X_columns_clf`.
# The X_columns_clf should reflect X_clf which had 'user_id', 'user_id_numeric', 'performance_score', 'Is_Regular_User' dropped.

# Recreate a dummy X_clf to get its column names correctly
df_encoded_for_clf_dummy = df_encoded_temp.copy()
# These columns might exist from previous steps in the notebook context, if re-running from scratch, they need to be recreated
# For robustness in API, we ensure that the columns used for prediction are exactly those expected by the model.
# The original `df_encoded` had 42 columns after feature engineering, but before `user_id_numeric` and `Is_Regular_User` were added.
# The X_reg had 40 columns. X_clf also had 40 columns by dropping these additional ones. Thus, X_reg_dummy.columns will be the same.
X_columns_clf = X_reg_dummy.columns.tolist() # This should be the correct list of feature columns for both regression and classification


@app.route('/predict_performance', methods=['POST'])
def predict_performance():
    print("Received request")
    data = request.get_json(force=True)
    print("Data received")

    processed_input = preprocess_input_data(data, loaded_scaler, X_columns_reg, categorical_cols, numerical_cols_to_scale)
    print("Input processed, shape:", processed_input.shape)

    # Ensure correct dtypes
    processed_input = processed_input.astype(np.float32)

    # Make prediction using the Random Forest Regressor model
    prediction = model_rf_reg.predict(processed_input)[0]
    print("Prediction made:", prediction)

    return jsonify({'performance_score': prediction})


@app.route('/classify_user', methods=['POST'])
def classify_user():
    try:
        data = request.get_json(force=True)

        processed_input = preprocess_input_data(data, loaded_scaler, X_columns_clf, categorical_cols, numerical_cols_to_scale)

        # Ensure correct dtypes
        processed_input = processed_input.astype(np.float32)

        # Make prediction using the primary XGBoost Classifier model
        primary_prediction = model_xgb_clf.predict(processed_input)[0]

        # Get probabilities from the secondary Logistic Regression model for interpretability
        secondary_probabilities = model_clf.predict_proba(processed_input)[0]

        return jsonify({
            'user_type_prediction': int(primary_prediction), # Convert numpy int to Python int
            'is_regular_user_proba': secondary_probabilities[1].tolist(), # Probability of being 'regular' (class 1)
            'is_irregular_user_proba': secondary_probabilities[0].tolist() # Probability of being 'irregular' (class 0)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/suggest_workout', methods=['POST'])
def suggest_workout():
    try:
        data = request.get_json(force=True)

        # Get fatigue score from input or predict it
        fatigue = data.get('training_fatigue_score', 25)  # default if not provided

        # Basic suggestion logic
        suggestions = []

        if fatigue > 30:
            suggestions.append("High fatigue detected. Suggest rest or light cardio (20-30 min walking/jogging).")
            suggestions.append("Focus on recovery: Ensure 7-9 hours sleep, hydration, and nutrition.")
        elif fatigue > 20:
            suggestions.append("Moderate fatigue. Suggest balanced workout: 45-60 min mixed training.")
            suggestions.append("Include compound lifts with moderate weights.")
        else:
            suggestions.append("Low fatigue. Ready for intense training: 60-90 min heavy lifting.")
            suggestions.append("Focus on progressive overload: Increase weights/reps where possible.")

        # Age-based adjustments
        age = data.get('age', 25)
        if age > 35:
            suggestions.append("Age consideration: Include more mobility work and recovery days.")
        elif age < 25:
            suggestions.append("Young athlete: Can handle higher intensity, but monitor form.")

        # Gender-based (general)
        gender = data.get('gender', 'Male')
        if gender == 'Female':
            suggestions.append("Consider hormonal cycles: Adjust intensity during certain phases.")

        # Performance-based
        # Since we have the model, perhaps predict performance and suggest based on that
        processed_input = preprocess_input_data(data, loaded_scaler, X_columns_reg, categorical_cols, numerical_cols_to_scale)
        processed_input = processed_input.astype(np.float32)
        predicted_perf = model_rf_reg.predict(processed_input)[0]

        if predicted_perf > 85:
            suggestions.append("High predicted performance: Challenge yourself with advanced exercises.")
        elif predicted_perf < 70:
            suggestions.append("Lower predicted performance: Focus on form and fundamentals.")

        # Specific exercise suggestions based on current data
        bench_reps = data.get('Barbell Bench Press_Reps', 8)
        if bench_reps < 6:
            suggestions.append("Bench Press: Low reps - Focus on increasing strength with heavier weights.")
        elif bench_reps > 10:
            suggestions.append("Bench Press: High reps - Good endurance, try adding weight for strength.")

        deadlift_reps = data.get('Deadlift_Reps', 5)
        if deadlift_reps < 5:
            suggestions.append("Deadlift: Low reps - Prioritize technique to avoid injury.")
        else:
            suggestions.append("Deadlift: Good rep range - Continue building strength.")

        return jsonify({'workout_suggestions': suggestions, 'predicted_performance': predicted_perf})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict_next_day', methods=['POST'])
def predict_next_day():
    try:
        data = request.get_json(force=True)

        # Use the classification model as proxy for regularity (high performer = more likely regular)
        processed_input = preprocess_input_data(data, loaded_scaler, X_columns_clf, categorical_cols, numerical_cols_to_scale)
        processed_input = processed_input.astype(np.float32)

        # Primary prediction (high performer = 1, low = 0)
        primary_prediction = model_xgb_clf.predict(processed_input)[0]

        # Probabilities
        secondary_probabilities = model_clf.predict_proba(processed_input)[0]

        # Interpret as next day attendance: if high performer, more likely to come
        will_come_proba = secondary_probabilities[1]  # probability of being high performer (regular)

        prediction = "Likely to attend" if will_come_proba > 0.5 else "Unlikely to attend"

        return jsonify({
            'next_day_prediction': prediction,
            'attendance_probability': will_come_proba,
            'performance_class': int(primary_prediction),  # 1 = high performer, 0 = low
            'note': 'Prediction based on performance classification as proxy for regularity'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
