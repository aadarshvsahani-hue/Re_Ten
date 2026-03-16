# ReTen Backend Project

## Overview
ReTen Backend is a Flask application that utilizes machine learning models to predict performance scores and classify users based on their data. The application is designed to provide insights into user behavior and performance metrics.

## Project Structure
```
ReTen_Backend
├── app.py                          # Flask application code
├── scaler.joblib                   # Trained scaler for normalizing numerical features
├── model_rf_reg.joblib             # Trained Random Forest Regressor model
├── model_xgb_clf.joblib            # Trained XGBoost Classifier model
├── model_clf.joblib                # Secondary Logistic Regression model for probabilities
├── ReTen_Final_Organized_Dataset.xlsx # Dataset used for training the models
├── requirements.txt                # Python package dependencies
└── README.md                       # Project documentation
```

## Setup Instructions
1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd ReTen_Backend
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`.
   ```
   pip install -r requirements.txt
   ```

3. **Run the application**:
   Start the Flask application by running:
   ```
   python app.py
   ```
   The application will be accessible at `http://localhost:5000`.

## Testing the API

### Sample Test Data
Use the provided `test_sample.py` script to test the API endpoints with sample data.

1. Start the Flask app:
   ```
   python app.py
   ```

2. In another terminal, run the test script:
   ```
   python test_sample.py
   ```

### Manual Testing with curl

#### Predict Performance Score
```bash
curl -X POST http://localhost:5000/predict_performance \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user_1",
    "age": 25,
    "height": 175.5,
    "goal": "muscle_gain",
    "week": 5,
    "workout_days": 4,
    "missed_days": 1,
    "avg_duration": 60.0,
    "avg_intensity": 7.5,
    "training_load": 180.0,
    "avg_sleep": 7.5,
    "sleep_score": 85.0,
    "fatigue": 3.0,
    "recovery": 8.0,
    "stress_level": 4.0,
    "motivation": 8.5,
    "illness": 0,
    "injury": 0
  }'
```

#### Classify User
```bash
curl -X POST http://localhost:5000/classify_user \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user_1",
    "age": 25,
    "height": 175.5,
    "goal": "muscle_gain",
    "week": 5,
    "workout_days": 4,
    "missed_days": 1,
    "avg_duration": 60.0,
    "avg_intensity": 7.5,
    "training_load": 180.0,
    "avg_sleep": 7.5,
    "sleep_score": 85.0,
    "fatigue": 3.0,
    "recovery": 8.0,
    "stress_level": 4.0,
    "motivation": 8.5,
    "illness": 0,
    "injury": 0
  }'
```

### Expected Responses

#### Performance Prediction
```json
{
  "performance_score": 85.7
}
```

#### User Classification
```json
{
  "user_type_prediction": 1,
  "is_regular_user_proba": 0.85,
  "is_irregular_user_proba": 0.15
}
```

## Models
- **Random Forest Regressor**: Used for predicting performance scores based on user data.
- **XGBoost Classifier**: Used for classifying users into different categories.
- **Logistic Regression**: Provides probabilities for user classification.

## Dataset
The dataset used for training the models is stored in `ReTen_Final_Organized_Dataset.xlsx`. It contains various features relevant to the predictions.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.