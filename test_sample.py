import requests
import json

# Sample input data for testing the ReTen Backend API
# Based on the features from Dataset.py

sample_input = {
  "age": 28,
  "gender": "Male",
  "height": 179.4,
  "weight": 65.7,
  "heart_rate_avg": 87,
  "heart_rate_max": 187,
  "resting_heart_rate": 56,
  "total_sleep_time": 6.9,
  "sleep_quality_score": 1,
  "Daily_Calorie_intake": 2157,
  "Supplements_Taken": "Protein",
  "workout_time_weighted_training": 43.0,
  "workout_time_bodyweight_training": 37.2,
  "workout_time_cardio": 89.7,
  "total_workout_session_time": 169.9,
  "perceived_exertion": 4,
  "total_calories_burned_weighted_training": 222.2830212818837,
  "total_calories_burned_bodyweight_training": 164.0672509566791,
  "total_calories_burned_cardio": 590.3365653216586,
  "training_fatigue_score": 29.96769406013505,
  "stress_score": 34.99625927742554,
  "Barbell Bench Press_Weight": 62.5,
  "Barbell Bench Press_Reps": 8,
  "Barbell Bench Press_Sets": 1,
  "Deadlift_Weight": 171.2,
  "Deadlift_Reps": 4,
  "Deadlift_Sets": 3,
  "Back Squat_Weight": 133.6,
  "Back Squat_Reps": 6,
  "Back Squat_Sets": 5,
  "Overhead Press_Weight": 69.6,
  "Overhead Press_Reps": 1,
  "Overhead Press_Sets": 2,
  "Pull-ups_Weight": 7.3,
  "Pull-ups_Reps": 6,
  "Pull-ups_Sets": 4
}

# Note: 'retained' is not included as it's the target variable for classification
# The API will preprocess this data and make predictions

def test_predict_performance():
    """Test the /predict_performance endpoint"""
    url = "http://localhost:5000/predict_performance"
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, data=json.dumps(sample_input), headers=headers)
        if response.status_code == 200:
            result = response.json()
            print("Performance Prediction Success:")
            print(f"Predicted performance_score: {result['performance_score']}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

def test_classify_user():
    """Test the /classify_user endpoint"""
    url = "http://localhost:5000/classify_user"
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, data=json.dumps(sample_input), headers=headers)
        if response.status_code == 200:
            result = response.json()
            print("User Classification Success:")
            print(f"User type prediction: {result['user_type_prediction']}")
            print(f"Probability of being regular user: {result['is_regular_user_proba']}")
            print(f"Probability of being irregular user: {result['is_irregular_user_proba']}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    print("Testing ReTen Backend API with sample data:")
    print(json.dumps(sample_input, indent=2))
    print("\n" + "="*50)

    # Test both endpoints
    test_predict_performance()
    print("\n" + "="*50)
    test_classify_user()