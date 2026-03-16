import requests
import json

# Simple test data
test_data = {
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

url = "http://localhost:5000/predict_next_day"
headers = {'Content-Type': 'application/json'}

try:
    response = requests.post(url, data=json.dumps(test_data), headers=headers, timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")