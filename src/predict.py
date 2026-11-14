
# ==========================================
# RAINFALL PREDICTION SYSTEM
# Prediction Program
# ==========================================

import os
import joblib
import pandas as pd


# 1. Define project paths

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "rainfall_model.pkl"
)

INPUT_PATH = os.path.join(
    BASE_DIR, "reports", "sample_weather.csv"
)


# 2. Load trained model

print("\nLoading trained rainfall prediction model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# 3. Load weather information

print("\nLoading weather information...")

weather_data = pd.read_csv(INPUT_PATH)

print("Weather information loaded!")


# 4. Generate rainfall prediction

print("\nPredicting rainfall...")

prediction = model.predict(weather_data)[0]

probabilities = model.predict_proba(weather_data)[0]

rain_probability = probabilities[1] * 100
no_rain_probability = probabilities[0] * 100


# 5. Display prediction result

print("\n======================================")
print("       RAINFALL PREDICTION RESULT")
print("======================================")

if prediction == 1:

    print("Prediction: Rain Tomorrow")

else:

    print("Prediction: No Rain Tomorrow")

print(
    "Rain Probability: {:.2f}%".format(
        rain_probability
    )
)

print(
    "No Rain Probability: {:.2f}%".format(
        no_rain_probability
    )
)

print("======================================")
