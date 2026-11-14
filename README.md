
# Rainfall Prediction System

## Project Overview

The Rainfall Prediction System is an end-to-end Machine Learning project that predicts whether rainfall will occur tomorrow using historical weather data.

The system uses weather parameters such as temperature, humidity, wind speed, atmospheric pressure and rainfall information.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Machine Learning Algorithm

Random Forest Classifier

## Dataset

Rain in Australia Dataset.

Source:
https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package

Download weatherAUS.csv and place it inside the data folder.

## Project Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Engineering
4. Train-Test Splitting
5. Model Training
6. Model Evaluation
7. Rainfall Prediction

## Model Performance

- Accuracy: 82.21%
- Algorithm: Random Forest Classifier
- Evaluation: Accuracy, Precision, Recall, F1-score and Confusion Matrix

## Project Structure

rainfall-prediction-system/

- data/
- src/
  - train.py
  - predict.py
- models/
- reports/
- requirements.txt
- README.md
- .gitignore

## Installation

Clone the repository:

git clone YOUR_REPOSITORY_URL

Install dependencies:

pip install -r requirements.txt

Download the dataset and place weatherAUS.csv inside data/.

## Run the Project

Train the model:

python src/train.py

Generate prediction:

python src/predict.py

## Output

The model predicts whether rainfall will occur tomorrow based on weather parameters.

## Author

Machine Learning Project
