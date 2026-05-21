# AI Disease Risk Prediction System

An AI-powered diabetes risk prediction project using Python, Pandas,
Scikit-learn, and Streamlit.

## Setup

Install the required packages:

```powershell
pip install -r requirements.txt
```

## Run the Streamlit App

```powershell
streamlit run dashboard/app.py
```

The app loads `models/diabetes_model.pkl` when possible. If the saved model is
missing or incompatible, it trains a Random Forest model from
`data/diabetes.csv` and saves a fresh model automatically.

## Project Structure

- `dashboard/app.py` - Streamlit prediction app
- `data/diabetes.csv` - diabetes dataset
- `models/diabetes_model.pkl` - saved model
- `notebooks/eda.ipynb` - exploratory analysis and model training notebook

# AI-Powered Disease Risk Prediction System

## Overview
This project is a machine learning-based healthcare analytics system designed to predict diabetes risk using patient medical data.

## Features
- Diabetes risk prediction
- Exploratory Data Analysis (EDA)
- Logistic Regression & Random Forest models
- Accuracy evaluation
- Confusion Matrix & Classification Report
- Streamlit web application
- Real-time prediction system

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Machine Learning Workflow
1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis
4. Feature Selection
5. Model Training
6. Model Evaluation
7. Model Saving
8. Streamlit Deployment

## Best Model
Logistic Regression achieved the best performance with approximately 74.67% accuracy.

## Future Scope
- Multi-disease prediction
- Advanced ML algorithms
- Online deployment
- Healthcare dashboard enhancement