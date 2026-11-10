# Flight Fare Prediction

## Project Overview
This project uses Machine Learning to predict flight ticket prices based on flight-related information such as airline, journey date, source, destination, departure time, arrival time, duration, and number of stops.

The objective is to develop a regression model that estimates flight fares using historical flight data.

## Dataset
- Original records: 10,683
- Original features: 10
- Target variable: `Price`
- Data cleaning and feature engineering were performed before model training.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Machine Learning Workflow
1. Dataset Loading
2. Data Understanding and Quality Checks
3. Exploratory Data Analysis (EDA)
4. Data Cleaning
5. Feature Engineering
6. Feature Selection
7. Data Preprocessing
8. Train-Test Split
9. Model Building
10. Model Evaluation
11. Hyperparameter Tuning
12. Model Export using Pickle

## Models Evaluated
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Optimized Random Forest Regressor

## Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 1542.87 | 2362.98 | 0.7301 |
| Decision Tree | 976.86 | 1917.86 | 0.8222 |
| Random Forest | 634.21 | 1339.28 | 0.9133 |
| Optimized Random Forest | 615.69 | 1324.50 | 0.9152 |

The Optimized Random Forest achieved the best test performance among the evaluated models.

## Model Export
The trained model and preprocessing object are saved together in:

`flight_fare_model.pkl`

The package includes:
- Preprocessor
- Trained Random Forest model
- Categorical feature names
- Numerical feature names
- Scikit-learn version
