# NFL Play Data Analysis and Prediction

## Project Overview
This project aims to predict the outcome of NFL plays based on various features such as play type, yards to gain, offense formation, and defensive personnel. Using data science techniques, the project provides insights that can help defensive coordinators plan their game strategies more effectively.

## Features
- **Exploratory Data Analysis (EDA)**: Analyze and visualize the dataset to understand the distribution and relationships between variables.
- **Data Cleaning and Feature Engineering**: Process and clean the data, handle missing values, and convert categorical variables into a suitable format for modeling.
- **Modeling**: Implement Logistic Regression and XGBoost models to predict the outcome of plays.
- **Hyperparameter Tuning**: Use RandomizedSearchCV for optimizing the hyperparameters of the XGBoost model.
- **Email Notifications**: Send email notifications with the results of the model runs or any errors encountered during the process.

## Dataset
The dataset used in this project is extracted from the NFL Big Data Bowl 2021. It contains play-level information for each game.

## Dependencies
The project relies on the following libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn
- xgboost
- missingno
- yellowbrick
- smtplib

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/moe94z/NFL_Data_Predictive_Modeling_Analytics.git
   cd nfl-play-prediction

## Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt

