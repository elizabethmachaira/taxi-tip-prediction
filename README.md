# Taxi Tip Prediction — NYC Yellow Cab Data Challenge

Predicts tip amount for NYC yellow taxicab trips using regression modelling and feature engineering on the TLC trip record dataset.

## Overview

This was the final project for the Foundations of Data Science elective. The task is a regression challenge: given trip features (distance, fare, pickup/dropoff zones, time of day, etc.), predict the tip amount left by passengers.

## Notebooks

| Notebook | Description |
|---|---|
| `DATA CHALLENGE - PRED.ipynb` | Main pipeline: feature engineering, modelling, final predictions |
| `taxi.ipynb` | EDA and data cleaning |
| `HyperOpt.ipynb` | Hyperparameter optimisation experiments |

## Methods

- Exploratory data analysis of NYC TLC trip records
- Feature engineering: temporal features, spatial zone encoding, fare decomposition
- Models: linear regression, gradient boosting (XGBoost/LightGBM), random forest
- Hyperparameter tuning with cross-validation

## Stack

Python · pandas · scikit-learn · XGBoost · matplotlib · seaborn
