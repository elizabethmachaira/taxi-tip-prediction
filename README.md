# Taxi Tip Prediction — NYC Yellow Cab Data Challenge

Predicts tip amount for NYC yellow taxicab trips using XGBoost with Bayesian hyperparameter optimisation. Final project for the Foundations of Data Science elective at Bocconi.

## Overview

A regression challenge on the NYC TLC trip record dataset. Given trip features (distance, fare, pickup/dropoff zones, time of day, passenger count, etc.), the goal is to predict the dollar tip amount.

## Notebooks

| Notebook | Description |
|---|---|
| `taxi.ipynb` | EDA: tip distribution, correlation analysis, feature exploration |
| `DATA CHALLENGE - PRED.ipynb` | Data cleaning and feature engineering for the test set |
| `HyperOpt.ipynb` | XGBoost model with Bayesian hyperparameter optimisation (hyperopt) |

## Methods

**Feature Engineering**
- Airport ride detection (JFK, LaGuardia), holiday flag, weekend flag
- Spatial features: borough encoding, zone pair lookup
- Trip correction: fare/distance imputation for known flat-rate routes
- Speed calculation and KNN imputation for outlier trip durations

**Modelling**
- XGBoost regressor (`reg:absoluteerror`)
- Bayesian hyperparameter search with `hyperopt` (Tree-structured Parzen Estimator, 100 evaluations)
- Hyperparameters tuned: `max_depth`, `learning_rate`, `n_estimators`, `colsample_bytree`, `subsample`, `gamma`, `reg_alpha`, `reg_lambda`
- Evaluation metric: Mean Absolute Error (MAE)

## Stack

Python · XGBoost · hyperopt · pandas · scikit-learn · geopandas · plotly · seaborn
