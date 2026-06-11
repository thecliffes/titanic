# Titanic Survival Prediction

A machine learning model to predict passenger survival on the Titanic, built for the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic).

## Results

| Submission | Model | Params | Public Score |
|------------|-------|--------|--------------|
| 1 | SVM | `C=1, kernel=rbf, gamma=scale` | **0.78229** |

## Dataset

| File | Description |
|------|-------------|
| `train.csv` | 891 passengers with known survival outcomes |
| `test.csv` | 418 passengers to predict |

## Features Used

| Feature | Description |
|---------|-------------|
| `Pclass` | Passenger class (1st, 2nd, 3rd) |
| `Sex` | Gender (encoded: male=0, female=1) |
| `Age` | Age in years (missing values filled with mean) |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Fare` | Ticket fare |
| `Embarked` | Port of embarkation (C=0, Q=1, S=2) |

## Models

Four models are trained and tuned via `GridSearchCV` (5-fold CV). The best performer is automatically selected for the final submission.

| Model | Tuned Parameters | Combinations |
|-------|-----------------|--------------|
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` | 162 |
| Logistic Regression | `C`, `penalty`, `solver` | 18 |
| XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` | 108 |
| SVM | `C`, `kernel`, `gamma` | 20 |

## Files

| File | Description |
|------|-------------|
| `model.ipynb` | Data loading, preprocessing, model training and prediction |
| `visualisations.ipynb` | Visual analysis of the Titanic dataset |

## model.ipynb Structure

| Cell | Description |
|------|-------------|
| 1 | Load `train.csv` and `test.csv` |
| 2 | Impute missing values, encode categorical features |
| 3 | Baseline Random Forest — 80/20 train/val split |
| 4 | Hyperparameter tuning — Random Forest (162 combinations, 5-fold CV) |
| 5 | Hyperparameter tuning — Logistic Regression (18 combinations, 5-fold CV) |
| 6 | Hyperparameter tuning — XGBoost (108 combinations, 5-fold CV) |
| 7 | Hyperparameter tuning — SVM (20 combinations, 5-fold CV) |
| 8 | Preprocess test data, predict with best model, save `submission.csv` |

## visualisations.ipynb Structure

| Cell | Description |
|------|-------------|
| 1 | Load and preprocess `train.csv` |
| 2 | Deaths by age and sex — histogram split by Male/Female |

## Usage

1. Ensure `train.csv` and `test.csv` are in the same directory as the notebooks
2. Run `model.ipynb` top to bottom — all four models are tuned and the best is selected automatically
3. Upload the generated `submission.csv` to Kaggle
4. Run `visualisations.ipynb` independently to explore the data visually

## Requirements

```
pandas
scikit-learn
xgboost
seaborn
matplotlib
```
