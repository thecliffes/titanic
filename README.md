# Titanic Survival Prediction

A machine learning model to predict passenger survival on the Titanic, built for the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic).

## Results

| # | Model | Params | Features | Public Score |
|---|-------|--------|----------|--------------|
| 1 | SVM | `C=1, kernel=rbf, gamma=scale` | Original 7 | **0.78229** |
| 2 | Random Forest | `max_depth=None, max_features=sqrt, min_samples_leaf=1, min_samples_split=10, n_estimators=300` | Original 7 | **0.77511** |
| 3 | Random Forest | val-score selection | + Title, FamilySize, IsAlone | **0.77511** |
| 4 | XGBoost | `colsample_bytree=0.8, learning_rate=0.1, max_depth=3, n_estimators=100, subsample=1.0` | + Deck, FareBin | **0.76315** |
| 5 | Random Forest | `colsample_bytree=0.8, learning_rate=0.01, max_depth=3, n_estimators=300, subsample=1.0` | + Title, FamilySize, IsAlone | **0.77511** |
| 6 | XGBoost | `colsample_bytree=0.8, learning_rate=0.01, max_depth=3, n_estimators=300, subsample=1.0` | + Title, FamilySize, IsAlone, CV selection | **0.77990** |

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
| `Age` | Age in years (missing values filled with title-group median) |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Fare` | Ticket fare |
| `Embarked` | Port of embarkation (C=0, Q=1, S=2) |
| `Title` | Extracted from Name (Mr=0, Miss=1, Mrs=2, Master=3, Rare=4) |
| `FamilySize` | SibSp + Parch + 1 |
| `IsAlone` | 1 if FamilySize == 1, else 0 |

## Models

Four models are trained and tuned via `GridSearchCV` (5-fold CV). The best performer is automatically selected for the final submission.

| Model | Tuned Parameters | Combinations |
|-------|-----------------|--------------|
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` | 162 |
| Logistic Regression | `C`, `penalty`, `solver` | 18 |
| XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` | 108 |
| SVM | `C`, `kernel`, `gamma` | 5 |

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
| 3 | Survival rate by passenger class — bar chart with percentage labels |
| 4 | Survival rate by sex & class — annotated heatmap |
| 5 | Age distribution: survived vs died — overlapping KDE |
| 6 | Fare distribution by survival — violin plot (capped at 99th percentile) |
| 7 | Family size vs survival rate — bar chart (engineered feature: SibSp + Parch) |
| 8 | Embarkation port & survival — stacked bar (C/Q/S with full port names) |
| 9 | Feature correlation — annotated Pearson heatmap (lower triangle) |
| 10 | Fare vs age by survival — scatter plot |

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
