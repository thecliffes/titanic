# Titanic Survival Prediction

A machine learning model to predict passenger survival on the Titanic, built for the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic).

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

## Model

- **Algorithm:** Random Forest Classifier (100 estimators)
- **Train/validation split:** 80/20
- **Validation accuracy:** ~81%

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
| 3 | Train model and evaluate on validation set |
| 4 | Preprocess test data, predict, save `submission.csv` |

## visualisations.ipynb Structure

| Cell | Description |
|------|-------------|
| 1 | Load and preprocess `train.csv` |
| 2 | Deaths by age and sex — histogram split by Male/Female |

## Usage

1. Ensure `train.csv` and `test.csv` are in the same directory as the notebooks
2. Run `model.ipynb` top to bottom to train the model and generate `submission.csv`
3. Run `visualisations.ipynb` independently to explore the data visually
4. Upload the generated `submission.csv` to Kaggle

## Requirements

```
pandas
scikit-learn
seaborn
matplotlib
```
