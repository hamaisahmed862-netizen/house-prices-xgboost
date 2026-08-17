# House Prices Prediction using XGBoost

A machine learning project for predicting residential house prices using the **House Prices - Advanced Regression Techniques** dataset from Kaggle.

The project focuses on data preprocessing, feature engineering, categorical encoding, model training, evaluation, feature importance analysis, cross-validation, and generating predictions for Kaggle submission.

---

## Project Overview

The goal of this project is to predict the `SalePrice` of houses based on features such as:

- Overall quality
- Living area
- Basement area
- Garage information
- Number of bathrooms
- Year built
- Remodeling information
- Kitchen quality
- Neighborhood and other categorical features

Several regression models were experimented with, with **XGBoost** achieving the best validation performance.

---

## Dataset

The dataset used is:

**House Prices - Advanced Regression Techniques**

It contains:

- `train.csv` — 1,460 training records
- `test.csv` — 1,459 records
- `sample_submission.csv` — Kaggle submission format
- `data_description.txt` — Description of the dataset features

The dataset files are not included in this GitHub repository.

---

## Machine Learning Pipeline

The project follows these main steps:

1. Load the dataset
2. Separate features and target
3. Split the training data into training and validation sets
4. Handle missing values
5. Perform feature engineering
6. Encode categorical variables using One-Hot Encoding
7. Combine numerical and encoded categorical features
8. Train regression models
9. Evaluate models using RMSE, MAE, and R²
10. Analyze XGBoost feature importance
11. Perform 5-fold cross-validation
12. Train the final model on the complete training dataset
13. Generate predictions for the Kaggle test dataset
14. Create `submission.csv`

---

## Feature Engineering

Several additional features were created to provide the model with more meaningful information.

### House Age

```text
HouseAge = YrSold - YearBuilt
````

### Remodeling Age

```text
RemodAge = YrSold - YearRemodAdd
```

### Total Bathrooms

Half bathrooms are given a weight of 0.5:

```text
TotalBathrooms =
    FullBath
    + 0.5 × HalfBath
    + BsmtFullBath
    + 0.5 × BsmtHalfBath
```

### Total Square Footage

```text
TotalSF =
    TotalBsmtSF
    + 1stFlrSF
    + 2ndFlrSF
```

### Total Living Area

```text
TotalLivingArea =
    GrLivArea
    + TotalBsmtSF
```

Additional missing-value handling was also performed for selected numerical and categorical features.

---

## Models Tested

The project experimented with several regression algorithms:

| Model             | Validation RMSE | Validation R² |
| ----------------- | --------------: | ------------: |
| **XGBoost**       |   **24,121.13** |    **0.9241** |
| Lasso Regression  |       27,829.47 |        0.8990 |
| Ridge Regression  |       29,726.93 |        0.8848 |
| Random Forest     |       32,651.35 |        0.8610 |
| Decision Tree     |       37,018.69 |        0.8213 |
| Linear Regression |       65,473.99 |        0.4411 |

Based on the validation results, **XGBoost performed the best** among the tested models.

---

## Final XGBoost Model

The final XGBoost model uses:

```python
XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42,
    n_jobs=-1
)
```

### Validation Results

```text
Train RMSE:       8847.71
Validation RMSE: 24121.13

Train R²:        0.9869
Validation R²:   0.9241

Validation MAE: 15183.76
```

The model achieved a validation R² of approximately **92.4%**.

---

## Feature Importance

The most important features identified by XGBoost included:

1. `OverallQual`
2. `TotalSF`
3. `GarageCars`
4. `KitchenQual_TA`
5. `BsmtQual_Ex`
6. `TotalBathrooms`
7. `TotalLivingArea`
8. `KitchenQual_Ex`
9. `KitchenAbvGr`
10. `GarageType_Attchd`

`OverallQual` was the most important feature in the final model.

---

## Cross-Validation

5-fold cross-validation was also performed.

```text
Fold 1 RMSE: 24907.82
Fold 2 RMSE: 34642.59
Fold 3 RMSE: 32244.38
Fold 4 RMSE: 19866.34
Fold 5 RMSE: 19700.52

Mean CV RMSE: 26272.33
Standard Deviation: 6193.77
```

The variation between folds indicates that model performance differs depending on the validation subset.

---

## Project Structure

```text
house-prices-advanced-regression-techniques/
│
├── data/
│   └── data_description.txt
│
├── src/
│   ├── main.py
│   └── main_experiments.py
│
├── .gitignore
└── README.md
```

The Kaggle dataset files and generated submission file are excluded from the repository using `.gitignore`.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Git
* GitHub
* Kaggle

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/hamaisahmed862-netizen/house-prices-xgboost.git
```

### 2. Navigate to the project

```bash
cd house-prices-xgboost
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install required libraries

```bash
pip install pandas numpy scikit-learn xgboost matplotlib
```

### 6. Add the Kaggle dataset

Place the following files inside the `data/` folder:

```text
data/
├── train.csv
├── test.csv
├── sample_submission.csv
└── data_description.txt
```

### 7. Run the model

```bash
python src/main.py
```

The program trains the model, evaluates it, performs feature importance analysis and cross-validation, and generates:

```text
submission.csv
```

---

## Kaggle Submission

The final model generates predictions in the required Kaggle format:

```text
Id,SalePrice
1461,...
1462,...
1463,...
```

The generated `submission.csv` can be uploaded to the Kaggle House Prices competition.

---

## Results

The final XGBoost model achieved:

**Validation RMSE: 24,121.13**

**Validation R²: 0.9241**

The Kaggle submission achieved a score of approximately **0.13119**.

---

## Author

**Hamais Ahmed**

GitHub:
[https://github.com/hamaisahmed862-netizen](https://github.com/hamaisahmed862-netizen)

