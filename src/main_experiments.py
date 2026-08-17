# =========================================================
# HOUSE PRICE PREDICTION PROJECT
# Final Machine Learning Pipeline
# =========================================================

import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import root_mean_squared_error, r2_score

from xgboost import XGBRegressor


# =========================================================
# 1. LOAD DATA
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

train = pd.read_csv(BASE_DIR / "data" / "train.csv")
test_original = pd.read_csv(BASE_DIR / "data" / "test.csv")

print("Train Shape:", train.shape)
print("Test Shape:", test_original.shape)


# =========================================================
# 2. SEPARATE FEATURES AND TARGET
# =========================================================

X = train.drop(["Id", "SalePrice"], axis=1)
y = train["SalePrice"]

test = test_original.drop("Id", axis=1)


# =========================================================
# 3. TRAIN / VALIDATION SPLIT
# =========================================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# 4. MISSING VALUE HANDLING
# =========================================================

# Categorical columns where NaN means the feature does not exist

none_columns = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "MasVnrType"
]

X_train[none_columns] = X_train[none_columns].fillna("None")
X_valid[none_columns] = X_valid[none_columns].fillna("None")
test[none_columns] = test[none_columns].fillna("None")


# ----- LotFrontage -----

median_frontage = X_train["LotFrontage"].median()

X_train["LotFrontage"] = X_train["LotFrontage"].fillna(
    median_frontage
)

X_valid["LotFrontage"] = X_valid["LotFrontage"].fillna(
    median_frontage
)

test["LotFrontage"] = test["LotFrontage"].fillna(
    median_frontage
)


# ----- MasVnrArea -----

X_train["MasVnrArea"] = X_train["MasVnrArea"].fillna(0)
X_valid["MasVnrArea"] = X_valid["MasVnrArea"].fillna(0)
test["MasVnrArea"] = test["MasVnrArea"].fillna(0)


# ----- GarageYrBlt -----

X_train["GarageYrBlt"] = X_train["GarageYrBlt"].fillna(0)
X_valid["GarageYrBlt"] = X_valid["GarageYrBlt"].fillna(0)
test["GarageYrBlt"] = test["GarageYrBlt"].fillna(0)


# =========================================================
# 5. FEATURE ENGINEERING
# =========================================================

# ----- House Age -----

X_train["HouseAge"] = (
    X_train["YrSold"] - X_train["YearBuilt"]
)

X_valid["HouseAge"] = (
    X_valid["YrSold"] - X_valid["YearBuilt"]
)

test["HouseAge"] = (
    test["YrSold"] - test["YearBuilt"]
)


# ----- Remodelling Age -----

X_train["RemodAge"] = (
    X_train["YrSold"] - X_train["YearRemodAdd"]
)

X_valid["RemodAge"] = (
    X_valid["YrSold"] - X_valid["YearRemodAdd"]
)

test["RemodAge"] = (
    test["YrSold"] - test["YearRemodAdd"]
)


# ----- Total Bathrooms -----

X_train["TotalBathrooms"] = (
    X_train["FullBath"]
    + 0.5 * X_train["HalfBath"]
    + X_train["BsmtFullBath"]
    + 0.5 * X_train["BsmtHalfBath"]
)

X_valid["TotalBathrooms"] = (
    X_valid["FullBath"]
    + 0.5 * X_valid["HalfBath"]
    + X_valid["BsmtFullBath"]
    + 0.5 * X_valid["BsmtHalfBath"]
)

test["TotalBathrooms"] = (
    test["FullBath"]
    + 0.5 * test["HalfBath"]
    + test["BsmtFullBath"]
    + 0.5 * test["BsmtHalfBath"]
)


# ----- Total SF -----

X_train["TotalSF"] = (
    X_train["TotalBsmtSF"]
    + X_train["1stFlrSF"]
    + X_train["2ndFlrSF"]
)

X_valid["TotalSF"] = (
    X_valid["TotalBsmtSF"]
    + X_valid["1stFlrSF"]
    + X_valid["2ndFlrSF"]
)

test["TotalSF"] = (
    test["TotalBsmtSF"]
    + test["1stFlrSF"]
    + test["2ndFlrSF"]
)


# ----- Total Porch Area -----

X_train["TotalPorchSF"] = (
    X_train["OpenPorchSF"]
    + X_train["3SsnPorch"]
    + X_train["EnclosedPorch"]
    + X_train["ScreenPorch"]
    + X_train["WoodDeckSF"]
)

X_valid["TotalPorchSF"] = (
    X_valid["OpenPorchSF"]
    + X_valid["3SsnPorch"]
    + X_valid["EnclosedPorch"]
    + X_valid["ScreenPorch"]
    + X_valid["WoodDeckSF"]
)

test["TotalPorchSF"] = (
    test["OpenPorchSF"]
    + test["3SsnPorch"]
    + test["EnclosedPorch"]
    + test["ScreenPorch"]
    + test["WoodDeckSF"]
)


# ----- Garage Age -----

X_train["GarageAge"] = np.where(
    X_train["GarageYrBlt"] == 0,
    0,
    X_train["YrSold"] - X_train["GarageYrBlt"]
)

X_valid["GarageAge"] = np.where(
    X_valid["GarageYrBlt"] == 0,
    0,
    X_valid["YrSold"] - X_valid["GarageYrBlt"]
)

test["GarageAge"] = np.where(
    test["GarageYrBlt"] == 0,
    0,
    test["YrSold"] - test["GarageYrBlt"]
)


# ----- Total Living Area -----

X_train["TotalLivingArea"] = (
    X_train["GrLivArea"]
    + X_train["TotalBsmtSF"]
)

X_valid["TotalLivingArea"] = (
    X_valid["GrLivArea"]
    + X_valid["TotalBsmtSF"]
)

test["TotalLivingArea"] = (
    test["GrLivArea"]
    + test["TotalBsmtSF"]
)


# =========================================================
# 6. ONE-HOT ENCODING
# =========================================================

categorical_cols = X_train.select_dtypes(
    include=["object", "string"]
).columns

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoder.fit(X_train[categorical_cols])


X_train_encoded = encoder.transform(
    X_train[categorical_cols]
)

X_valid_encoded = encoder.transform(
    X_valid[categorical_cols]
)

test_encoded = encoder.transform(
    test[categorical_cols]
)


# Convert encoded arrays to DataFrames

encoded_feature_names = encoder.get_feature_names_out(
    categorical_cols
)

X_train_encoded = pd.DataFrame(
    X_train_encoded,
    columns=encoded_feature_names,
    index=X_train.index
)

X_valid_encoded = pd.DataFrame(
    X_valid_encoded,
    columns=encoded_feature_names,
    index=X_valid.index
)

test_encoded = pd.DataFrame(
    test_encoded,
    columns=encoded_feature_names,
    index=test.index
)


# =========================================================
# 7. NUMERIC FEATURES
# =========================================================

numeric_cols = X_train.select_dtypes(
    include=["int64", "float64"]
).columns

X_train_numeric = X_train[numeric_cols]
X_valid_numeric = X_valid[numeric_cols]
test_numeric = test[numeric_cols]


# =========================================================
# 8. FINAL UN-SCALED DATA
#    Used for Tree, Random Forest and XGBoost
# =========================================================

X_train_final = pd.concat(
    [X_train_numeric, X_train_encoded],
    axis=1
)

X_valid_final = pd.concat(
    [X_valid_numeric, X_valid_encoded],
    axis=1
)

test_final = pd.concat(
    [test_numeric, test_encoded],
    axis=1
)


print("\nFinal Feature Shape:")
print("Train:", X_train_final.shape)
print("Validation:", X_valid_final.shape)
print("Test:", test_final.shape)


# =========================================================
# 9. SCALED DATA
#    Used for Linear Regression, Ridge and Lasso
# =========================================================

scaler = StandardScaler()

X_train_numeric_scaled = scaler.fit_transform(
    X_train_numeric
)

X_valid_numeric_scaled = scaler.transform(
    X_valid_numeric
)

test_numeric_scaled = scaler.transform(
    test_numeric
)


X_train_numeric_scaled = pd.DataFrame(
    X_train_numeric_scaled,
    columns=numeric_cols,
    index=X_train.index
)

X_valid_numeric_scaled = pd.DataFrame(
    X_valid_numeric_scaled,
    columns=numeric_cols,
    index=X_valid.index
)

test_numeric_scaled = pd.DataFrame(
    test_numeric_scaled,
    columns=numeric_cols,
    index=test.index
)


X_train_scaled = pd.concat(
    [X_train_numeric_scaled, X_train_encoded],
    axis=1
)

X_valid_scaled = pd.concat(
    [X_valid_numeric_scaled, X_valid_encoded],
    axis=1
)

test_scaled = pd.concat(
    [test_numeric_scaled, test_encoded],
    axis=1
)


# =========================================================
# RESULTS STORAGE
# =========================================================

results = []


def evaluate_model(
    model_name,
    y_train,
    train_predictions,
    y_valid,
    valid_predictions
):

    train_rmse = root_mean_squared_error(
        y_train,
        train_predictions
    )

    valid_rmse = root_mean_squared_error(
        y_valid,
        valid_predictions
    )

    train_r2 = r2_score(
        y_train,
        train_predictions
    )

    valid_r2 = r2_score(
        y_valid,
        valid_predictions
    )

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print("Train RMSE:", train_rmse)
    print("Validation RMSE:", valid_rmse)

    print("Train R²:", train_r2)
    print("Validation R²:", valid_r2)

    results.append({
        "Model": model_name,
        "Train RMSE": train_rmse,
        "Validation RMSE": valid_rmse,
        "Train R²": train_r2,
        "Validation R²": valid_r2
    })


# =========================================================
# 10. LINEAR REGRESSION
# =========================================================

linear_model = LinearRegression()

linear_model.fit(
    X_train_scaled,
    y_train
)

linear_train_predictions = linear_model.predict(
    X_train_scaled
)

linear_valid_predictions = linear_model.predict(
    X_valid_scaled
)

evaluate_model(
    "Linear Regression",
    y_train,
    linear_train_predictions,
    y_valid,
    linear_valid_predictions
)


# =========================================================
# 11. RIDGE REGRESSION
# =========================================================

ridge_model = Ridge(
    alpha=1.0
)

ridge_model.fit(
    X_train_scaled,
    y_train
)

ridge_train_predictions = ridge_model.predict(
    X_train_scaled
)

ridge_valid_predictions = ridge_model.predict(
    X_valid_scaled
)

evaluate_model(
    "Ridge Regression",
    y_train,
    ridge_train_predictions,
    y_valid,
    ridge_valid_predictions
)


# =========================================================
# 12. LASSO REGRESSION
# =========================================================

lasso_model = Lasso(
    alpha=50,
    max_iter=50000,
    random_state=42
)

lasso_model.fit(
    X_train_scaled,
    y_train
)

lasso_train_predictions = lasso_model.predict(
    X_train_scaled
)

lasso_valid_predictions = lasso_model.predict(
    X_valid_scaled
)

evaluate_model(
    "Lasso Regression",
    y_train,
    lasso_train_predictions,
    y_valid,
    lasso_valid_predictions
)


print("\nLasso Feature Selection:")
print("Total Features:", len(lasso_model.coef_))
print(
    "Features Removed:",
    np.sum(lasso_model.coef_ == 0)
)
print(
    "Features Kept:",
    np.sum(lasso_model.coef_ != 0)
)


# =========================================================
# 13. DECISION TREE
# =========================================================

tree_model = DecisionTreeRegressor(
    max_depth=5,
    min_samples_leaf=10,
    random_state=42
)

tree_model.fit(
    X_train_final,
    y_train
)

tree_train_predictions = tree_model.predict(
    X_train_final
)

tree_valid_predictions = tree_model.predict(
    X_valid_final
)

evaluate_model(
    "Decision Tree",
    y_train,
    tree_train_predictions,
    y_valid,
    tree_valid_predictions
)


# =========================================================
# 14. RANDOM FOREST
# =========================================================

forest_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=2,
    min_samples_split=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

forest_model.fit(
    X_train_final,
    y_train
)

forest_train_predictions = forest_model.predict(
    X_train_final
)

forest_valid_predictions = forest_model.predict(
    X_valid_final
)

evaluate_model(
    "Random Forest",
    y_train,
    forest_train_predictions,
    y_valid,
    forest_valid_predictions
)


# =========================================================
# 15. XGBOOST
# =========================================================

xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train_final,
    y_train
)


xgb_train_predictions = xgb_model.predict(
    X_train_final
)

xgb_valid_predictions = xgb_model.predict(
    X_valid_final
)


evaluate_model(
    "XGBoost",
    y_train,
    xgb_train_predictions,
    y_valid,
    xgb_valid_predictions
)


# =========================================================
# 16. XGBOOST FEATURE IMPORTANCE
# =========================================================

feature_importance = pd.DataFrame({
    "Feature": X_train_final.columns,
    "Importance": xgb_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTop 30 XGBoost Features:")
print(
    feature_importance.head(30)
)


# =========================================================
# 17. RESIDUAL ANALYSIS
# =========================================================

residuals = (
    y_valid - xgb_valid_predictions
)

error_df = pd.DataFrame({
    "Actual": y_valid,
    "Predicted": xgb_valid_predictions,
    "Error": residuals,
    "AbsoluteError": abs(residuals)
})

print("\nLargest Prediction Errors:")
print(
    error_df.sort_values(
        "AbsoluteError",
        ascending=False
    ).head(15)
)

print(
    "\nXGBoost MAE:",
    abs(residuals).mean()
)

print(
    "XGBoost RMSE:",
    root_mean_squared_error(
        y_valid,
        xgb_valid_predictions
    )
)


# =========================================================
# 18. 5-FOLD CROSS VALIDATION
# =========================================================

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

xgb_cv_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42,
    n_jobs=-1
)

cv_scores = cross_val_score(
    xgb_cv_model,
    X_train_final,
    y_train,
    cv=kfold,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

cv_rmse = -cv_scores

print("\n" + "=" * 60)
print("5-FOLD CROSS VALIDATION")
print("=" * 60)

print("CV RMSE for each fold:", cv_rmse)
print("Mean CV RMSE:", cv_rmse.mean())
print("Standard Deviation:", cv_rmse.std())


# =========================================================
# 19. FINAL MODEL
# =========================================================
#
# IMPORTANT:
# For the final Kaggle model, combine the original training
# and validation sets so the final model learns from all
# available labelled houses.
#
# We already used the validation set for model selection.
# Therefore, after selecting the model, we can train on all
# labelled data.
# =========================================================

X_full = pd.concat(
    [X_train, X_valid],
    axis=0
)

y_full = pd.concat(
    [y_train, y_valid],
    axis=0
)


# Encode full training data using the existing encoder

X_full_encoded = encoder.transform(
    X_full[categorical_cols]
)

X_full_encoded = pd.DataFrame(
    X_full_encoded,
    columns=encoded_feature_names,
    index=X_full.index
)


X_full_numeric = X_full[numeric_cols]


X_full_final = pd.concat(
    [X_full_numeric, X_full_encoded],
    axis=1
)


# =========================================================
# 20. TRAIN FINAL XGBOOST
# =========================================================

final_xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42,
    n_jobs=-1
)

final_xgb_model.fit(
    X_full_final,
    y_full
)


# =========================================================
# 21. PREDICT TEST DATA
# =========================================================

test_predictions = final_xgb_model.predict(
    test_final
)


# =========================================================
# 22. CREATE KAGGLE SUBMISSION
# =========================================================

submission = pd.DataFrame({
    "Id": test_original["Id"],
    "SalePrice": test_predictions
})


submission_path = BASE_DIR / "submission.csv"

submission.to_csv(
    submission_path,
    index=False
)


print("\n" + "=" * 60)
print("SUBMISSION CREATED")
print("=" * 60)

print(submission.head())

print("\nSubmission Shape:", submission.shape)

print(
    "\nSaved to:",
    submission_path
)


# =========================================================
# 23. FINAL MODEL COMPARISON
# =========================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(
    results_df.sort_values(
        "Validation RMSE"
    ).to_string(index=False)
)