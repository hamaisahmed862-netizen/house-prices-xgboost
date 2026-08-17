# ============================================================
# HOUSE PRICE PREDICTION - FINAL XGBOOST PIPELINE
# ============================================================

import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import root_mean_squared_error, r2_score
from xgboost import XGBRegressor


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = BASE_DIR / "submission.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_PATH)
test_original = pd.read_csv(TEST_PATH)

print("Train shape:", train.shape)
print("Test shape:", test_original.shape)


# ============================================================
# 3. SEPARATE FEATURES / TARGET
# ============================================================

X = train.drop(["Id", "SalePrice"], axis=1)
y = train["SalePrice"]

test = test_original.drop("Id", axis=1)


# ============================================================
# 4. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Copies prevent accidental modification of the original frames.
X_train = X_train.copy()
X_valid = X_valid.copy()
test = test.copy()


# ============================================================
# 5. MISSING VALUE HANDLING
# ============================================================

# NaN means the house does not have this feature.
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

for col in none_columns:
    X_train[col] = X_train[col].fillna("None")
    X_valid[col] = X_valid[col].fillna("None")
    test[col] = test[col].fillna("None")


# LotFrontage: calculate median from TRAIN only.
median_frontage = X_train["LotFrontage"].median()

X_train["LotFrontage"] = X_train["LotFrontage"].fillna(median_frontage)
X_valid["LotFrontage"] = X_valid["LotFrontage"].fillna(median_frontage)
test["LotFrontage"] = test["LotFrontage"].fillna(median_frontage)


# These numeric NaNs mean the feature is absent.
for df in [X_train, X_valid, test]:
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

def add_features(df):
    df = df.copy()

    # Age of house when sold
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    # Years since remodeling
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    # Garage age
    df["GarageAge"] = np.where(
        df["GarageYrBlt"] == 0,
        0,
        df["YrSold"] - df["GarageYrBlt"]
    )

    # Total bathrooms
    df["TotalBathrooms"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )

    # Basement + first + second floor area
    df["TotalSF"] = (
        df["TotalBsmtSF"]
        + df["1stFlrSF"]
        + df["2ndFlrSF"]
    )

    # Living area + basement
    df["TotalLivingArea"] = (
        df["GrLivArea"]
        + df["TotalBsmtSF"]
    )

    # Total porch/deck area
    df["TotalPorchSF"] = (
        df["OpenPorchSF"]
        + df["3SsnPorch"]
        + df["EnclosedPorch"]
        + df["ScreenPorch"]
        + df["WoodDeckSF"]
    )

    return df


X_train = add_features(X_train)
X_valid = add_features(X_valid)
test = add_features(test)


# ============================================================
# 7. ONE-HOT ENCODING
# ============================================================

categorical_cols = X_train.select_dtypes(
    include=["object", "string"]
).columns

numeric_cols = X_train.select_dtypes(
    include=["int64", "float64"]
).columns

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

# Fit ONLY on training data for the validation experiment.
encoder.fit(X_train[categorical_cols])

X_train_encoded = encoder.transform(X_train[categorical_cols])
X_valid_encoded = encoder.transform(X_valid[categorical_cols])
test_encoded = encoder.transform(test[categorical_cols])

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

X_train_numeric = X_train[numeric_cols]
X_valid_numeric = X_valid[numeric_cols]
test_numeric = test[numeric_cols]


# Tree models do NOT require StandardScaler.
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

# Ensure identical column order.
X_valid_final = X_valid_final.reindex(
    columns=X_train_final.columns,
    fill_value=0
)

test_final = test_final.reindex(
    columns=X_train_final.columns,
    fill_value=0
)

print("\nFinal feature shape:")
print("Train:", X_train_final.shape)
print("Validation:", X_valid_final.shape)
print("Test:", test_final.shape)


# ============================================================
# 8. FINAL XGBOOST SETTINGS
# ============================================================

xgb_params = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "random_state": 42,
    "n_jobs": -1
}


# ============================================================
# 9. VALIDATION MODEL
# ============================================================

xgb_model = XGBRegressor(**xgb_params)

xgb_model.fit(
    X_train_final,
    y_train
)

xgb_train_predictions = xgb_model.predict(X_train_final)
xgb_valid_predictions = xgb_model.predict(X_valid_final)

train_rmse = root_mean_squared_error(
    y_train,
    xgb_train_predictions
)

valid_rmse = root_mean_squared_error(
    y_valid,
    xgb_valid_predictions
)

train_r2 = r2_score(
    y_train,
    xgb_train_predictions
)

valid_r2 = r2_score(
    y_valid,
    xgb_valid_predictions
)

print("\n" + "=" * 60)
print("XGBOOST VALIDATION RESULTS")
print("=" * 60)

print("Train RMSE:", train_rmse)
print("Validation RMSE:", valid_rmse)
print("Train R²:", train_r2)
print("Validation R²:", valid_r2)


# ============================================================
# 10. RESIDUAL / ERROR ANALYSIS
# ============================================================

residuals = y_valid - xgb_valid_predictions

error_df = pd.DataFrame({
    "Actual": y_valid,
    "Predicted": xgb_valid_predictions,
    "Error": residuals,
    "AbsoluteError": np.abs(residuals)
})

print("\nLargest validation errors:")
print(
    error_df
    .sort_values("AbsoluteError", ascending=False)
    .head(15)
)

print("\nValidation MAE:", np.abs(residuals).mean())
print(
    "Validation RMSE:",
    root_mean_squared_error(y_valid, xgb_valid_predictions)
)


# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X_train_final.columns,
    "Importance": xgb_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTop 30 features:")
print(feature_importance.head(30).to_string(index=False))


# ============================================================
# 12. 5-FOLD CROSS VALIDATION
# ============================================================
#
# This is an additional stability check.
# It uses the already prepared training matrix.
#
# The CV score is not the same as the single 80/20 validation
# score, so do not expect them to be identical.
# ============================================================

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_model = XGBRegressor(**xgb_params)

cv_scores = cross_val_score(
    cv_model,
    X_train_final,
    y_train,
    scoring="neg_root_mean_squared_error",
    cv=kfold,
    n_jobs=1
)

cv_rmse = -cv_scores

print("\n" + "=" * 60)
print("5-FOLD CROSS VALIDATION")
print("=" * 60)

print("CV RMSE for each fold:", cv_rmse)
print("Mean CV RMSE:", cv_rmse.mean())
print("Standard Deviation:", cv_rmse.std())


# ============================================================
# 13. FINAL TRAINING ON ALL LABELED DATA
# ============================================================
#
# For the Kaggle submission, we no longer need the validation
# split. We can use all 1460 labeled houses.
#
# IMPORTANT:
# We refit the encoder on the FULL labeled dataset here.
# This allows the final model to learn categories that happened
# to exist only in the old validation portion.
# ============================================================

X_full = X.copy()
y_full = y.copy()
test_full = test.copy()

# Re-apply the same missing-value treatment to full training data.
for col in none_columns:
    X_full[col] = X_full[col].fillna("None")
    test_full[col] = test_full[col].fillna("None")

full_median_frontage = X_full["LotFrontage"].median()

X_full["LotFrontage"] = X_full["LotFrontage"].fillna(
    full_median_frontage
)

test_full["LotFrontage"] = test_full["LotFrontage"].fillna(
    full_median_frontage
)

for df in [X_full, test_full]:
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)

# Same feature engineering as above.
X_full = add_features(X_full)
test_full = add_features(test_full)


# Refit encoder on ALL labeled training data.
full_categorical_cols = X_full.select_dtypes(
    include=["object", "string"]
).columns

full_numeric_cols = X_full.select_dtypes(
    include=["int64", "float64"]
).columns

final_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

final_encoder.fit(X_full[full_categorical_cols])

X_full_encoded = final_encoder.transform(
    X_full[full_categorical_cols]
)

test_full_encoded = final_encoder.transform(
    test_full[full_categorical_cols]
)

final_encoded_names = final_encoder.get_feature_names_out(
    full_categorical_cols
)

X_full_encoded = pd.DataFrame(
    X_full_encoded,
    columns=final_encoded_names,
    index=X_full.index
)

test_full_encoded = pd.DataFrame(
    test_full_encoded,
    columns=final_encoded_names,
    index=test_full.index
)

X_full_numeric = X_full[full_numeric_cols]
test_full_numeric = test_full[full_numeric_cols]

X_full_final = pd.concat(
    [X_full_numeric, X_full_encoded],
    axis=1
)

test_full_final = pd.concat(
    [test_full_numeric, test_full_encoded],
    axis=1
)

test_full_final = test_full_final.reindex(
    columns=X_full_final.columns,
    fill_value=0
)

print("\nFull training feature shape:", X_full_final.shape)
print("Final test feature shape:", test_full_final.shape)


# ============================================================
# 14. TRAIN FINAL MODEL
# ============================================================

final_xgb_model = XGBRegressor(**xgb_params)

final_xgb_model.fit(
    X_full_final,
    y_full
)


# ============================================================
# 15. CREATE KAGGLE PREDICTIONS
# ============================================================

test_predictions = final_xgb_model.predict(
    test_full_final
)

# Safety check: SalePrice cannot be negative.
test_predictions = np.maximum(test_predictions, 0)


# ============================================================
# 16. CREATE SUBMISSION.CSV
# ============================================================

submission = pd.DataFrame({
    "Id": test_original["Id"],
    "SalePrice": test_predictions
})

submission.to_csv(
    SUBMISSION_PATH,
    index=False
)

print("\n" + "=" * 60)
print("KAGGLE SUBMISSION CREATED")
print("=" * 60)

print(submission.head())
print("\nSubmission shape:", submission.shape)
print("Saved to:", SUBMISSION_PATH)
