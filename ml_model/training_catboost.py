import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, Pool
import joblib
import matplotlib.pyplot as plt

MODEL_VARIANT = "CATBOOST_IMPROVED"
TARGET_COL = "price"

os.makedirs("models", exist_ok=True)

print("=== WCZYTYWANIE DANYCH ===")
train_df = pd.read_csv("train_otodom.csv", low_memory=False)
test_df = pd.read_csv("test_otodom.csv", low_memory=False)

print(f"\n=== INFORMACJE O DANYCH (PRZED CZYSZCZENIEM) ===")
print(f"Rozmiar train: {len(train_df):,}")
print(f"Rozmiar test: {len(test_df):,}")

print(f"\n=== ANALIZA I USUWANIE OUTLIERÓW ===")

y_all = train_df[TARGET_COL]
print(f"Przed usunięciem outlierów:")
print(f"  Min: {y_all.min():,.0f} PLN")
print(f"  Max: {y_all.max():,.0f} PLN")
print(f"  Median: {y_all.median():,.0f} PLN")
print(f"  Mean: {y_all.mean():,.0f} PLN")

# Percentyle
q01 = y_all.quantile(0.01)
q99 = y_all.quantile(0.99)
q995 = y_all.quantile(0.995)

print(f"\n  1% percentyl: {q01:,.0f} PLN")
print(f"  99% percentyl: {q99:,.0f} PLN")
print(f"  99.5% percentyl: {q995:,.0f} PLN")

mask_train = (train_df[TARGET_COL] >= q01) & (train_df[TARGET_COL] <= q99)
train_df_clean = train_df[mask_train].copy()

mask_test = (test_df[TARGET_COL] >= q01) & (test_df[TARGET_COL] <= q99)
test_df_clean = test_df[mask_test].copy()

print(f"\nUsunięto:")
print(f"  Train: {(~mask_train).sum()} outlierów ({(~mask_train).sum() / len(train_df) * 100:.1f}%)")
print(f"  Test: {(~mask_test).sum()} outlierów ({(~mask_test).sum() / len(test_df) * 100:.1f}%)")

y_clean = train_df_clean[TARGET_COL]
print(f"\nPo usunięciu outlierów:")
print(f"  Min: {y_clean.min():,.0f} PLN")
print(f"  Max: {y_clean.max():,.0f} PLN")
print(f"  Median: {y_clean.median():,.0f} PLN")
print(f"  Mean: {y_clean.mean():,.0f} PLN")

X_train = train_df_clean.drop(columns=[TARGET_COL])
y_train = train_df_clean[TARGET_COL]
X_test = test_df_clean.drop(columns=[TARGET_COL])
y_test = test_df_clean[TARGET_COL]

print(f"\n=== FINALNE ROZMIARY DANYCH ===")
print(f"Train: {len(X_train):,} przykładów")
print(f"Test: {len(X_test):,} przykładów")
print(f"Liczba cech: {X_train.shape[1]} ")

print(f"\n=== KONFIGURACJA MODELU {MODEL_VARIANT} ===")

model_params = {
    'iterations': 3000,
    'learning_rate': 0.08,
    'depth': 5,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,

    'l2_leaf_reg': 15,
    'bagging_temperature': 2,
    'random_strength': 2,

    'subsample': 0.7,

    'min_data_in_leaf': 50,
    'max_leaves': 32,

    'early_stopping_rounds': 100,

    'task_type': 'CPU',
    'thread_count': -1,
    'verbose': 100,
}

print("\nParametry modelu:")
for key, value in model_params.items():
    if key not in ['task_type', 'thread_count', 'verbose']:
        print(f"  {key}: {value}")

print(f"\n=== CROSS-VALIDATION (5-FOLD) ===")

USE_CV = True

if USE_CV:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_rmse = []
    cv_scores_mae = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\nFold {fold + 1}/5...")

        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        model_fold = CatBoostRegressor(**model_params)
        model_fold.set_params(verbose=0)  # Wyłącz verbose dla CV

        model_fold.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            use_best_model=True,
            plot=False
        )

        val_pred = model_fold.predict(X_val)
        fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        fold_mae = mean_absolute_error(y_val, val_pred)

        cv_scores_rmse.append(fold_rmse)
        cv_scores_mae.append(fold_mae)

        print(f"  RMSE: {fold_rmse:,.2f} PLN | MAE: {fold_mae:,.2f} PLN")

    print(f"\n{'=' * 50}")
    print(f"CV RESULTS:")
    print(f"  Mean RMSE: {np.mean(cv_scores_rmse):,.2f} ± {np.std(cv_scores_rmse):,.2f} PLN")
    print(f"  Mean MAE:  {np.mean(cv_scores_mae):,.2f} ± {np.std(cv_scores_mae):,.2f} PLN")
    print(f"{'=' * 50}")
else:
    print("Cross-validation POMINIĘTA (USE_CV=False)")
    cv_scores_rmse = []
    cv_scores_mae = []

print(f"\n=== TRENOWANIE FINALNEGO MODELU ===")

train_pool = Pool(data=X_train, label=y_train)
test_pool = Pool(data=X_test, label=y_test)

model = CatBoostRegressor(**model_params)

model.fit(
    train_pool,
    eval_set=test_pool,
    use_best_model=True,
    plot=False
)

model_path = f"models/catboost_model_{MODEL_VARIANT}_otodom5.cbm"
model.save_model(model_path)
print(f"\n✅ Model zapisany: {model_path}")


print(f"\n=== PREDYKCJA I EWALUACJA ===")

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Metryki Train
train_mae = mean_absolute_error(y_train, train_preds)
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
train_mape = np.mean(np.abs((y_train - train_preds) / y_train)) * 100

# Metryki Test
test_mae = mean_absolute_error(y_test, test_preds)
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
test_mape = np.mean(np.abs((y_test - test_preds) / y_test)) * 100

train_corr = np.corrcoef(y_train, train_preds)[0, 1]
test_corr = np.corrcoef(y_test, test_preds)[0, 1]

print(f"\n{'=' * 60}")
print(f"WYNIKI MODELU {MODEL_VARIANT}")
print(f"{'=' * 60}")

print(f"\n TRAIN SET:")
print(f"  MAE  = {train_mae:>12,.2f} PLN")
print(f"  RMSE = {train_rmse:>12,.2f} PLN")
print(f"  MAPE = {train_mape:>12.2f} %")
print(f"  R²   = {train_corr ** 2:>12.4f}")

print(f"\n TEST SET:")
print(f"  MAE  = {test_mae:>12,.2f} PLN")
print(f"  RMSE = {test_rmse:>12,.2f} PLN")
print(f"  MAPE = {test_mape:>12.2f} %")
print(f"  R²   = {test_corr ** 2:>12.4f}")

# feature importance
feature_importance = model.get_feature_importance(train_pool)
feature_names = X_train.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df.head(20).to_string(index=False))

# Zapisz
importance_df.to_csv(f"models/feature_importance_{MODEL_VARIANT}_otodom5.csv", index=False)
print(f"\n✅ Feature importance zapisane: models/feature_importance_{MODEL_VARIANT}_otodom5.csv")

print(f"\n=== ANALIZA BŁĘDÓW (TEST SET) ===")

errors = np.abs(y_test - test_preds)
relative_errors = (errors / y_test) * 100

print(f"\nBłędy absolutne:")
print(f"  Min:     {errors.min():>10,.2f} PLN")
print(f"  Median:  {np.median(errors):>10,.2f} PLN")
print(f"  Mean:    {errors.mean():>10,.2f} PLN")
print(f"  Max:     {errors.max():>10,.2f} PLN")

