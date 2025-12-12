import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
from catboost import CatBoostRegressor, Pool
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_VARIANT = "CATBOOST"
TARGET_COL = "Price (in rupees)"

os.makedirs("models", exist_ok=True)

print("=== WCZYTYWANIE DANYCH ===")
train_df = pd.read_csv("train.csv", low_memory=False)
test_df = pd.read_csv("test.csv", low_memory=False)

#CZYSZCZENIE OUTLIERÓW
print(f"\n=== CZYSZCZENIE OUTLIERÓW ===")
print(f"Przed czyszczeniem - train: {len(train_df):,}, test: {len(test_df):,}")

train_df = train_df[train_df[TARGET_COL] > 0].copy()
test_df = test_df[test_df[TARGET_COL] > 0].copy()

price_threshold_train = train_df[TARGET_COL].quantile(0.995)
price_threshold_test = test_df[TARGET_COL].quantile(0.995)

print(f"Próg ceny (99.5 percentyl) - train: {price_threshold_train:,.0f} PLN, test: {price_threshold_test:,.0f} PLN")

train_df = train_df[train_df[TARGET_COL] <= price_threshold_train].copy()
test_df = test_df[test_df[TARGET_COL] <= price_threshold_test].copy()

min_price_threshold = train_df[TARGET_COL].quantile(0.001)
train_df = train_df[train_df[TARGET_COL] >= min_price_threshold].copy()
test_df = test_df[test_df[TARGET_COL] >= min_price_threshold].copy()

print(f"Po czyszczeniu - train: {len(train_df):,}, test: {len(test_df):,}")

print(f"\n=== TRANSFORMACJA CENY ===")
train_df['log_price'] = np.log1p(train_df[TARGET_COL])
test_df['log_price'] = np.log1p(test_df[TARGET_COL])

X_train = train_df.drop(columns=[TARGET_COL, 'log_price'])
y_train = train_df['log_price']
y_train_original = train_df[TARGET_COL]

X_test = test_df.drop(columns=[TARGET_COL, 'log_price'])
y_test = test_df['log_price']
y_test_original = test_df[TARGET_COL]

print(f"\n=== INFORMACJE O DANYCH ===")
print(f"Rozmiar train: {len(X_train):,}")
print(f"Rozmiar test: {len(X_test):,}")
print(f"Liczba cech: {X_train.shape[1]}")
print(f"Zakres cen treningowych: {y_train_original.min():,.0f} - {y_train_original.max():,.0f} PLN")
print(f"Mediana cen: {y_train_original.median():,.0f} PLN")
print(f"Średnia cen: {y_train_original.mean():,.0f} PLN")
print(f"Zakres log(cena) train: {y_train.min():.2f} - {y_train.max():.2f}")

# Identyfikacja cech kategorycznych (jeśli istnieją)
categorical_features = []
for col in X_train.columns:
    if X_train[col].dtype == 'object' or X_train[col].nunique() < 20:
        if col not in ['Carpet Area', 'Super Area', 'TotalFloors', 'CurrentFloor',
                       'Bathroom', 'Balcony', 'Bedroom', 'Car Parking']:
            categorical_features.append(col)

if categorical_features:
    print(f"\nCechy kategoryczne ({len(categorical_features)}): {categorical_features[:5]}...")

train_pool = Pool(
    data=X_train,
    label=y_train,
    cat_features=categorical_features if categorical_features else None
)

test_pool = Pool(
    data=X_test,
    label=y_test,
    cat_features=categorical_features if categorical_features else None
)

print(f"\n=== KONFIGURACJA MODELU {MODEL_VARIANT} ===")

model = CatBoostRegressor(
    iterations=4000,
    learning_rate=0.025,
    depth=6,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,

    # Regularyzacja
    l2_leaf_reg=12,
    bagging_temperature=0.8,

    min_data_in_leaf=40,

    subsample=0.8,

    early_stopping_rounds=200,

    border_count=254,

    task_type='CPU',
    thread_count=-1,

    verbose=100,
)

print(f"Liczba iteracji: {model.get_params()['iterations']}")
print(f"Learning rate: {model.get_params()['learning_rate']}")
print(f"Głębokość drzew: {model.get_params()['depth']}")
print(f"Loss function: {model.get_params()['loss_function']}")
print(f"Subsample: {model.get_params()['subsample']}")
print(f"L2 regularization: {model.get_params()['l2_leaf_reg']}")
print(f"Min data in leaf: {model.get_params()['min_data_in_leaf']}")

print(f"\n=== TRENOWANIE MODELU {MODEL_VARIANT} ===")

model.fit(
    train_pool,
    eval_set=test_pool,
    use_best_model=True,
    plot=False
)

model_path = f"models/catboost_model_{MODEL_VARIANT}_ind3.cbm"
model.save_model(model_path)

print(f"\n✅ Saved model to {model_path}")

print(f"\n=== PREDYKCJA ===")

# Predykcja w skali log
train_preds_log = model.predict(X_train)
test_preds_log = model.predict(X_test)

train_preds = np.expm1(train_preds_log)
test_preds = np.expm1(test_preds_log)

train_preds = np.maximum(train_preds, 0)
test_preds = np.maximum(test_preds, 0)

train_mae = mean_absolute_error(y_train_original, train_preds)
train_rmse = np.sqrt(mean_squared_error(y_train_original, train_preds))
train_r2 = r2_score(y_train_original, train_preds)

test_mae = mean_absolute_error(y_test_original, test_preds)
test_rmse = np.sqrt(mean_squared_error(y_test_original, test_preds))
test_r2 = r2_score(y_test_original, test_preds)


def calculate_mape(y_true, y_pred):
    mask = (y_true > 0) & (y_pred > 0)
    if mask.sum() > 0:
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return np.nan


train_mape = calculate_mape(y_train_original, train_preds)
test_mape = calculate_mape(y_test_original, test_preds)

train_medae = np.median(np.abs(y_train_original - train_preds))
test_medae = np.median(np.abs(y_test_original - test_preds))

print(f"\n=== WYNIKI MODELU {MODEL_VARIANT} ===")
print(f"\n📊 TRAIN SET:")
print(f"MAE      = {train_mae:,.2f} PLN")
print(f"MedAE    = {train_medae:,.2f} PLN")
print(f"RMSE     = {train_rmse:,.2f} PLN")
print(f"R²       = {train_r2:.4f}")
print(f"MAPE     = {train_mape:.2f}%")

print(f"\n📊 TEST SET:")
print(f"MAE      = {test_mae:,.2f} PLN")
print(f"MedAE    = {test_medae:,.2f} PLN")
print(f"RMSE     = {test_rmse:,.2f} PLN")
print(f"R²       = {test_r2:.4f}")
print(f"MAPE     = {test_mape:.2f}%")

# Feature importance
feature_importance = model.get_feature_importance(train_pool)
feature_names = X_train.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

importance_df.to_csv(f"models/feature_importance_{MODEL_VARIANT}_ind3.csv", index=False)
print(f"\n✅ Feature importance zapisane do models/feature_importance_{MODEL_VARIANT}_ind3.csv")

