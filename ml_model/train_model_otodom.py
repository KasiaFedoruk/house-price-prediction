import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import matplotlib.pyplot as plt

MODEL_VARIANT = "B"
EPOCHS = 300
BATCH_SIZE = 32
TARGET_COL = "price"


def build_model_B(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.35),
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.Huber(delta=1.0),
        metrics=['mae']
    )
    return model


def build_model_C(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.35),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.Huber(delta=1.0),
        metrics=['mae']
    )
    return model


os.makedirs("models", exist_ok=True)

train_df = pd.read_csv("train_otodom.csv", low_memory=False)
test_df = pd.read_csv("test_otodom.csv", low_memory=False)

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]
X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

USE_LOG_TRANSFORM = True

if USE_LOG_TRANSFORM:
    y_train_scaled = np.log1p(y_train.values)
    y_test_scaled = np.log1p(y_test.values)
    price_scaler = None
else:
    price_scaler = StandardScaler()
    y_train_scaled = price_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = price_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

early_stop = EarlyStopping(
    monitor='val_mae',
    patience=40,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_mae',
    factor=0.5,
    patience=12,
    min_lr=0.00001,
    verbose=1
)

model = build_model_B(X_train.shape[1])

history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

model_path = f"models/mlp_model_{MODEL_VARIANT}_1_otodom.keras"
scaler_path = f"models/scaler_{MODEL_VARIANT}_1_otodom.joblib"
price_scaler_path = f"models/price_scaler_{MODEL_VARIANT}_1_otodom.joblib"

model.save(model_path)
joblib.dump(scaler, scaler_path)

if price_scaler is not None:
    joblib.dump(price_scaler, price_scaler_path)

preds_scaled = model.predict(X_test_scaled).ravel()

if USE_LOG_TRANSFORM:
    preds = np.expm1(preds_scaled)
else:
    preds = price_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

mask = (y_test > 0) & (preds > 0)
if mask.sum() > 0:
    mape = np.mean(np.abs((y_test[mask] - preds[mask]) / y_test[mask])) * 100
else:
    mape = np.nan


