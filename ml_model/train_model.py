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

MODEL_VARIANT = "C"
EPOCHS = 250
BATCH_SIZE = 128
TARGET_COL = "Price (in rupees)"


def build_model_C(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
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

train_df = pd.read_csv("train.csv", low_memory=False)
test_df = pd.read_csv("test.csv", low_memory=False)

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]
X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_scaled = np.log1p(y_train.values)
y_test_scaled = np.log1p(y_test.values)

early_stop = EarlyStopping(
    monitor='val_mae',
    patience=25,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_mae',
    factor=0.5,
    patience=10,
    min_lr=0.00001,
    verbose=1
)

model = build_model_C(X_train.shape[1])

history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.10,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

model_path = f"models/mlp_model_{MODEL_VARIANT}_ind.keras"
scaler_path = f"models/scaler_{MODEL_VARIANT}_ind.joblib"

model.save(model_path)
joblib.dump(scaler, scaler_path)

preds_scaled = model.predict(X_test_scaled).ravel()
preds = np.expm1(preds_scaled)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

mask = (y_test > 0) & (preds > 0)
if mask.sum() > 0:
    mape = np.mean(np.abs((y_test[mask] - preds[mask]) / y_test[mask])) * 100
else:
    mape = np.nan


