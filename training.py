import xgboost as xgb
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from preprocessing import *

def prepare_data(data, test_size=0.2, random_state=42):
    """Prepare data for training by splitting features and target."""
    y = data['equipment_energy_consumption']
    x = data.drop(['equipment_energy_consumption'], axis=1)
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

def train_xgboost(x_train, y_train):
    """Train XGBoost model with GridSearchCV."""
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    }
    
    xgb_model = xgb.XGBRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=2,
        n_jobs=-1
    )
    
    grid_search.fit(x_train, y_train)
    return grid_search

def create_lstm_dataset(data, target_index, n_steps):
    """Create time series dataset for LSTM."""
    X, Y = [], []
    for i in range(len(data) - n_steps):
        X.append(np.delete(data[i:i+n_steps], target_index, axis=1))
        Y.append(data[i+n_steps, target_index])
    return np.array(X), np.array(Y)

def build_lstm_model(input_shape):
    """Build and compile LSTM model."""
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False, kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(X_train, Y_train, X_test, Y_test, epochs=100, batch_size=32):
    """Train LSTM model."""
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, Y_test),
        callbacks=[early_stop],
        verbose=1
    )
    return model

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model performance and print metrics."""
    print(f'----{model_name}----')
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

def main():
    # Prepare data
    x_train, x_test, y_train, y_test = prepare_data(data)
    
    # Train and evaluate XGBoost
    xgb_model = train_xgboost(x_train, y_train)
    xgb_pred = xgb_model.predict(x_test)
    evaluate_model(y_test, xgb_pred, "XGBoost")
    
    # Prepare and train LSTM
    target_column = 'equipment_energy_consumption'
    target_index = data.columns.get_loc(target_column)
    X, Y = create_lstm_dataset(scaled_data, target_index, n_steps=10)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    
    lstm_model = train_lstm(X_train, Y_train, X_test, Y_test)
    lstm_pred = lstm_model.predict(X_test)
    evaluate_model(Y_test, lstm_pred, "LSTM")

if __name__ == "__main__":
    main()



