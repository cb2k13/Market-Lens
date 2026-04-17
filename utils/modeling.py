import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def prepare_ml_data(df: pd.DataFrame) -> pd.DataFrame:
    
    data = df.copy()

    if "Close" not in data.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")

    # Lagged returns
    data["Return_Lag_1"] = data["Daily Return"].shift(1)
    data["Return_Lag_2"] = data["Daily Return"].shift(2)
    data["Return_Lag_3"] = data["Daily Return"].shift(3)

    # Price distance from moving averages
    data["Close_vs_MA20"] = (data["Close"] - data["MA20"]) / data["MA20"]
    data["Close_vs_MA50"] = (data["Close"] - data["MA50"]) / data["MA50"]

    # Momentum
    data["Momentum_5"] = data["Close"].pct_change(5)
    data["Momentum_10"] = data["Close"].pct_change(10)

    # Prediction target is the next day's return
    data["Target_Next_Return"] = data["Daily Return"].shift(-1)

    data = data.dropna().copy()
    return data


def train_regression_model(data: pd.DataFrame):
    """
    Train a linear regression model on time-series data.
    Returns model, predictions df, metrics, and next-row prediction.
    """
    feature_cols = [
        "Daily Return",
        "Return_Lag_1",
        "Return_Lag_2",
        "Return_Lag_3",
        "MA20",
        "MA50",
        "Volatility20",
        "RSI14",
        "Close_vs_MA20",
        "Close_vs_MA50",
        "Momentum_5",
        "Momentum_10",
    ]

    missing = [col for col in feature_cols + ["Target_Next_Return"] if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = data[feature_cols]
    y = data["Target_Next_Return"]

    if len(data) < 40:
        raise ValueError("Not enough data to train the model. Try a longer time range like 1y or 2y.")

    split_idx = int(len(data) * 0.8)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Direction accuracy on whether predicted sign matches actual sign
    actual_direction = np.sign(y_test)
    predicted_direction = np.sign(y_pred)
    direction_accuracy = (actual_direction == predicted_direction).mean()

    results_df = pd.DataFrame({
        "Date": data.iloc[split_idx:]["Date"].values,
        "Actual Return": y_test.values,
        "Predicted Return": y_pred,
    })

    # Predict next day using latest available row
    latest_features = X.iloc[[-1]]
    next_day_prediction = model.predict(latest_features)[0]

    metrics = {
        "rmse": rmse,
        "direction_accuracy": direction_accuracy,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_cols": feature_cols,
    }

    return model, results_df, metrics, next_day_prediction