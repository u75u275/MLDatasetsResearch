"""
Intra-day прогноз ціни AAPL з використанням yfinance + RandomForestRegressor.

Тренування:
    - intraday 5m дані за останніх 7 днів (включно з учора), робочі години 09:30–16:00.
    - лаги: 12 * 5 хв (1 година історії)
    - таргет: ціна через 1 годину (12 кроків)

Тест:
    - сьогоднішній день, останні 2 години (24 точки)

Метрики:
    - CV MAE, CV MAPE, CV MSE — крос-валідація на тижневому train
    - Holdout MAE/MAPE/MSE — на сьогоднішньому 2h вікні
"""

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer


# ----------------------------------------------------------
# 1. Завантаження intraday даних за останні 10 днів
# ----------------------------------------------------------

def download_intraday_5m_last_10_days(ticker="AAPL"):
    """
    Завантажує близько 10 днів 5-хвилинних свічок.
    Повертає DataFrame з OHLCV, tz=America/New_York.
    """

    data = yf.download(
        tickers=ticker,
        period="10d",
        interval="5m",
        auto_adjust=True,
        prepost=False,
        progress=False,
        multi_level_index=False,
    )

    if data.empty:
        raise ValueError("Не вдалося завантажити intraday-дані з Yahoo Finance.")

    # Перевести в час Нью-Йорка
    if data.index.tz is None:
        data.index = data.index.tz_localize("America/New_York")
    else:
        data = data.tz_convert("America/New_York")

    return data


# ----------------------------------------------------------
# 2. Лагові ознаки
# ----------------------------------------------------------

def make_lag_features_from_close(df, price_col="Close", max_lag=12, horizon_steps=12):
    df = df.copy()
    series = df[price_col]

    for lag in range(1, max_lag + 1):
        df[f"lag_{lag}"] = series.shift(lag)

    df["target"] = series.shift(-horizon_steps)
    df = df.dropna()

    features = [f"lag_{lag}" for lag in range(1, max_lag + 1)]
    X = df[features]
    y = df["target"]

    return X, y


# ----------------------------------------------------------
# 3. MAPE
# ----------------------------------------------------------

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0


# ----------------------------------------------------------
# 4. Крос-валідація
# ----------------------------------------------------------

def run_cross_validation(X_train, y_train, n_splits=5, random_state=42):

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # MAE
    mae_scores = -cross_val_score(
        model, X_train, y_train, cv=tscv,
        scoring="neg_mean_absolute_error", n_jobs=-1
    )

    # MAPE
    mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    mape_scores = -cross_val_score(
        model, X_train, y_train, cv=tscv,
        scoring=mape_scorer, n_jobs=-1
    )

    # MSE
    mse_scores = -cross_val_score(
        model, X_train, y_train, cv=tscv,
        scoring="neg_mean_squared_error", n_jobs=-1
    )

    return {"MAE": mae_scores, "MAPE": mape_scores, "MSE": mse_scores}


def print_cv_summary(metrics):
    for name in ["MAE", "MAPE", "MSE"]:
        mean_val = metrics[name].mean()
        std_val = metrics[name].std(ddof=1)
        if name == "MAPE":
            print(f"CV {name}: {mean_val:.5f}% ± {std_val:.5f}%")
        else:
            print(f"CV {name}: {mean_val:.5f} ± {std_val:.5f}")


# ----------------------------------------------------------
# 5. Основна логіка
# ----------------------------------------------------------

def main():
    TICKER = "AAPL"
    MAX_LAG = 12
    HORIZON_STEPS = 12
    TEST_WINDOW = 24  # 2 години

    print("Завантажую останні 10 днів intraday-даних...")
    df = download_intraday_5m_last_10_days(TICKER)

    # Дати
    now_ny = pd.Timestamp.now(tz="America/New_York")
    today = now_ny.date()
    yesterday = (now_ny - pd.Timedelta(days=1)).date()
    train_start = yesterday - pd.Timedelta(days=7)

    print(f"\nTrain-вікно: {train_start} → {yesterday}")
    print(f"Test-вікно: сьогодні ({today})")

    # Фільтруємо робочі години
    df = df.between_time("09:30", "16:00")

    # Тренувальний період: останні 7 днів включно з учорашнім
    mask_train = (df.index.date >= train_start) & (df.index.date <= yesterday)
    df_train = df.loc[mask_train].copy()

    # Тестовий період: сьогоднішні дані
    mask_test = (df.index.date == today)
    df_today = df.loc[mask_test].copy()

    print(f"Кількість рядків train: {len(df_train)}")
    print(f"Кількість рядків today: {len(df_today)}")

    # Перевірка даних
    if len(df_train) < MAX_LAG + HORIZON_STEPS + 30:
        raise ValueError("Недостатньо даних для тижневого тренування.")

    if len(df_today) < MAX_LAG + HORIZON_STEPS + TEST_WINDOW:
        raise ValueError("Недостатньо сьогоднішніх даних для тестових 2 годин.")

    # Лагові дані
    X_train, y_train = make_lag_features_from_close(df_train, "Close", MAX_LAG, HORIZON_STEPS)
    X_today, y_today = make_lag_features_from_close(df_today, "Close", MAX_LAG, HORIZON_STEPS)

    # Test-вікно = останні 24 точки
    X_test = X_today.tail(TEST_WINDOW)
    y_test = y_today.tail(TEST_WINDOW)

    # -----------------------------
    # Крос-валідація (train 7 днів)
    # -----------------------------
    print("\n=== CV на тижневому train-вікні ===")
    cv_metrics = run_cross_validation(X_train, y_train)
    print_cv_summary(cv_metrics)

    # -----------------------------
    # Навчання фінальної моделі
    # -----------------------------
    model = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # -----------------------------
    # Holdout на сьогодні
    # -----------------------------
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print("\n=== Тест на сьогоднішніх 2 годинах ===")
    print(f"Holdout MAE:  {mae:.5f}")
    print(f"Holdout MAPE: {mape:.5f}%")
    print(f"Holdout MSE:  {mse:.5f}")

    # Таблиця порівнянь
    results = pd.DataFrame({
        "time_t": X_test.index,
        "real_price_t_plus_1h": y_test.values,
        "predicted_t_plus_1h": y_pred,
    }).set_index("time_t")

    print("\n=== Порівняння прогнозів за сьогодні ===")
    print(results.head(12))
    print("...")
    print(results.tail(12))


if __name__ == "__main__":
    main()
