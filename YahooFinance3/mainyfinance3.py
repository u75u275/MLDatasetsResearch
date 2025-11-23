"""
Intra-day прогноз ціни AAPL з використанням yfinance + RandomForestRegressor.

Тренування:
    - intraday 5m дані за останній тиждень (від учора назад на 7 днів),
      робочі години 09:30–16:00.
    - лаги: 12 * 5 хв (1 година історії)
    - таргет: ціна через 1 годину (12 кроків)

Тест:
    - сьогоднішні дані, останні 2 години (24 точки)

Фічі:
    - лаги ціни
    - EMA(10), EMA(20)
    - RSI(14)
    - MACD, MACD_signal

Метрики:
    - CV MAE, CV MAPE, CV MSE (TimeSeriesSplit + cross_val_score) на train
    - Holdout MAE/MAPE/MSE на сьогоднішньому 2-годинному вікні

Візуалізація:
    - графік реальної та прогнозованої ціни t+1h на сьогоднішньому вікні
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer


# ----------------------------------------------------------
# 1. Завантаження intraday даних
# ----------------------------------------------------------

def download_intraday_5m_last_10_days(ticker="AAPL"):
    """
    Завантажує близько 10 днів 5-хвилинних свічок з Yahoo Finance.
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
# 2. Технічні індикатори: EMA, RSI, MACD
# ----------------------------------------------------------

def add_ema(df, period=14):
    df[f"EMA_{period}"] = df["Close"].ewm(span=period, adjust=False).mean()
    return df

def add_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period).mean()
    rs = gain / (loss + 1e-8)
    df[f"RSI_{period}"] = 100 - (100 / (1 + rs))
    return df

def add_macd(df):
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df

def add_technical_features(df):
    df = add_ema(df, 10)
    df = add_ema(df, 20)
    df = add_rsi(df, 14)
    df = add_macd(df)
    df = df.dropna()
    return df


# ----------------------------------------------------------
# 3. Побудова фіч: індикатори + лаги + таргет
# ----------------------------------------------------------

def make_features(df, max_lag=12, horizon=12):
    """
    df      : DataFrame з колонками включно з 'Close'
    max_lag : скільки лагів (кроків по 5 хв) брати
    horizon : на скільки кроків уперед прогнозуємо (12 = 1 година)

    Повертає:
        X: матриця фіч (EMA, RSI, MACD, лаги)
        y: вектор таргету (ціна Close в t + horizon)
    """
    df = df.copy()

    # Технічні індикатори
    df = add_technical_features(df)

    # Лагові ознаки по Close
    for lag in range(1, max_lag + 1):
        df[f"lag_{lag}"] = df["Close"].shift(lag)

    # Таргет: ціна через horizon кроків
    df["target"] = df["Close"].shift(-horizon)

    df = df.dropna()

    feature_cols = (
        [c for c in df.columns if c.startswith("EMA_") or c.startswith("RSI_") or c.startswith("MACD")]
        + [f"lag_{i}" for i in range(1, max_lag + 1)]
    )

    X = df[feature_cols]
    y = df["target"]

    return X, y


# ----------------------------------------------------------
# 4. MAPE
# ----------------------------------------------------------

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0


# ----------------------------------------------------------
# 5. Крос-валідація
# ----------------------------------------------------------

def run_cross_validation(X_train, y_train, n_splits=5, random_state=42):

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
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
        scores = metrics[name]
        mean_val = scores.mean()
        std_val = scores.std(ddof=1)
        if name == "MAPE":
            print(f"CV {name}: {mean_val:.5f}% ± {std_val:.5f}%")
        else:
            print(f"CV {name}: {mean_val:.5f} ± {std_val:.5f}")


# ----------------------------------------------------------
# 6. Візуалізація: реальна vs прогнозована ціна
# ----------------------------------------------------------

def plot_real_vs_pred(time_index, real, pred, title="Прогноз ціни AAPL (t+1h)"):
    plt.figure(figsize=(12, 6))
    plt.plot(time_index, real, label="Реальна ціна t+1h", linewidth=2)
    plt.plot(time_index, pred, label="Прогноз моделі", linewidth=2)
    plt.xlabel("Час")
    plt.ylabel("Ціна, USD")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# 7. Основна логіка
# ----------------------------------------------------------

def main():
    TICKER = "AAPL"
    MAX_LAG = 12
    HORIZON = 12          # 12 * 5 хв = 1 година
    TEST_WINDOW = 24      # 24 * 5 хв = 2 години

    print("Завантажую останні 10 днів intraday-даних...")
    df = download_intraday_5m_last_10_days(TICKER)

    # Фільтруємо по торговим годинам
    df = df.between_time("09:30", "16:00")

    now_ny = pd.Timestamp.now(tz="America/New_York")
    today = now_ny.date()
    yesterday = (now_ny - pd.Timedelta(days=1)).date()
    train_start = yesterday - pd.Timedelta(days=7)

    print(f"\nTrain-вікно: {train_start} → {yesterday}")
    print(f"Test-вікно: сьогодні ({today})")

    # Тренувальний період: тиждень до вчора включно
    mask_train = (df.index.date >= train_start) & (df.index.date <= yesterday)
    df_train = df.loc[mask_train].copy()

    # Тест: сьогоднішній день
    mask_test = (df.index.date == today)
    df_today = df.loc[mask_test].copy()

    print(f"Кількість рядків train (сирі): {len(df_train)}")
    print(f"Кількість рядків today (сирі): {len(df_today)}")

    if len(df_train) < MAX_LAG + HORIZON + 30:
        raise ValueError("Недостатньо даних для тижневого train-вікна.")

    if len(df_today) < MAX_LAG + HORIZON + TEST_WINDOW:
        raise ValueError("Недостатньо сьогоднішніх даних для тестових 2 годин.")

    # --- Фічі для train і test ---
    X_train, y_train = make_features(df_train, max_lag=MAX_LAG, horizon=HORIZON)
    X_today, y_today = make_features(df_today, max_lag=MAX_LAG, horizon=HORIZON)

    print("\nФорма X_train:", X_train.shape)
    print("Форма y_train:", y_train.shape)
    print("Форма X_today:", X_today.shape)
    print("Форма y_today:", y_today.shape)

    if len(X_today) < TEST_WINDOW:
        raise ValueError("Після побудови фіч сьогодні залишилось менше 24 точок.")

    # Тестове вікно: останні 2 години
    X_test = X_today.tail(TEST_WINDOW)
    y_test = y_today.tail(TEST_WINDOW)

    print(f"\nРозмір тестового вікна (сьогодні, 2 години): {X_test.shape[0]} точок")
    print(f"Початок тестового вікна (t):  {X_test.index[0]}")
    print(f"Кінець тестового вікна (t):   {X_test.index[-1]}")

    # -----------------------------
    # Крос-валідація на train
    # -----------------------------
    print("\n=== CV на тижневому train-вікні (з EMA, RSI, MACD) ===")
    cv_metrics = run_cross_validation(X_train, y_train)
    print_cv_summary(cv_metrics)

    # -----------------------------
    # Навчання фінальної моделі
    # -----------------------------
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # -----------------------------
    # Holdout на сьогоднішніх 2 годинах
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

    # -----------------------------
    # Графік реальної та прогнозної ціни
    # -----------------------------
    plot_real_vs_pred(
        time_index=X_test.index,
        real=y_test.values,
        pred=y_pred,
        title="AAPL: прогноз ціни через 1 годину (EMA, RSI, MACD + лаги)"
    )


if __name__ == "__main__":
    main()
