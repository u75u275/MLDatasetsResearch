import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# 1. Завантаження інтраденних (<1d) 5-хвилинних даних за вчора для Apple [finance:Apple Inc.]
ticker = "AAPL"
df = yf.download(tickers=ticker, interval="5m", period="1d")

# Приведення індексу колонок до простого вигляду, якщо MultiIndex
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Виділити тільки частину основної торгової сесії (за потребою)
df = df.between_time("09:30", "16:00")

# 2. Формування лагових (відстаючих) ознак
lookback = 12  # 12 * 5 хв = 1 година
for lag in range(1, lookback+1):
    df[f'lag_{lag}'] = df['Close'].shift(lag)

df = df.dropna()

feature_cols = [col for col in df.columns if col.startswith("lag_")]
X = df[feature_cols].copy()
y = df['Close'].shift(-1).dropna()
X = X.iloc[:-1, :]  # Вирівнюємо розміри
y = y.iloc[:X.shape[0]]

# 3. Крос-валідація для оцінки якості моделі
kf = KFold(n_splits=5, shuffle=False)
model = RandomForestRegressor(n_estimators=100, random_state=0)

mae = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
mape = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_percentage_error")
mse = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")

print(f"CV MAE: {abs(mae.mean()):.4f} ± {mae.std():.4f}")
print(f"CV MAPE: {abs(mape.mean()):.4f} ± {mape.std():.4f}")
print(f"CV MSE: {abs(mse.mean()):.4f} ± {mse.std():.4f}")

# 4. Навчання на всіх даних + прогноз для останніх 2 годин (24*5 хв)
hours = 2
n_steps = int(60 / 5 * hours)  # 24 інтервали, якщо 2 години
model.fit(X, y)
X_future = X.tail(n_steps)
future_preds = model.predict(X_future)

# 5. Порівняння з реальними цінами: синхронізуємо довжину
actuals = df['Close'].iloc[-n_steps:]
min_len = min(len(actuals), len(future_preds))
actuals = actuals[-min_len:]
future_preds = future_preds[-min_len:]

comparison_df = pd.DataFrame({
    'Actual': actuals.values,
    'Predicted': future_preds
}, index=actuals.index)

print("\nПорівняння реальних і прогнозованих цін за останні 2 години (5-хв інтервали), USD:")
print(comparison_df)

# 6. Оцінка помилок по цьому відрізку
mae_last = mean_absolute_error(comparison_df['Actual'], comparison_df['Predicted'])
mape_last = mean_absolute_percentage_error(comparison_df['Actual'], comparison_df['Predicted'])
mse_last = mean_squared_error(comparison_df['Actual'], comparison_df['Predicted'])

print(f"\nОцінка якості прогнозу за останні 2 години:")
print(f"MAE: {mae_last:.4f}")
print(f"MAPE: {mape_last:.4f}")
print(f"MSE: {mse_last:.4f}")
