# v.3

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_absolute_error, mean_squared_error

import os

import numpy as np
import pandas as pd
import requests

API_KEY = os.getenv("FMP_API_KEY", "")
BASE_URL = "https://financialmodelingprep.com/stable"
SYMBOL = "AAPL"

# Спробуй спочатку FULL-варіант із Quickstart:
endpoint = f"{BASE_URL}/historical-price-eod/full"
params = {"symbol": SYMBOL, "apikey": API_KEY}

resp = requests.get(endpoint, params=params, timeout=30)
try:
    resp.raise_for_status()
except requests.HTTPError as e:
    print("HTTPError:", e)
    print("URL:", resp.url)
    print("Raw:", resp.text[:800])
    raise

# Розбір JSON
try:
    payload = resp.json()
except ValueError:
    print("URL:", resp.url)
    print("Bad JSON:", resp.text[:800])
    raise

print("URL:", resp.url)
print("Response type:", type(payload))
#print("----------------payload ",payload)

# --- Універсальний парсер ---
hist = None

if isinstance(payload, dict):
    # Очікуваний формат для деяких historical-ендпоінтів: {"symbol": "...", "historical": [ ... ]}
    if "historical" in payload and isinstance(payload["historical"], list):
        hist = payload["historical"]
    else:
        # Може бути повідомлення про помилку
        emsg = payload.get("Error Message") or payload.get("error")
        if emsg:
            raise ValueError(f"FMP error: {emsg}")
        # Падаємо з діагностикою
        raise ValueError(f"Unexpected dict format: keys={list(payload.keys())}")

elif isinstance(payload, list):
    # Деякі стабільні ендпоінти повертають список записів EOD напряму  --- САМЕ ТАК !
    if len(payload) == 0:
        raise ValueError("Empty list response. Check symbol/apikey/plan.")
    first = payload[0]
    print("---payload[0] ",payload[0])
    if isinstance(first, dict) and {"date", "open", "high", "low", "close"} <= set(first.keys()):
        print("------------")
        hist = payload
    elif isinstance(first, dict) and "historical" in first and isinstance(first["historical"], list):
        hist = first["historical"]
    else:
        print("Sample element:", first)
        raise ValueError("List in unexpected format for historical EOD.")
else:
    raise TypeError(f"Unexpected payload type: {type(payload)}")

# --- DataFrame ---
df = pd.DataFrame(hist)
if "date" not in df.columns:
    print("Columns:", df.columns.tolist())
    raise ValueError("No 'date' column in payload.")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")

#print("---перелік всіх колонок ",df.columns.tolist())

#print(df.to_string())

#pd.set_option("display.max_columns", None)  # показувати всі колонки
#pd.set_option("display.width", 200)         # ширина в символах
#print(df.head())
#print(df.tail())

#block = pd.concat([df.head(5), df.tail(5)])
#print(block.to_string())  # один суцільний блок без скорочень

#print(pd.concat([df.head(), df.tail()]))



# -----------------------------
# Допоміжні фінансові функції
# -----------------------------

def pick_price_column(df: pd.DataFrame) -> str:
    """
    Обираємо колонку ціни для розрахунків:
    пріоритет: close -> adjClose -> vwap -> (high+low)/2 -> (open+close)/2
    """
    candidates = [c for c in df.columns.str.lower()]
    colmap = {c.lower(): c for c in df.columns}  # мапимо назад на оригінальні імена

    if "close" in candidates:
        return colmap["close"]
    if "adjclose" in candidates:
        return colmap["adjclose"]
    if "vwap" in candidates:
        return colmap["vwap"]
    if "high" in candidates and "low" in candidates:
        df["_mid_hl"] = (df[colmap["high"]] + df[colmap["low"]]) / 2.0
        return "_mid_hl"
    if "open" in candidates and "close" in candidates:
        df["_mid_oc"] = (df[colmap["open"]] + df[colmap["close"]]) / 2.0
        return "_mid_oc"
    raise ValueError("Не знайшов підходящої цінової колонки в df.")

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# -----------------------------
# Допоміжні загалні функції
# -----------------------------

def print_full_table(df: pd.DataFrame, width: int = 2000):
    """Друкує DataFrame повністю, з усіма стовпцями і рядками, одним блоком."""
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", width,
        "display.expand_frame_repr", False  # не переносити в кілька панелей
    ):
        print(df.to_string())

def print_X_y(X: pd.DataFrame, y: pd.Series, max_rows: int | None = None):
    """
    Друк X та y з назвами колонок одним блоком.
    Якщо max_rows=None — друкуємо повністю; інакше обмежуємо рядки.
    """
    if max_rows is None:
        dfX = X
        dfY = y.to_frame(name=y.name or "target")
    else:
        dfX = X.head(max_rows)
        dfY = y.head(max_rows).to_frame(name=y.name or "target")
    both = dfX.copy()
    both[dfY.columns[0]] = dfY.values

    YELLOW = "\033[33m"
    RESET = "\033[0m"
    # зробити жовтим назву індексу 'date'
    both.index.name = f"{YELLOW}date{RESET}"

    with pd.option_context("display.max_columns", None, "display.width", 2000, "display.expand_frame_repr", False):
        print(both.to_string())



# -----------------------------
# Побудова фіч і цілі
# -----------------------------

df_feat = df.copy()  # не псуємо оригінал

PRICE_COL = pick_price_column(df_feat)
VOL_COL = next((c for c in ["volume","Volume","VOL","vol"] if c in df_feat.columns), None)

price = df_feat[PRICE_COL].astype(float)

# Доходності та імпульс
df_feat["ret_1d"]  = price.pct_change(1)
df_feat["ret_5d"]  = price.pct_change(5)
df_feat["mom_10"]  = price.pct_change(10)

# Ковзні середні та спреди
df_feat["sma_10"]  = price.rolling(10).mean()
df_feat["sma_30"]  = price.rolling(30).mean()
df_feat["sma_spread_10_30"] = df_feat["sma_10"] / df_feat["sma_30"] - 1

# Волатильність (похідна від щоденних доходностей)
df_feat["vola_10"] = df_feat["ret_1d"].rolling(10).std()
df_feat["vola_20"] = df_feat["ret_1d"].rolling(20).std()

# RSI та MACD
df_feat["rsi_14"] = rsi(price, 14)
macd_line, signal_line, macd_hist = macd(price, 12, 26, 9)
df_feat["macd"] = macd_line
df_feat["macd_signal"] = signal_line
df_feat["macd_hist"] = macd_hist

# Обсяг (якщо є)
if VOL_COL is not None:
    df_feat["vol_ma_10"] = df_feat[VOL_COL].rolling(10).mean()
    df_feat["vol_ma_30"] = df_feat[VOL_COL].rolling(30).mean()
    df_feat["vol_spike"] = (df_feat[VOL_COL] / (df_feat["vol_ma_10"] + 1e-9))  # spike > 1 => аномально високий обсяг

# Ціль: чи зросте ціна завтра (1/0)
df_feat["target_up"] = (price.shift(-1) > price).astype(int)

# Остаточний набір фіч
feature_cols = [
    "ret_1d","ret_5d","mom_10",
    "sma_10","sma_30","sma_spread_10_30",
    "vola_10","vola_20",
    "rsi_14","macd","macd_signal","macd_hist"
]
if VOL_COL is not None:
    feature_cols += ["vol_ma_10","vol_ma_30","vol_spike"]

dataset = df_feat.dropna(subset=feature_cols + ["target_up"]).copy()
X = dataset[feature_cols]
y = dataset["target_up"]

print(f"Розмір навчальної вибірки: {X.shape}")
print("------------")

# 1) Надрукувати X з усіма назвами стовпців
# print("=== Матриця X (features) ===")
# print_full_table(X)

# 2) Надрукувати Y як одноколонкову таблицю з назвою стовпця
# y_df = y.to_frame(name="target_up")   # перетворюємо Series -> DataFrame, даємо заголовок
# print("\n=== Вектор y (target_up) ===")
# print_full_table(y_df)


# print_X_y(X, y, max_rows=10)  # або None для повного виводу


# Тренування базової моделі (Random Forest) з time‑series CV

tscv = TimeSeriesSplit(n_splits=5)
accs, aucs, f1s = [], [], []

for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    y_pred = (p >= 0.5).astype(int)

    accs.append(accuracy_score(y_te, y_pred))
    aucs.append(roc_auc_score(y_te, p))
    f1s.append(f1_score(y_te, y_pred))

print(f"CV Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
print(f"CV ROC-AUC:  {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
print(f"CV F1:       {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

# Фінальна модель на всіх даних (для прогнозу «на завтра»)
final_model = RandomForestClassifier(
    n_estimators=600,
    max_depth=6,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
final_model.fit(X, y)

#---------------------------------------------
# Вибираємо останній рядок із фічами
#---------------------------------------------
last_row = df_feat.iloc[[-1]][feature_cols]

# Перевіримо: чи немає NaN
if last_row.isna().any().any():
    print("⚠ Останній ряд має пропуски у фічах – модель не може зробити прогноз.")
else:
    #---------------------------------------------
    # 2. Проганяємо через final_model
    #---------------------------------------------
    prob_up = final_model.predict_proba(last_row)[0][1]
    pred_class = "up" if prob_up >= 0.5 else "down"

    print(f"Ймовірність, що завтра ціна AAPL зросте: {prob_up:.4f}")
    print(f"Класифікація моделі: {pred_class.upper()}")

