import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt


# =========================
# 1. 使用 yfinance 抓取股票資料
# =========================
ticker = "0050.TW"  # 元大台灣 50 ETF
data = yf.download(
    ticker,
    start="2020-01-01",
    end="2025-01-01",
    progress=True
)

# 使用收盤價
prices = data["Close"]

# =========================
# 2. 特徵工程
# =========================
N = 5  # 使用過去 N 天的資料預測第 N+1 天

# 計算每日報酬率
returns = prices.pct_change()

# 建立特徵資料表
df = pd.DataFrame()
for i in range(1, N + 1):
    df[f"return_t-{i}"] = returns.shift(i)

# 標籤：下一天上漲=1，否則=0
df["target"] = (returns.shift(-1) > 0).astype(int)
df.dropna(inplace=True)

# =========================
# 3. 切分訓練資料與測試資料
# =========================
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# =========================
# 4. 資料標準化
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 5. 建立分類模型(應用邏輯迴歸模型)
# =========================
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# =========================
# 6. Confusion Matrix
# =========================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix (0050.TW)")
plt.show()

# =========================
# 7. Classification Report 之呈現
# =========================
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
