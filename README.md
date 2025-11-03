# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 3-11-2025

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import warnings

warnings.filterwarnings('ignore')

# Custom float converter for comma decimal separator
def custom_float(x):
    try:
        return float(x.replace(',', '.'))
    except Exception:
        return None

# Step 1: Read CSV and correct decimal issues
df = pd.read_csv('Gold_price_2025-new.csv')
price_cols = [c for c in df.columns if 'price' in c.lower()]
for c in price_cols:
    df[c] = df[c].astype(str).apply(custom_float)

# Use 'priceClose' if possible
series = df['priceClose']

# Step 2: Display shape and first 20 rows
print("Shape:", df.shape)
print("First 20 rows:\n", df.head(20))

# Step 3: Set figure size for plots (see later)

# Step 4: Plot the first 50 values of 'priceClose'
plt.figure(figsize=(12, 6))
plt.plot(series[:50])
plt.title('First 50 Values of priceClose')
plt.xlabel('Index')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# Step 5: Rolling average with window size 5
rolling_5 = series.rolling(window=5).mean()
print("First 10 values of rolling mean (window=5):\n", rolling_5.head(10))

# Step 6: Rolling average with window size 10
rolling_10 = series.rolling(window=10).mean()

# Step 7: Plot original and rolling average (window=10)
plt.figure(figsize=(12, 6))
plt.plot(series, label='Original')
plt.plot(rolling_10, label='10-period Rolling Mean', linestyle='--')
plt.title('Original Data and 10-period Rolling Mean')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Step 8: Exponential smoothing and plot
model = SimpleExpSmoothing(series, initialization_method="legacy-heuristic")
fitted_vals = model.fit(smoothing_level=0.2, optimized=False).fittedvalues

plt.figure(figsize=(12, 6))
plt.plot(series, label='Original')
plt.plot(fitted_vals, label='Exponential Smoothing (alpha=0.2)', linestyle='--')
plt.title('Original Data and Exponential Smoothing')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

```
### OUTPUT:

<img width="979" height="528" alt="image" src="https://github.com/user-attachments/assets/3e986651-3b89-403b-9568-f6ade29b165e" />

<img width="956" height="391" alt="image" src="https://github.com/user-attachments/assets/4e91ffa6-85ee-4ceb-a2b8-427be725e17a" />

<img width="950" height="381" alt="image" src="https://github.com/user-attachments/assets/556e2ad5-c479-46f4-b39b-bd1104025ebe" />

<img width="957" height="534" alt="image" src="https://github.com/user-attachments/assets/5226d8fc-9b16-4ee9-85ca-42e8abc35e34" />

### RESULT:
Thus the program run successfully based on the SARIMA model.
