import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def load_csv(file_path):
    """Simple CSV loading (without pandas) - handles non-numeric values."""
    data = []
    with open(file_path, 'r') as f:
        header = f.readline().strip().split(',')
        for line in f:
            row = []
            for idx, val in enumerate(line.strip().split(',')):
                if idx == 0: 
                    row.append(val)
                else:
                    try:
                        row.append(float(val))
                    except ValueError:
                        print(f"Warning: Non-numeric value '{val}' found in column {header[idx]}. Replacing with 0.0.")
                        row.append(0.0)
            data.append(dict(zip(header, row)))
    return data

def min_max_scaling(data, columns):
    """Simplified min-max scaling."""
    for col in columns:
        min_val = min(data[i][col] for i in range(len(data)))
        max_val = max(data[i][col] for i in range(len(data)))
        for i in range(len(data)):
            data[i][col] = (data[i][col] - min_val) / (max_val - min_val)
    return data

def dtw(s, t):
    """Computes the DTW distance between two time series."""
    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n+1, m+1))
    dtw_matrix += np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = sum(abs(x - y) for x, y in zip(s[i-1], t[j-1])) # Sum of absolute differences for all features
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j], 
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )
    return dtw_matrix[n, m]

def dtw_alignment(s, t):
    """
    Computes the DTW alignment path between two time series.
    """
    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n+1, m+1))
    dtw_matrix += np.inf
    dtw_matrix[0, 0] = 0

    # Calculate DTW matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = sum(abs(x - y) for x, y in zip(s[i-1], t[j-1])) # Sum of absolute differences for all features
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j], 
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )

    # Backtrack to find alignment path
    alignment_path = []
    i, j = n, m
    while i > 0 or j > 0:
        alignment_path.append((i, j))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            best_prev = np.argmin([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            if best_prev == 0:
                i -= 1
            elif best_prev == 1:
                j -= 1
            else:
                i -= 1
                j -= 1

    return alignment_path[::-1]
def visualize_dtw_alignment(s, t, alignment_path):
    """
    Visualizes the DTW alignment path between two time series.
    """
    n, m = len(s), len(t)
    plt.figure(figsize=(10, 6))
    plt.plot(s[:, 0], label='Time Series 1', color='blue', marker='o')
    plt.plot(t[:, 0], label='Time Series 2', color='red', marker='x')

    # Plot alignment path
    for i, j in alignment_path:
        plt.plot([i-1, j-1], [s[i-1, 0], t[j-1, 0]], color='green', linestyle='--')

    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title('DTW Alignment Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()

def create_dataset(dataset, look_back=1):
    """Creates dataset for both normal and trend-based KNN."""
    X_trend, X_normal, Y = [], [], []
    for i in range(len(dataset) - look_back - 10):
        a = dataset[i:(i+look_back)]
        X_trend.append(a) # Keep trend data as sequences
        X_normal.append(a[look_back-1]) # Only the last day's features for normal KNN
        Y.append(dataset[i + look_back + 9][3])
    return np.array(X_trend), np.array(X_normal), np.array(Y)

def k_nearest_neighbors_trend(X_train, y_train, X_test, k=5):
    """Predicts using Trend-based KNN with DTW distance."""
    predictions = []
    for test_seq in X_test:
        distances = [dtw(test_seq, train_seq) for train_seq in X_train]
        k_nearest_indices = np.argsort(distances)[:k]  
        prediction = np.mean(y_train[k_nearest_indices])
        predictions.append(prediction)
    return predictions

def rmse(y_true, y_pred):
  """
  Calculates the Root Mean Squared Error (RMSE) between two arrays.
  """
  return np.sqrt(np.mean((y_true - y_pred)**2))

def r2_score(y_true, y_pred):
  """
  Calculates the R-squared score, a measure of how well the regression predictions 
  explain the variance in the true target values.
  """
  tss = np.sum((y_true - np.mean(y_true))**2)
  rss = np.sum((y_true - y_pred)**2)
  return 1 - (rss / tss)

# --- Load and preprocess data ---
file_path = 'stock_price.csv'
data = load_csv(file_path)
features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
data = min_max_scaling(data, features)

# --- Settings ---
look_back = 5
k = 5

# --- Prepare datasets ---
X_trend, X_normal, y = create_dataset(
    [list(row.values())[1:] for row in data], look_back
)
train_size = int(len(X_trend) * 0.80)
X_trend_train, X_trend_test = X_trend[0:train_size], X_trend[train_size:len(X_trend)]
X_normal_train, X_normal_test = X_normal[0:train_size], X_normal[train_size:len(X_normal)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# --- Select two sequences from the test set for visualization ---
test_seq_index = 40 # Index of the first sequence in the test set
test_seq_index2 = 56 # Index of the second sequence in the test set
s = X_trend_test[test_seq_index]
t = X_trend_test[test_seq_index2]
alignment_path = dtw_alignment(s, t)

# --- Visualize the alignment ---
visualize_dtw_alignment(s, t, alignment_path)

# --- Trend-based KNN (using custom DTW) ---
predictions_trend = k_nearest_neighbors_trend(X_trend_train, y_train, X_trend_test, k=k)
rmse_trend = rmse(y_test, predictions_trend)
print(f"Trend-based KNN (Custom DTW) RMSE: {rmse_trend:.4f}")
r2_trend = r2_score(y_test, predictions_trend)
print(f"Trend-based KNN (Custom DTW) R-squared: {r2_trend:.4f}")

# --- Normal KNN ---
model_normal = KNeighborsRegressor(n_neighbors=k, metric='euclidean')  # Using Euclidean distance for normal KNN
model_normal.fit(X_normal_train, y_train)
predictions_normal = model_normal.predict(X_normal_test)
rmse_normal = rmse(y_test, predictions_normal)
print(f"Normal KNN RMSE: {rmse_normal:.4f}")
r2_normal = r2_score(y_test, predictions_normal)
print(f"Normal KNN R-squared: {r2_normal:.4f}")

# --- Pearson Correlation Coefficient Calculation --
lag = 10 # Example: Calculate correlation with the closing price 10 days later

# Shift the closing prices array
shifted_prices = y[lag:]
original_prices = y[:-lag]

# Calculate Pearson correlation
correlation, _ = pearsonr(original_prices, shifted_prices)
print(f"Pearson Correlation Coefficient for a lag of {lag} days: {correlation:.4f}")

# --- Visualization ---
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Stock Prices', color='blue')
plt.plot(predictions_trend, label='Trend-based KNN Predictions (DTW)', color='red', linestyle='--')
plt.plot(predictions_normal, label='Normal KNN Predictions', color='green', linestyle='-.')
plt.xlabel('Time Steps (Test Data)')
plt.ylabel('Normalized Stock Price')
plt.title('Trend-based KNN vs. Normal KNN')
plt.legend()
plt.grid(True)
plt.show()

