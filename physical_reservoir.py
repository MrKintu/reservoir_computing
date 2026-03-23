# %% [markdown]
# # Physical Reservoir Simulation for stock time series (software model)
# This notebook simulates a simple physical-like reservoir (nonlinear dynamical system with delay),
# trains a linear readout, evaluates, and saves the model. This is a software simulation of a physical reservoir.

# %% [markdown]
# Import libraries

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from data_utils import load_close_matrix, train_val_test_split, windowed_dataset, select_random_ticker
from logger import get_logger

MODELS_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Initialize logger
logger = get_logger("physical_reservoir")

# %% [markdown]
# 1) Load data

# %%
# Load all ticker data
df = load_close_matrix()
if df.empty:
    logger.error("No ticker CSVs found in ./tickers. Run ticker.py first.")
    raise RuntimeError("No ticker CSVs found in ./tickers. Run ticker.py first.")

# Select a random ticker (reproducible with seed=42)
series = select_random_ticker(df, random_state=5000)
logger.info(f"Using randomly selected ticker: {series.name}, length: {len(series)}")
print(f"Using randomly selected ticker: {series.name}, length: {len(series)}")

# %% [markdown]
# 2) Physical reservoir simulation (nonlinear delay reservoir)

# %%
class DelayReservoir:
    def __init__(self, n_neurons=1000, input_gain=0.5, feedback=0.9, nonlin=np.tanh, random_state=42):
        self.logger = get_logger("physical_reservoir")
        self.n_neurons = n_neurons  # virtual nodes
        self.input_gain = input_gain
        self.feedback = feedback
        self.nonlin = nonlin
        self.rs = np.random.Generator(np.random.PCG64(random_state))
        # mask to map input to virtual nodes
        self.mask = (self.rs.random(self.n_neurons) - 0.5) * 2.0
        self.state = np.zeros(self.n_neurons)
        self.logger.debug(f"Initialized DelayReservoir with n_neurons={n_neurons}, input_gain={input_gain}, feedback={feedback}")

    def run(self, inputs):
        # inputs: (T, 1)
        T = inputs.shape[0]
        self.logger.debug(f"Running DelayReservoir for {T} timesteps")
        states = np.zeros((T, self.n_neurons))
        for t in range(T):
            u = inputs[t, 0]
            # single scalar node update with virtual node mask
            x = self.nonlin(self.feedback * self.state + self.input_gain * self.mask * u)
            self.state = x
            states[t] = self.state
        self.logger.debug(f"DelayReservoir run completed, states shape: {states.shape}")
        return states

# %% [markdown]
# 3) Prepare windows and normalize

# %%
train, val, test = train_val_test_split(series, train_frac=0.7, val_frac=0.15)
window_size = 10
logger.info(f"Data split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
logger.info(f"Using window size: {window_size}")
X_train, y_train = windowed_dataset(train, window_size)
X_val, y_val = windowed_dataset(np.concatenate([train[-window_size:], val]), window_size)
X_test, y_test = windowed_dataset(np.concatenate([val[-window_size:], test]), window_size)

mu = X_train.mean()
sigma = X_train.std() + 1e-9
X_train_n = (X_train - mu) / sigma
X_val_n = (X_val - mu) / sigma
X_test_n = (X_test - mu) / sigma
logger.debug(f"Data normalized - mu: {mu:.6f}, sigma: {sigma:.6f}")

# %% [markdown]
# 4) Run delay reservoir and train readout

# %%
dr = DelayReservoir(n_neurons=300, input_gain=0.6, feedback=0.95)
logger.info("Created DelayReservoir with 300 virtual nodes")
def windows_to_inputs(X):
    return X.reshape(X.shape[0], X.shape[1], 1)

def delay_states_for_windows(dr, x_n):
    feats = []
    for w in x_n:
        seq = w.reshape(len(w), 1)
        states = dr.run(seq)
        feats.append(states[-1])  # last virtual node states
    return np.vstack(feats)

R_train = delay_states_for_windows(dr, X_train_n)
R_val = delay_states_for_windows(dr, X_val_n)
R_test = delay_states_for_windows(dr, X_test_n)

ridge = Ridge(alpha=1.0, random_state=42)
logger.info("Training ridge regression readout...")
ridge.fit(R_train, y_train)
y_pred = ridge.predict(R_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 1) Baseline Comparison - Naive Last-Value
naive_pred = y_test[:-1]
naive_mse = mean_squared_error(y_test[1:], naive_pred)
naive_mae = mean_absolute_error(y_test[1:], naive_pred)
improvement_mse = ((naive_mse - mse) / naive_mse) * 100
improvement_mae = ((naive_mae - mae) / naive_mae) * 100

# 2) Directional Accuracy
y_true_direction = (y_test[1:] > y_test[:-1]).astype(int)
y_pred_direction = (y_pred[1:] > y_pred[:-1]).astype(int)
directional_acc = accuracy_score(y_true_direction, y_pred_direction) * 100

# Log all metrics
logger.info(f"Physical Reservoir Test MSE: {mse:.6f}, MAE: {mae:.6f}")
logger.info(f"R-squared: {r2:.4f}")
logger.info(f"Naive Baseline MSE: {naive_mse:.6f}, MAE: {naive_mae:.6f}")
logger.info(f"Physical vs Naive Improvement: MSE {improvement_mse:.2f}%, MAE {improvement_mae:.2f}%")
logger.info(f"Directional Accuracy: {directional_acc:.2f}%")

print(f"Physical Reservoir Test MSE: {mse:.6f}, MAE: {mae:.6f}")
print(f"R-squared: {r2:.4f}")
print(f"Naive Baseline MSE: {naive_mse:.6f}, MAE: {naive_mae:.6f}")
print(f"Physical vs Naive Improvement: MSE {improvement_mse:.2f}%, MAE {improvement_mae:.2f}%")
print(f"Directional Accuracy: {directional_acc:.2f}%")

# %% [markdown]
# 5) Visualize predictions

# %%
plt.figure(figsize=(12,6))
plt.plot(y_test, label="True", linewidth=2)
plt.plot(y_pred, label="Physical Reservoir Pred", linewidth=2)
# Add 5-day moving average as confidence band
pred_series = pd.Series(y_pred)
ma_band = pred_series.rolling(5).mean()
plt.fill_between(range(len(y_pred)), ma_band - 0.1, ma_band + 0.1, alpha=0.2, color='orange', label='Confidence Band')
plt.legend()
plt.title("Physical Reservoir: True vs Predicted with Confidence Bands")
plt.show()

# %% [markdown]
# 6) Residual Analysis

# %%
residuals = y_test - y_pred
plt.figure(figsize=(10,4))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Analysis - Are Errors Random?')
plt.grid(True, alpha=0.3)
plt.show()

# Additional residual statistics
logger.info(f"Residual Mean: {np.mean(residuals):.6f}")
logger.info(f"Residual Std: {np.std(residuals):.6f}")
print(f"Residual Mean: {np.mean(residuals):.6f}")
print(f"Residual Std: {np.std(residuals):.6f}")

# %% [markdown]
# 7) Save model

# %%
joblib.dump({"physical_reservoir": dr, "readout": ridge, "mu": mu, "sigma": sigma, "window_size": window_size}, os.path.join(MODELS_DIR, "physical_reservoir_model.pkl"))
logger.info("Saved Physical Reservoir model to models/physical_reservoir_model.pkl")
print("[OK] Saved Physical Reservoir model to models/physical_reservoir_model.pkl")


