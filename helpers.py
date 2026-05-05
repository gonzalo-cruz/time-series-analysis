import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error

import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(DATA_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def adf_test(series, name=""):
    result = adfuller(series.dropna())
    print(f"  ADF Test [{name}]: stat={result[0]:.4f}, p={result[1]:.4f} "
          f"→ {'ESTACIONARIA' if result[1] < 0.05 else 'NO estacionaria'}")
    return result[1]


def plot_series(series, title, xlabel="Fecha", ylabel="Valor", path=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series.index, series.values, linewidth=0.8)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_acf_pacf(series, lags, title, path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    plot_acf(series.dropna(),  lags=lags, ax=axes[0], title=f"ACF - {title}")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], title=f"PACF - {title}", method="ywm")
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    plt.close(fig)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def save_predictions(ids, values, path):
    pd.DataFrame({"id": ids, "value": values}).to_csv(path, index=False)
    print(f"  Predicciones guardadas en: {path}")
