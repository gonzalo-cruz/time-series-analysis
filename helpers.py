import warnings
warnings.filterwarnings("ignore")

import itertools
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
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


def fit_safe(y_train, order, seas_order, n_fc, exog_tr=None, exog_fc=None):
    d, D = order[1], seas_order[1]
    trend = "c" if d + D < 2 else "n"
    try:
        fit = SARIMAX(y_train, exog=exog_tr,
                      order=order, seasonal_order=seas_order,
                      trend=trend,
                      enforce_stationarity=False,
                      enforce_invertibility=False).fit(disp=False, maxiter=100)
        pred = fit.get_forecast(n_fc, exog=exog_fc).predicted_mean.values
        if not np.all(np.isfinite(pred)):
            return None, None
        return pred, fit
    except Exception:
        return None, None


def build_grid(p_r, d_r, q_r, P_r, D_r, Q_r, max_pq=3, max_PQ=2):
    return [(p, d, q, P, D, Q)
            for p, d, q, P, D, Q in itertools.product(p_r, d_r, q_r, P_r, D_r, Q_r)
            if p + q <= max_pq and P + Q <= max_PQ]


def run_grid(y_tr, y_val, combos, m_s, n_fc, exog_tr=None, exog_fc=None, tag=""):
    rows, N, t0 = [], len(combos), time.time()
    for i, (p, d, q, P, D, Q) in enumerate(combos):
        if (i + 1) % 20 == 0:
            print(f"    [{tag}] {i+1}/{N}  ({time.time()-t0:.0f}s)", flush=True)
        pred, fit = fit_safe(y_tr, (p, d, q), (P, D, Q, m_s), n_fc,
                             exog_tr=exog_tr, exog_fc=exog_fc)
        if pred is None:
            continue
        rows.append(dict(p=p, d=d, q=q, P=P, D=D, Q=Q,
                         rmse=rmse(y_val, pred),
                         aic=getattr(fit, "aic", np.nan)))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def fourier_annual(idx, K):
    t = idx.dayofyear / 365.25
    cols = {f"sin{k}": np.sin(2 * np.pi * k * t) for k in range(1, K + 1)}
    cols.update({f"cos{k}": np.cos(2 * np.pi * k * t) for k in range(1, K + 1)})
    return pd.DataFrame(cols, index=idx).values


GRID_SMALL = build_grid(range(3), [0, 1], range(3), range(2), [0, 1], range(2))
