"""
Serie A – Precio bursátil diario
Predice los 10 días siguientes al periodo de entrenamiento (A1-A10).

Grid search SARIMA sobre m=5 (días laborables). Validación: últimas 20 obs.
Se selecciona el modelo con menor RMSE en validación y se reentrena sobre todos los datos.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gc
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

from helpers import (DATA_DIR, FIG_DIR, adf_test, plot_series, plot_acf_pacf,
                     rmse, save_predictions, fit_safe, run_grid, GRID_SMALL)

print("\n" + "="*60)
print("SERIE A – Precio bursatil diario")
print("="*60)

df_A = pd.read_csv(os.path.join(DATA_DIR, "train_series_A.csv"), parse_dates=["Date"])
serie_A = df_A.sort_values("Date").set_index("Date")["value"].astype(float)

test_A = pd.read_csv(os.path.join(DATA_DIR, "test_serie_A.csv"))
test_A["timestamp"] = pd.to_datetime(test_A["timestamp"].str.strip())
n_pred_A = len(test_A)

print(f"  Training: {serie_A.index[0].date()} -> {serie_A.index[-1].date()} ({len(serie_A)} obs)")
print(f"  Predicciones requeridas: {n_pred_A}")

plot_series(serie_A, "Serie A – Precio bursatil diario", "Fecha", "Precio",
            path=os.path.join(FIG_DIR, "A_serie.png"))

decomp_A = seasonal_decompose(serie_A, model="additive", period=5)
fig, axes = plt.subplots(4, 1, figsize=(12, 9))
for ax, comp, lbl in zip(axes,
    [serie_A, decomp_A.trend, decomp_A.seasonal, decomp_A.resid],
    ["Original", "Tendencia", "Estacionalidad (semanal)", "Residuos"]):
    ax.plot(comp, linewidth=0.6)
    ax.set_title(lbl)
    ax.grid(True, alpha=0.3)
plt.suptitle("Descomposicion Serie A (periodo=5)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "A_descomposicion.png"), dpi=100)
plt.close()
del decomp_A
gc.collect()

adf_test(serie_A, "Serie A (nivel)")
adf_test(serie_A.diff().dropna(), "Serie A (primera diferencia)")
plot_acf_pacf(serie_A.diff().dropna(), lags=30, title="Serie A diferenciada",
              path=os.path.join(FIG_DIR, "A_acf_pacf.png"))

# Grid search: últimas 20 obs como validación
train_A = serie_A.iloc[:-20]
val_A   = serie_A.iloc[-20:]
print(f"\n  Grid search: {len(GRID_SMALL)} combinaciones SARIMA (m=5)...")
df_Ar = run_grid(train_A, val_A.values, GRID_SMALL, 5, 20, tag="A")

print(f"\n  Top 10 Serie A:")
print(df_Ar.head(10).to_string())

row = df_Ar.iloc[0]
p, d, q = int(row.p), int(row.d), int(row.q)
P, D, Q = int(row.P), int(row.D), int(row.Q)
lbl_A = f"SARIMA({p},{d},{q})({P},{D},{Q},5)"
rmse_val_A = row.rmse
print(f"\n  Mejor modelo: {lbl_A}  RMSE={rmse_val_A:.4f}")

_, fit_A_val = fit_safe(train_A, (p, d, q), (P, D, Q, 5), 20)

resid_A = fit_A_val.resid.dropna()
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
axes[0, 0].plot(resid_A.values, lw=0.6, color="steelblue")
axes[0, 0].axhline(0, color="r", lw=0.8, ls="--")
axes[0, 0].set_title("Residuos"); axes[0, 0].grid(alpha=0.3)
axes[0, 1].hist(resid_A, bins=35, density=True, alpha=0.7)
xr = np.linspace(resid_A.min(), resid_A.max(), 200)
axes[0, 1].plot(xr, stats.norm.pdf(xr, resid_A.mean(), resid_A.std()), "r-", lw=2)
axes[0, 1].set_title("Histograma + normal"); axes[0, 1].grid(alpha=0.3)
plot_acf(resid_A, lags=20, ax=axes[1, 0], title="ACF residuos")
axes[1, 1].axis("off")
lb = acorr_ljungbox(resid_A, lags=[5, 10, 15], return_df=True)
txt = "Ljung-Box:\n"
for lag, row_lb in lb.iterrows():
    ok = "ok" if row_lb["lb_pvalue"] > 0.05 else "FALLA"
    txt += f"  lag={int(lag):2d}  p={row_lb['lb_pvalue']:.4f} {ok}\n"
jb, jp = stats.jarque_bera(resid_A)
txt += f"\nJarque-Bera p={jp:.4f} {'ok' if jp > 0.05 else 'FALLA'}"
axes[1, 1].text(0.05, 0.95, txt, transform=axes[1, 1].transAxes, va="top",
                fontsize=9, fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))
plt.suptitle(f"Residuos – {lbl_A} – Serie A", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "A_residuos.png"), dpi=100, bbox_inches="tight")
plt.close()
del fit_A_val, train_A, val_A
gc.collect()

print(f"\n  Refitting sobre {len(serie_A)} obs...")
pred_A, _ = fit_safe(serie_A, (p, d, q), (P, D, Q, 5), n_pred_A)
print(f"  Predicciones A: {np.round(pred_A, 2)}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(serie_A[-60:], label="Historico (ultimos 60 dias)", linewidth=0.9)
ax.plot(test_A["timestamp"].values, pred_A, "b-o",
        label=f"{lbl_A}  RMSE val={rmse_val_A:.2f}",
        markersize=5, linewidth=1.2)
ax.set_title(f"Serie A – Predicciones ({lbl_A})")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "A_predicciones.png"), dpi=100)
plt.close()
gc.collect()

save_predictions([f"A{i}" for i in range(1, n_pred_A + 1)], pred_A,
                 os.path.join(DATA_DIR, "pred_A.csv"))
print("Serie A completada.")
