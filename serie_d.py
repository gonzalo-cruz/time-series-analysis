"""
Serie D – Serie mensual (consumo/ventas)
Predice los 10 meses de 1991 (D1-D10, Ene-Oct).

Grid search SARIMA con m=12. Validación: últimos 12 meses (año 1990).
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
print("SERIE D – Serie mensual")
print("="*60)

df_D = pd.read_csv(os.path.join(DATA_DIR, "train_series_D.csv"))
df_D["Date"] = pd.to_datetime(df_D["Date"], format="%d-%b-%Y")
df_D = df_D.sort_values("Date").set_index("Date")
serie_D = df_D["value"].astype(float)

test_D = pd.read_csv(os.path.join(DATA_DIR, "test_series_D.csv"))
test_D["Date"] = pd.to_datetime(test_D[" Date"].str.strip(), format="%d-%b-%Y")
n_pred_D = len(test_D)

print(f"  Training: {serie_D.index[0].date()} -> {serie_D.index[-1].date()} ({len(serie_D)} obs)")
print(f"  Predicciones requeridas: {n_pred_D} (Ene-Oct 1991)")

plot_series(serie_D, "Serie D – Serie mensual", "Fecha", "Valor",
            path=os.path.join(FIG_DIR, "D_serie.png"))

decomp_D = seasonal_decompose(serie_D, model="additive", period=12)
fig, axes = plt.subplots(4, 1, figsize=(12, 9))
for ax, comp, lbl in zip(axes,
    [serie_D, decomp_D.trend, decomp_D.seasonal, decomp_D.resid],
    ["Original", "Tendencia", "Estacionalidad anual", "Residuos"]):
    ax.plot(comp, linewidth=0.9)
    ax.set_title(lbl)
    ax.grid(True, alpha=0.3)
plt.suptitle("Descomposicion Serie D (periodo=12)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "D_descomposicion.png"), dpi=100)
plt.close()
del decomp_D
gc.collect()

adf_test(serie_D, "Serie D (nivel)")
adf_test(serie_D.diff(12).dropna(), "Serie D (diferencia estacional)")
plot_acf_pacf(serie_D, lags=36, title="Serie D",
              path=os.path.join(FIG_DIR, "D_acf_pacf.png"))

# Grid search: últimos 12 meses como validación
train_D = serie_D.iloc[:-12]
val_D   = serie_D.iloc[-12:]
print(f"\n  Grid search: {len(GRID_SMALL)} combinaciones SARIMA (m=12)...")
df_Dr = run_grid(train_D, val_D.values, GRID_SMALL, 12, 12, tag="D")

print(f"\n  Top 10 Serie D:")
print(df_Dr.head(10).to_string())

row = df_Dr.iloc[0]
p, d, q = int(row.p), int(row.d), int(row.q)
P, D, Q = int(row.P), int(row.D), int(row.Q)
lbl_D = f"SARIMA({p},{d},{q})({P},{D},{Q},12)"
rmse_val_D = row.rmse
print(f"\n  Mejor modelo: {lbl_D}  RMSE={rmse_val_D:.2f}")

_, fit_D_val = fit_safe(train_D, (p, d, q), (P, D, Q, 12), 12)

resid_D = fit_D_val.resid.dropna()
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
axes[0, 0].plot(resid_D.values, lw=0.7, color="steelblue")
axes[0, 0].axhline(0, color="r", lw=0.8, ls="--")
axes[0, 0].set_title("Residuos"); axes[0, 0].grid(alpha=0.3)
axes[0, 1].hist(resid_D, bins=25, density=True, alpha=0.7)
xr = np.linspace(resid_D.min(), resid_D.max(), 200)
axes[0, 1].plot(xr, stats.norm.pdf(xr, resid_D.mean(), resid_D.std()), "r-", lw=2)
axes[0, 1].set_title("Histograma + normal"); axes[0, 1].grid(alpha=0.3)
plot_acf(resid_D, lags=24, ax=axes[1, 0], title="ACF residuos")
axes[1, 1].axis("off")
lb = acorr_ljungbox(resid_D, lags=[12, 24], return_df=True)
txt = "Ljung-Box:\n"
for lag, row_lb in lb.iterrows():
    ok = "ok" if row_lb["lb_pvalue"] > 0.05 else "FALLA"
    txt += f"  lag={int(lag):2d}  p={row_lb['lb_pvalue']:.4f} {ok}\n"
jb, jp = stats.jarque_bera(resid_D)
txt += f"\nJarque-Bera p={jp:.4f} {'ok' if jp > 0.05 else 'FALLA'}"
axes[1, 1].text(0.05, 0.95, txt, transform=axes[1, 1].transAxes, va="top",
                fontsize=9, fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))
plt.suptitle(f"Residuos – {lbl_D} – Serie D", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "D_residuos.png"), dpi=100, bbox_inches="tight")
plt.close()
del fit_D_val, train_D, val_D
gc.collect()

print(f"\n  Refitting sobre {len(serie_D)} obs...")
pred_D, _ = fit_safe(serie_D, (p, d, q), (P, D, Q, 12), n_pred_D)
test_dates_D = test_D["Date"].values
print(f"  Predicciones D1-D{n_pred_D}: {np.round(pred_D, 2)}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(serie_D, label="Historico", linewidth=0.9)
ax.plot(test_dates_D, pred_D, "b-o",
        label=f"{lbl_D}  RMSE val={rmse_val_D:.2f}",
        markersize=5, linewidth=1.2)
ax.set_title(f"Serie D – Predicciones Ene-Oct 1991 ({lbl_D})")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "D_predicciones.png"), dpi=100)
plt.close()
gc.collect()

save_predictions([f"D{i}" for i in range(1, n_pred_D + 1)], pred_D,
                 os.path.join(DATA_DIR, "pred_D.csv"))
print("Serie D completada.")
