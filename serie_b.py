"""
Serie B – Nacimientos diarios
Predice todos los dias de 2003 (B1-B365).

Búsqueda en dos fases:
  Fase 1: grid completo (GRID_SMALL) con K=3 Fourier, ventana 4 años.
  Fase 2: top-5 estructuras × K∈{3,4,5} × ventanas {4y, all} → elige la mejor combinación.
m=7 captura la estacionalidad semanal; Fourier captura la anual sin fuga de datos.
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
                     rmse, save_predictions, fit_safe, run_grid, fourier_annual, GRID_SMALL)

print("\n" + "="*60)
print("SERIE B – Nacimientos diarios")
print("="*60)

df_B = pd.read_csv(os.path.join(DATA_DIR, "train_series_B.csv"))
df_B["Date"] = pd.to_datetime(df_B[["year", "month", "day"]])
df_B = df_B.sort_values("Date").set_index("Date")
serie_B = df_B["births"].astype(float)

test_B = pd.read_csv(os.path.join(DATA_DIR, "test_serie_B.csv"))
test_B["Date"] = pd.to_datetime(test_B[["year", "month", "day"]])
pred_dates_B = pd.DatetimeIndex(test_B["Date"].values)
n_pred_B = len(test_B)

print(f"  Training: {serie_B.index[0].date()} -> {serie_B.index[-1].date()} ({len(serie_B)} obs)")
print(f"  Predicciones requeridas: {n_pred_B} (año 2003)")

plot_series(serie_B, "Serie B – Nacimientos diarios", "Fecha", "Nacimientos",
            path=os.path.join(FIG_DIR, "B_serie.png"))

work_B = serie_B.iloc[-1095:]
decomp_B = seasonal_decompose(work_B, model="additive", period=7)
fig, axes = plt.subplots(4, 1, figsize=(12, 9))
for ax, comp, lbl in zip(axes,
    [work_B, decomp_B.trend, decomp_B.seasonal, decomp_B.resid],
    ["Original", "Tendencia", "Estacionalidad (semanal)", "Residuos"]):
    ax.plot(comp, linewidth=0.6)
    ax.set_title(lbl)
    ax.grid(True, alpha=0.3)
plt.suptitle("Descomposicion Serie B (periodo=7)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "B_descomposicion.png"), dpi=100)
plt.close()
del decomp_B, work_B
gc.collect()

adf_test(serie_B, "Serie B")
plot_acf_pacf(serie_B, lags=30, title="Serie B",
              path=os.path.join(FIG_DIR, "B_acf_pacf.png"))

windows_B = {
    "4y": serie_B[serie_B.index.year >= 1999],
    "all": serie_B,
}

# Fase 1: grid completo con K=3, ventana 4y
w4 = windows_B["4y"]
tr1 = w4[w4.index.year < 2002]
va1 = w4[w4.index.year == 2002]
print(f"\n  Fase 1: {len(GRID_SMALL)} modelos con K=3, ventana 4y...")
K3_tr, K3_va = fourier_annual(tr1.index, 3), fourier_annual(va1.index, 3)
df_B1 = run_grid(tr1, va1.values, GRID_SMALL, 7, len(va1),
                 exog_tr=K3_tr, exog_fc=K3_va, tag="B-f1")

print(f"\n  Top 10 Fase 1:")
print(df_B1.head(10).to_string())

top_structs = [tuple(df_B1.iloc[i][["p", "d", "q", "P", "D", "Q"]].astype(int))
               for i in range(min(5, len(df_B1)))]
del tr1, va1, K3_tr, K3_va
gc.collect()

# Fase 2: top-5 estructuras × K∈{3,4,5} × ventanas {4y, all}
print(f"\n  Fase 2: {len(top_structs)} estructuras × K=3,4,5 × 2 ventanas ({len(top_structs)*3*2} fits)...")
all_B2 = []
for wname, wdata in windows_B.items():
    tr_w = wdata[wdata.index.year < 2002]
    va_w = wdata[wdata.index.year == 2002]
    for K in [3, 4, 5]:
        exog_tr_w = fourier_annual(tr_w.index, K)
        exog_va_w = fourier_annual(va_w.index, K)
        for (p, d, q, P, D, Q) in top_structs:
            pred, _ = fit_safe(tr_w, (p, d, q), (P, D, Q, 7), len(va_w),
                               exog_tr=exog_tr_w, exog_fc=exog_va_w)
            if pred is None:
                continue
            all_B2.append(dict(p=p, d=d, q=q, P=P, D=D, Q=Q, K=K, window=wname,
                               rmse=rmse(va_w.values, pred)))
    del tr_w, va_w
    gc.collect()

df_B2 = pd.DataFrame(all_B2).sort_values("rmse").reset_index(drop=True)
print(f"\n  Top 10 Fase 2:")
print(df_B2[["p", "d", "q", "P", "D", "Q", "K", "window", "rmse"]].head(10).to_string())

best_B = df_B2.iloc[0]
p, d, q = int(best_B.p), int(best_B.d), int(best_B.q)
P, D, Q = int(best_B.P), int(best_B.D), int(best_B.Q)
K_best, w_best = int(best_B.K), best_B.window
lbl_B = f"SARIMA({p},{d},{q})({P},{D},{Q},7)+F{K_best}[{w_best}]"
rmse_val_B = best_B.rmse
print(f"\n  Mejor modelo: {lbl_B}  RMSE={rmse_val_B:.2f}")

# Residuos sobre la ventana ganadora
wdata_b = windows_B[w_best]
tr_b = wdata_b[wdata_b.index.year < 2002]
va_b = wdata_b[wdata_b.index.year == 2002]
_, fit_B_val = fit_safe(tr_b, (p, d, q), (P, D, Q, 7), len(va_b),
                        exog_tr=fourier_annual(tr_b.index, K_best),
                        exog_fc=fourier_annual(va_b.index, K_best))

resid_B = fit_B_val.resid.dropna()
resid_plot = resid_B.iloc[-500:]
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
axes[0, 0].plot(resid_plot.values, lw=0.5, color="steelblue")
axes[0, 0].axhline(0, color="r", lw=0.8, ls="--")
axes[0, 0].set_title("Residuos"); axes[0, 0].grid(alpha=0.3)
axes[0, 1].hist(resid_B, bins=40, density=True, alpha=0.7)
xr = np.linspace(resid_B.min(), resid_B.max(), 200)
axes[0, 1].plot(xr, stats.norm.pdf(xr, resid_B.mean(), resid_B.std()), "r-", lw=2)
axes[0, 1].set_title("Histograma + normal"); axes[0, 1].grid(alpha=0.3)
plot_acf(resid_B, lags=30, ax=axes[1, 0], title="ACF residuos")
axes[1, 1].axis("off")
lb = acorr_ljungbox(resid_B, lags=[7, 14, 21], return_df=True)
txt = "Ljung-Box:\n"
for lag, row_lb in lb.iterrows():
    ok = "ok" if row_lb["lb_pvalue"] > 0.05 else "FALLA"
    txt += f"  lag={int(lag):2d}  p={row_lb['lb_pvalue']:.4f} {ok}\n"
jb, jp = stats.jarque_bera(resid_B)
txt += f"\nJarque-Bera p={jp:.4f} {'ok' if jp > 0.05 else 'FALLA'}"
axes[1, 1].text(0.05, 0.95, txt, transform=axes[1, 1].transAxes, va="top",
                fontsize=9, fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))
plt.suptitle(f"Residuos – {lbl_B} – Serie B", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "B_residuos.png"), dpi=100, bbox_inches="tight")
plt.close()
del fit_B_val, tr_b, va_b
gc.collect()

# Reentrenar con todos los datos históricos
print(f"\n  Refitting sobre {len(serie_B)} obs...")
exog_all  = fourier_annual(serie_B.index, K_best)
exog_test = fourier_annual(pred_dates_B, K_best)
pred_B, _ = fit_safe(serie_B, (p, d, q), (P, D, Q, 7), n_pred_B,
                     exog_tr=exog_all, exog_fc=exog_test)
print(f"  Predicciones B: min={pred_B.min():.0f}, max={pred_B.max():.0f}, media={pred_B.mean():.0f}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(serie_B[-180:], label="Historico (ultimos 180 dias)", linewidth=0.7)
ax.plot(pred_dates_B, pred_B, "r-",
        label=f"{lbl_B}  RMSE val={rmse_val_B:.0f}", linewidth=0.9)
ax.set_title(f"Serie B – Predicciones 2003 ({lbl_B})")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "B_predicciones.png"), dpi=100)
plt.close()
gc.collect()

save_predictions([f"B{i}" for i in range(1, n_pred_B + 1)], pred_B,
                 os.path.join(DATA_DIR, "pred_B.csv"))
print("Serie B completada.")
