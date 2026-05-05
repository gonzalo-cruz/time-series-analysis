"""
Serie C – Temperatura global mensual
Predice los 12 meses de 2025 (C1-C12).

Grid search SARIMA sobre ventana 40 años (últimos 480 meses), m=12.
Además se prueban los top-5 con regresor de tendencia lineal exógeno.
Se selecciona la combinación con menor RMSE en validación (últimos 24 meses).
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
print("SERIE C – Temperatura global mensual")
print("="*60)

df_C_wide = pd.read_csv(os.path.join(DATA_DIR, "train_series_C.csv"))
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
df_C_long = df_C_wide.melt(id_vars="Year", value_vars=months,
                            var_name="Month", value_name="temp")
df_C_long["Month_num"] = df_C_long["Month"].map({m: i+1 for i, m in enumerate(months)})
df_C_long["Date"] = pd.to_datetime(
    df_C_long["Year"].astype(str) + "-" + df_C_long["Month_num"].astype(str) + "-01"
)
serie_C_full = df_C_long.sort_values("Date").set_index("Date")["temp"].astype(float)
del df_C_wide, df_C_long
gc.collect()

test_C = pd.read_csv(os.path.join(DATA_DIR, "test_serie_C.csv"))
n_pred_C = len(test_C)

print(f"  Training completo: {serie_C_full.index[0].date()} -> {serie_C_full.index[-1].date()} ({len(serie_C_full)} obs)")
print(f"  Predicciones requeridas: {n_pred_C} (meses de 2025)")

plot_series(serie_C_full, "Serie C – Temperatura global mensual (°C)", "Año", "Temperatura",
            path=os.path.join(FIG_DIR, "C_serie.png"))

decomp_C = seasonal_decompose(serie_C_full, model="additive", period=12)
fig, axes = plt.subplots(4, 1, figsize=(12, 9))
for ax, comp, lbl in zip(axes,
    [serie_C_full, decomp_C.trend, decomp_C.seasonal, decomp_C.resid],
    ["Original", "Tendencia", "Estacionalidad anual", "Residuos"]):
    ax.plot(comp, linewidth=0.7)
    ax.set_title(lbl)
    ax.grid(True, alpha=0.3)
plt.suptitle("Descomposicion Serie C (periodo=12)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "C_descomposicion.png"), dpi=100)
plt.close()
del decomp_C
gc.collect()

adf_test(serie_C_full, "Serie C (nivel)")
adf_test(serie_C_full.diff(12).dropna(), "Serie C (diferencia estacional)")
plot_acf_pacf(serie_C_full, lags=36, title="Serie C",
              path=os.path.join(FIG_DIR, "C_acf_pacf.png"))

# Usar solo los últimos 40 años (480 obs)
serie_C = serie_C_full.iloc[-480:]
del serie_C_full
gc.collect()

train_C = serie_C.iloc[:-24]
val_C   = serie_C.iloc[-24:]
n_tr = len(train_C)

# Grid search sin tendencia
print(f"\n  Grid search: {len(GRID_SMALL)} combinaciones SARIMA (m=12, ventana 40y)...")
df_Cr = run_grid(train_C, val_C.values, GRID_SMALL, 12, 24, tag="C")

# Probar los top-5 con tendencia lineal exógena
all_C_extra = []
if not df_Cr.empty:
    print(f"\n  Top 10 sin tendencia:")
    print(df_Cr.head(10).to_string())
    t_tr = np.arange(n_tr, dtype=float).reshape(-1, 1)
    t_va = np.arange(n_tr, n_tr + 24, dtype=float).reshape(-1, 1)
    print("  Probando top-5 con tendencia lineal exog...")
    for _, r_c in df_Cr.head(5).iterrows():
        p_, d_, q_ = int(r_c.p), int(r_c.d), int(r_c.q)
        P_, D_, Q_ = int(r_c.P), int(r_c.D), int(r_c.Q)
        pred_t, _ = fit_safe(train_C, (p_, d_, q_), (P_, D_, Q_, 12), 24,
                             exog_tr=t_tr, exog_fc=t_va)
        if pred_t is None:
            continue
        all_C_extra.append(dict(p=p_, d=d_, q=q_, P=P_, D=D_, Q=Q_,
                                rmse=rmse(val_C.values, pred_t),
                                aic=np.nan, window="40y+trend"))

df_Cr["window"] = "40y"
rows_C = [df_Cr]
if all_C_extra:
    rows_C.append(pd.DataFrame(all_C_extra))
df_C_all = pd.concat(rows_C, ignore_index=True).sort_values("rmse").reset_index(drop=True)

print(f"\n  Top 10 incluyendo tendencia:")
print(df_C_all[["p", "d", "q", "P", "D", "Q", "window", "rmse"]].head(10).to_string())

best_C = df_C_all.iloc[0]
p, d, q = int(best_C.p), int(best_C.d), int(best_C.q)
P, D, Q = int(best_C.P), int(best_C.D), int(best_C.Q)
use_trend = "trend" in str(best_C.get("window", ""))
lbl_C = f"SARIMA({p},{d},{q})({P},{D},{Q},12)[{'40y+trend' if use_trend else '40y'}]"
rmse_val_C = best_C.rmse
print(f"\n  Mejor modelo: {lbl_C}  RMSE={rmse_val_C:.4f}")

n_w = len(serie_C)
if use_trend:
    exog_tr_c  = np.arange(n_tr, dtype=float).reshape(-1, 1)
    exog_va_c  = np.arange(n_tr, n_tr + 24, dtype=float).reshape(-1, 1)
    exog_full_c = np.arange(n_w, dtype=float).reshape(-1, 1)
    exog_tst_c  = np.arange(n_w, n_w + n_pred_C, dtype=float).reshape(-1, 1)
else:
    exog_tr_c = exog_va_c = exog_full_c = exog_tst_c = None

_, fit_C_val = fit_safe(train_C, (p, d, q), (P, D, Q, 12), 24,
                        exog_tr=exog_tr_c, exog_fc=exog_va_c)

resid_C = fit_C_val.resid.dropna()
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
axes[0, 0].plot(resid_C.values, lw=0.6, color="steelblue")
axes[0, 0].axhline(0, color="r", lw=0.8, ls="--")
axes[0, 0].set_title("Residuos"); axes[0, 0].grid(alpha=0.3)
axes[0, 1].hist(resid_C, bins=30, density=True, alpha=0.7)
xr = np.linspace(resid_C.min(), resid_C.max(), 200)
axes[0, 1].plot(xr, stats.norm.pdf(xr, resid_C.mean(), resid_C.std()), "r-", lw=2)
axes[0, 1].set_title("Histograma + normal"); axes[0, 1].grid(alpha=0.3)
plot_acf(resid_C, lags=24, ax=axes[1, 0], title="ACF residuos")
axes[1, 1].axis("off")
lb = acorr_ljungbox(resid_C, lags=[12, 24], return_df=True)
txt = "Ljung-Box:\n"
for lag, row_lb in lb.iterrows():
    ok = "ok" if row_lb["lb_pvalue"] > 0.05 else "FALLA"
    txt += f"  lag={int(lag):2d}  p={row_lb['lb_pvalue']:.4f} {ok}\n"
jb, jp = stats.jarque_bera(resid_C)
txt += f"\nJarque-Bera p={jp:.4f} {'ok' if jp > 0.05 else 'FALLA'}"
axes[1, 1].text(0.05, 0.95, txt, transform=axes[1, 1].transAxes, va="top",
                fontsize=9, fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))
plt.suptitle(f"Residuos – {lbl_C} – Serie C", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "C_residuos.png"), dpi=100, bbox_inches="tight")
plt.close()
del fit_C_val, train_C, val_C
gc.collect()

print(f"\n  Refitting sobre {len(serie_C)} obs...")
pred_C, _ = fit_safe(serie_C, (p, d, q), (P, D, Q, 12), n_pred_C,
                     exog_tr=exog_full_c, exog_fc=exog_tst_c)
test_dates_C = pd.date_range("2025-01-01", periods=12, freq="MS")
print(f"  Predicciones C1-C12: {np.round(pred_C, 4)}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(serie_C[-60:], label="Historico (ultimos 60 meses)", linewidth=0.9)
ax.plot(test_dates_C, pred_C, "b-o",
        label=f"{lbl_C}  RMSE val={rmse_val_C:.4f}",
        markersize=5, linewidth=1.2)
ax.set_title(f"Serie C – Predicciones 2025 ({lbl_C})")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "C_predicciones.png"), dpi=100)
plt.close()
gc.collect()

save_predictions([f"C{i}" for i in range(1, n_pred_C + 1)], pred_C,
                 os.path.join(DATA_DIR, "pred_C.csv"))
print("Serie C completada.")
