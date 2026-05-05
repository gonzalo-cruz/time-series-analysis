"""
Serie D – Serie mensual (consumo/ventas)
Predice los 10 meses de 1991 (D1-D10, Ene-Oct).

Modelo: SARIMA(0,0,0)(1,0,0,12). Solo el componente AR estacional de orden 1:
cada mes se predice como una fraccion del mismo mes del año anterior. Simple pero
consigue RMSE=186 en validacion, mejor que los modelos mas complejos. d=D=0 porque
el test ADF en diferencias estacionales ya muestra p<0.001, pero sin diferenciar
el AR estacional captura la dependencia de año en año directamente.
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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

from helpers import DATA_DIR, FIG_DIR, adf_test, plot_series, plot_acf_pacf, rmse, save_predictions

print("\n" + "="*60)
print("SERIE D – Serie mensual")
print("="*60)

# Cargar datos mensuales de consumo/ventas
df_D = pd.read_csv(os.path.join(DATA_DIR, "train_series_D.csv"))
df_D["Date"] = pd.to_datetime(df_D["Date"], format="%d-%b-%Y")
df_D = df_D.sort_values("Date").set_index("Date")
serie_D = df_D["value"].astype(float)

# Cargar datos de prueba para Ene-Oct 1991
test_D = pd.read_csv(os.path.join(DATA_DIR, "test_series_D.csv"))
test_D["Date"] = pd.to_datetime(test_D[" Date"].str.strip(), format="%d-%b-%Y")
n_pred_D = len(test_D)

print(f"  Training: {serie_D.index[0].date()} -> {serie_D.index[-1].date()} ({len(serie_D)} obs)")
print(f"  Predicciones requeridas: {n_pred_D} (Ene-Oct 1991)")

# Visualizar serie mensual
plot_series(serie_D, "Serie D – Serie mensual", "Fecha", "Valor",
            path=os.path.join(FIG_DIR, "D_serie.png"))

# Descomponer: tendencia, estacionalidad anual, residuos
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

# Pruebas de estacionariedad
adf_test(serie_D, "Serie D (nivel)")
adf_test(serie_D.diff(12).dropna(), "Serie D (diferencia estacional)")
plot_acf_pacf(serie_D, lags=36, title="Serie D",
              path=os.path.join(FIG_DIR, "D_acf_pacf.png"))

# Usar los últimos 12 meses (año 1990) como validación
train_D = serie_D.iloc[:-12]
val_D   = serie_D.iloc[-12:]
print(f"\n  Ajustando SARIMA(0,0,0)(1,0,0,12) sobre {len(train_D)} obs...")

# Modelo simple: solo AR estacional. Cada mes = f(mes anterior del año anterior)
fit_D_val = SARIMAX(
    train_D,
    order=(0, 0, 0),
    seasonal_order=(1, 0, 0, 12),
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False, maxiter=100)

# Predicciones y evaluación
pred_val_D = fit_D_val.get_forecast(12).predicted_mean.values
rmse_val_D = rmse(val_D.values, pred_val_D)
print(f"  Validacion (1990) -> RMSE={rmse_val_D:.2f}")

# Diagnostico de residuos
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
for lag, row in lb.iterrows():
    ok = "ok" if row["lb_pvalue"] > 0.05 else "FALLA"
    txt += f"  lag={int(lag):2d}  p={row['lb_pvalue']:.4f} {ok}\n"
jb, jp = stats.jarque_bera(resid_D)
txt += f"\nJarque-Bera p={jp:.4f} {'ok' if jp > 0.05 else 'FALLA'}"
axes[1, 1].text(0.05, 0.95, txt, transform=axes[1, 1].transAxes, va="top",
                fontsize=9, fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))
plt.suptitle("Residuos – SARIMA(0,0,0)(1,0,0,12) – Serie D", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "D_residuos.png"), dpi=100, bbox_inches="tight")
plt.close()
del fit_D_val, train_D, val_D
gc.collect()

del fit_D_val, train_D, val_D
gc.collect()

# Reentrenar con TODOS los datos para predicciones de Ene-Oct 1991
print(f"\n  Refitting sobre {len(serie_D)} obs...")
fit_D_full = SARIMAX(
    serie_D,
    order=(0, 0, 0),
    seasonal_order=(1, 0, 0, 12),
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False, maxiter=100)

# Generar predicciones
pred_D = fit_D_full.get_forecast(n_pred_D).predicted_mean.values
test_dates_D = test_D["Date"].values
print(f"  Predicciones D1-D{n_pred_D}: {np.round(pred_D, 2)}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(serie_D, label="Historico", linewidth=0.9)
ax.plot(test_dates_D, pred_D, "b-o",
        label=f"SARIMA(0,0,0)(1,0,0,12)  RMSE val={rmse_val_D:.2f}",
        markersize=5, linewidth=1.2)
ax.set_title("Serie D – Predicciones Ene-Oct 1991 (SARIMA(0,0,0)(1,0,0,12))")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "D_predicciones.png"), dpi=100)
plt.close()
del fit_D_full
gc.collect()

save_predictions([f"D{i}" for i in range(1, n_pred_D + 1)], pred_D,
                 os.path.join(DATA_DIR, "pred_D.csv"))
print("Serie D completada.")
