"""
Serie A – Precio bursátil diario
Predice los 10 días siguientes al periodo de entrenamiento (A1-A10).

Usamos SARIMA(2,0,1)(1,1,1,5), seleccionado por grid search en validacion
(ultimas 20 observaciones). m=5 dias laborables captura la estacionalidad semanal.
d=0 porque el componente AR(2) ya recoge la dinamica de la serie sin diferenciar,
y D=1 maneja la componente estacional.
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
print("SERIE A – Precio bursatil diario")
print("="*60)

# Cargar datos de entrenamiento desde CSV
df_A = pd.read_csv(os.path.join(DATA_DIR, "train_series_A.csv"), parse_dates=["Date"])
serie_A = df_A.sort_values("Date").set_index("Date")["value"].astype(float)

# Cargar datos de prueba para saber cuántos días predecir
test_A = pd.read_csv(os.path.join(DATA_DIR, "test_serie_A.csv"))
test_A["timestamp"] = pd.to_datetime(test_A["timestamp"].str.strip())
n_pred_A = len(test_A)

print(f"  Training: {serie_A.index[0].date()} -> {serie_A.index[-1].date()} ({len(serie_A)} obs)")
print(f"  Predicciones requeridas: {n_pred_A}")

# Visualizar la serie original para entender tendencia y patrones
plot_series(serie_A, "Serie A – Precio bursatil diario", "Fecha", "Precio",
            path=os.path.join(FIG_DIR, "A_serie.png"))

# Descomponer la serie en componentes: tendencia, estacionalidad y residuos
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

# Prueba ADF: verificar si la serie es estacionaria
adf_test(serie_A, "Serie A (nivel)")
adf_test(serie_A.diff().dropna(), "Serie A (primera diferencia)")
# Gráficos ACF y PACF: ayudan a identificar parámetros AR y MA
plot_acf_pacf(serie_A.diff().dropna(), lags=30, title="Serie A diferenciada",
              path=os.path.join(FIG_DIR, "A_acf_pacf.png"))

# Usar los últimos 20 datos como conjunto de validación
train_A = serie_A.iloc[:-20]
val_A   = serie_A.iloc[-20:]
print(f"\n  Ajustando SARIMA(2,0,1)(1,1,1,5) sobre {len(train_A)} obs...")

# Entrenar modelo SARIMA con parámetros seleccionados
fit_A_val = SARIMAX(
    train_A,
    order=(2, 0, 1),
    seasonal_order=(1, 1, 1, 5),
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False, maxiter=100)

# Hacer predicciones en validación y calcular error (RMSE)
pred_val_A = fit_A_val.get_forecast(20).predicted_mean.values
rmse_val_A = rmse(val_A.values, pred_val_A)
print(f"  Validacion -> RMSE={rmse_val_A:.4f}")

# Extraer residuos y verificar si son ruido blanco (sin patrones)
resid_A = fit_A_val.resid.dropna()
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
# Residuos en el tiempo: deben fluctuar alrededor de 0
axes[0, 0].plot(resid_A.values, lw=0.6, color="steelblue")
axes[0, 0].axhline(0, color="r", lw=0.8, ls="--")
axes[0, 0].set_title("Residuos"); axes[0, 0].grid(alpha=0.3)
# Histograma: verificar distribución normal
axes[0, 1].hist(resid_A, bins=35, density=True, alpha=0.7)
xr = np.linspace(resid_A.min(), resid_A.max(), 200)
axes[0, 1].plot(xr, stats.norm.pdf(xr, resid_A.mean(), resid_A.std()), "r-", lw=2)
axes[0, 1].set_title("Histograma + normal"); axes[0, 1].grid(alpha=0.3)
# ACF: verificar autocorrelación (debe estar dentro de bandas)
plot_acf(resid_A, lags=20, ax=axes[1, 0], title="ACF residuos")
axes[1, 1].axis("off")
# Test Ljung-Box: p > 0.05 indica residuos independientes
lb = acorr_ljungbox(resid_A, lags=[5, 10, 15], return_df=True)
txt = "Ljung-Box:\n"
for lag, row in lb.iterrows():
    ok = "ok" if row["lb_pvalue"] > 0.05 else "FALLA"
    txt += f"  lag={int(lag):2d}  p={row['lb_pvalue']:.4f} {ok}\n"
jb, jp = stats.jarque_bera(resid_A)
txt += f"\nJarque-Bera p={jp:.4f} {'ok' if jp > 0.05 else 'FALLA'}"
axes[1, 1].text(0.05, 0.95, txt, transform=axes[1, 1].transAxes, va="top",
                fontsize=9, fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))
plt.suptitle("Residuos – SARIMA(2,0,1)(1,1,1,5) – Serie A", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "A_residuos.png"), dpi=100, bbox_inches="tight")
plt.close()
del fit_A_val, train_A, val_A
gc.collect()

# Reentrenar modelo con TODOS los datos para mejores estimaciones
print(f"\n  Refitting sobre {len(serie_A)} obs...")
fit_A_full = SARIMAX(
    serie_A,
    order=(2, 0, 1),
    seasonal_order=(1, 1, 1, 5),
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False, maxiter=100)

# Generar predicciones para n_pred_A períodos futuros
pred_A = fit_A_full.get_forecast(n_pred_A).predicted_mean.values
print(f"  Predicciones A: {np.round(pred_A, 2)}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(serie_A[-60:], label="Historico (ultimos 60 dias)", linewidth=0.9)
ax.plot(test_A["timestamp"].values, pred_A, "b-o",
        label=f"SARIMA(2,0,1)(1,1,1,5)  RMSE val={rmse_val_A:.2f}",
        markersize=5, linewidth=1.2)
ax.set_title("Serie A – Predicciones (SARIMA(2,0,1)(1,1,1,5))")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "A_predicciones.png"), dpi=100)
plt.close()
del fit_A_full
gc.collect()

save_predictions([f"A{i}" for i in range(1, n_pred_A + 1)], pred_A,
                 os.path.join(DATA_DIR, "pred_A.csv"))
print("Serie A completada.")
