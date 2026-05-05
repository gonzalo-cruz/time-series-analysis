"""
Serie B – Nacimientos diarios
Predice todos los dias de 2003 (B1-B365).

Modelo: SARIMA(2,0,1)(1,1,1,7) con terminos de Fourier anuales (K=5) como regresores
externos. m=7 captura la estacionalidad semanal (pocos nacimientos en fin de semana).
Para la estacionalidad anual usamos 5 pares seno/coseno calculados a partir del dia
del año. Estos terminos de Fourier se construyen exclusivamente con el indice de fecha,
sin mirar la serie, asi que no hay fuga de informacion al calcularlos para 2003.

Fourier anual: para cada frecuencia k=1..K se añaden sin(2*pi*k*t) y cos(2*pi*k*t)
donde t = dia_del_año / 365.25. Con K=5 tenemos 10 regresores que cubren los primeros
cinco armonicos de la estacionalidad anual.
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
print("SERIE B – Nacimientos diarios")
print("="*60)

# Cargar y estructurar datos de entrenamiento
df_B = pd.read_csv(os.path.join(DATA_DIR, "train_series_B.csv"))
df_B["Date"] = pd.to_datetime(df_B[["year", "month", "day"]])
df_B = df_B.sort_values("Date").set_index("Date")
serie_B = df_B["births"].astype(float)

# Cargar datos de prueba para 2003
test_B = pd.read_csv(os.path.join(DATA_DIR, "test_serie_B.csv"))
test_B["Date"] = pd.to_datetime(test_B[["year", "month", "day"]])
n_pred_B = len(test_B)

print(f"  Training: {serie_B.index[0].date()} -> {serie_B.index[-1].date()} ({len(serie_B)} obs)")
print(f"  Predicciones requeridas: {n_pred_B} (año 2003)")

# Visualizar serie de nacimientos diarios
plot_series(serie_B, "Serie B – Nacimientos diarios", "Fecha", "Nacimientos",
            path=os.path.join(FIG_DIR, "B_serie.png"))

# Usar últimos 3 años para descomposición (para claridad visual)
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

# Pruebas de estacionariedad y gráficos ACF/PACF
adf_test(serie_B, "Serie B")
plot_acf_pacf(serie_B, lags=30, title="Serie B",
              path=os.path.join(FIG_DIR, "B_acf_pacf.png"))


def fourier_annual(idx, K):
    """Regresores de Fourier anuales: K pares sin/cos, t = dia_del_año/365.25."""
    t = idx.dayofyear / 365.25
    cols = {}
    for k in range(1, K + 1):
        cols[f"sin{k}"] = np.sin(2 * np.pi * k * t)
        cols[f"cos{k}"] = np.cos(2 * np.pi * k * t)
    return pd.DataFrame(cols, index=idx).values

# Split: entrenar 1994-2001, validar 2002
train_B = serie_B[serie_B.index.year < 2002]
val_B   = serie_B[serie_B.index.year == 2002]

# Crear regresores de Fourier para ambos conjuntos
K = 5
exog_tr = fourier_annual(train_B.index, K)
exog_va = fourier_annual(val_B.index, K)

print(f"\n  Ajustando SARIMA(2,0,1)(1,1,1,7)+Fourier K={K} sobre {len(train_B)} obs...")
fit_B_val = SARIMAX(
    train_B,
    exog=exog_tr,
    order=(2, 0, 1),
    seasonal_order=(1, 1, 1, 7),
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False, maxiter=100)

# Predicciones de validación
pred_val_B = fit_B_val.get_forecast(len(val_B), exog=exog_va).predicted_mean.values
rmse_val_B = rmse(val_B.values, pred_val_B)
print(f"  Validacion (2002) -> RMSE={rmse_val_B:.2f}")

# Verificar que residuos sean ruido blanco
resid_B = fit_B_val.resid.dropna()
resid_plot = resid_B.iloc[-500:]  # Mostrar últimos 500 para claridad
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
for lag, row in lb.iterrows():
    ok = "ok" if row["lb_pvalue"] > 0.05 else "FALLA"
    txt += f"  lag={int(lag):2d}  p={row['lb_pvalue']:.4f} {ok}\n"
jb, jp = stats.jarque_bera(resid_B)
txt += f"\nJarque-Bera p={jp:.4f} {'ok' if jp > 0.05 else 'FALLA'}"
axes[1, 1].text(0.05, 0.95, txt, transform=axes[1, 1].transAxes, va="top",
                fontsize=9, fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))
plt.suptitle(f"Residuos – SARIMA(2,0,1)(1,1,1,7)+F{K} – Serie B", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "B_residuos.png"), dpi=100, bbox_inches="tight")
plt.close()
del fit_B_val, train_B, val_B
gc.collect()

del fit_B_val, train_B, val_B
gc.collect()

# Reentrenar con TODOS los datos para mejores estimaciones en 2003
print(f"\n  Refitting sobre {len(serie_B)} obs...")
exog_all  = fourier_annual(serie_B.index, K)  # Fourier para todo el histórico
pred_dates_B = pd.DatetimeIndex(test_B["Date"].values)
exog_test = fourier_annual(pred_dates_B, K)  # Fourier para fechas de predicción

fit_B_full = SARIMAX(
    serie_B,
    exog=exog_all,
    order=(2, 0, 1),
    seasonal_order=(1, 1, 1, 7),
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False, maxiter=100)

# Generar predicciones para 2003
pred_B = fit_B_full.get_forecast(n_pred_B, exog=exog_test).predicted_mean.values
print(f"  Predicciones B: min={pred_B.min():.0f}, max={pred_B.max():.0f}, media={pred_B.mean():.0f}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(serie_B[-180:], label="Historico (ultimos 180 dias)", linewidth=0.7)
ax.plot(pred_dates_B, pred_B, "r-",
        label=f"SARIMA(2,0,1)(1,1,1,7)+F{K}  RMSE val={rmse_val_B:.0f}", linewidth=0.9)
ax.set_title(f"Serie B – Predicciones 2003 (SARIMA+Fourier K={K})")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "B_predicciones.png"), dpi=100)
plt.close()
del fit_B_full
gc.collect()

save_predictions([f"B{i}" for i in range(1, n_pred_B + 1)], pred_B,
                 os.path.join(DATA_DIR, "pred_B.csv"))
print("Serie B completada.")
