"""
Serie C – Temperatura global mensual
Predice los 12 meses de 2025 (C1-C12).

Modelo: SARIMA(1,0,0)(0,1,1,12) con tendencia lineal como regresor externo.
Solo usamos los ultimos 40 años (1985-2024, 480 obs) porque la tasa de calentamiento
reciente es mucho mayor que la del siglo XIX, y los datos antiguos distorsionan
las predicciones a corto plazo.
La tendencia lineal exog captura el calentamiento global continuo de forma explicita,
complementando la diferenciacion estacional D=1.
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
print("SERIE C – Temperatura global mensual")
print("="*60)

# Cargar datos de temperatura en formato wide (años x meses)
df_C_wide = pd.read_csv(os.path.join(DATA_DIR, "train_series_C.csv"))
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# Convertir a formato long para facilitar indexación temporal
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

# Visualizar la serie completa de temperatura
plot_series(serie_C_full, "Serie C – Temperatura global mensual (°C)", "Año", "Temperatura",
            path=os.path.join(FIG_DIR, "C_serie.png"))

# Descomponer para identificar tendencia y ciclos anuales
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

# Pruebas de estacionariedad
adf_test(serie_C_full, "Serie C (nivel)")
adf_test(serie_C_full.diff(12).dropna(), "Serie C (diferencia estacional)")
plot_acf_pacf(serie_C_full, lags=36, title="Serie C",
              path=os.path.join(FIG_DIR, "C_acf_pacf.png"))

# Usar solo datos recientes evita que datos antiguos distorsionen predicciones
serie_C = serie_C_full.iloc[-480:]  # Últimos ~40 años
del serie_C_full
gc.collect()

# Split: entrenar 1985-2022, validar últimos 24 meses (2023-2024)
train_C = serie_C.iloc[:-24]
val_C   = serie_C.iloc[-24:]

# Crear regresores de tendencia lineal (captura calentamiento global)
n_tr = len(train_C)
exog_tr_c = np.arange(n_tr, dtype=float).reshape(-1, 1)
exog_va_c = np.arange(n_tr, n_tr + 24, dtype=float).reshape(-1, 1)

print(f"\n  Ajustando SARIMA(1,0,0)(0,1,1,12)+trend sobre {len(train_C)} obs...")
fit_C_val = SARIMAX(
    train_C,
    exog=exog_tr_c,
    order=(1, 0, 0),
    seasonal_order=(0, 1, 1, 12),
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False, maxiter=100)

# Predicciones de validación
pred_val_C = fit_C_val.get_forecast(24, exog=exog_va_c).predicted_mean.values
rmse_val_C = rmse(val_C.values, pred_val_C)
print(f"  Validacion (2023-2024) -> RMSE={rmse_val_C:.4f}")

# Diagnostico de residuos
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
for lag, row in lb.iterrows():
    ok = "ok" if row["lb_pvalue"] > 0.05 else "FALLA"
    txt += f"  lag={int(lag):2d}  p={row['lb_pvalue']:.4f} {ok}\n"
jb, jp = stats.jarque_bera(resid_C)
txt += f"\nJarque-Bera p={jp:.4f} {'ok' if jp > 0.05 else 'FALLA'}"
axes[1, 1].text(0.05, 0.95, txt, transform=axes[1, 1].transAxes, va="top",
                fontsize=9, fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))
plt.suptitle("Residuos – SARIMA(1,0,0)(0,1,1,12)+trend – Serie C", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "C_residuos.png"), dpi=100, bbox_inches="tight")
plt.close()
del fit_C_val, train_C, val_C
gc.collect()

del fit_C_val, train_C, val_C
gc.collect()

# Reentrenar con los 40 años completos (1985-2024)
print(f"\n  Refitting sobre {len(serie_C)} obs...")
n_full = len(serie_C)
exog_full_c = np.arange(n_full, dtype=float).reshape(-1, 1)
exog_tst_c  = np.arange(n_full, n_full + n_pred_C, dtype=float).reshape(-1, 1)

fit_C_full = SARIMAX(
    serie_C,
    exog=exog_full_c,
    order=(1, 0, 0),
    seasonal_order=(0, 1, 1, 12),
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False, maxiter=100)

# Generar predicciones para 2025
pred_C = fit_C_full.get_forecast(n_pred_C, exog=exog_tst_c).predicted_mean.values
test_dates_C = pd.date_range("2025-01-01", periods=12, freq="MS")
print(f"  Predicciones C1-C12: {np.round(pred_C, 4)}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(serie_C[-60:], label="Historico (ultimos 60 meses)", linewidth=0.9)
ax.plot(test_dates_C, pred_C, "b-o",
        label=f"SARIMA(1,0,0)(0,1,1,12)+trend  RMSE val={rmse_val_C:.4f}",
        markersize=5, linewidth=1.2)
ax.set_title("Serie C – Predicciones 2025 (SARIMA(1,0,0)(0,1,1,12)+trend)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "C_predicciones.png"), dpi=100)
plt.close()
del fit_C_full
gc.collect()

save_predictions([f"C{i}" for i in range(1, n_pred_C + 1)], pred_C,
                 os.path.join(DATA_DIR, "pred_C.csv"))
print("Serie C completada.")
