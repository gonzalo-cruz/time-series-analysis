# Práctica Final — Análisis y Predicción de Series Temporales
**Competición Kaggle · Universidad Rey Juan Carlos (URJC)**

## Descripción

Predicción de cuatro series temporales con distinta frecuencia y naturaleza. Para cada serie se realiza análisis exploratorio, tests de estacionariedad, selección del modelo por RMSE en validación y predicción final reajustada sobre todos los datos.

El informe completo se encuentra en `informe.pdf`.

## Series

| Serie | Variable | Frecuencia | Obs. train | Predicciones |
|-------|----------|------------|------------|--------------|
| A | Precio bursátil | Diaria (días hábiles) | 2025 | 10 días |
| B | Nacimientos diarios | Diaria | 3287 | 365 días (año 2003) |
| C | Temperatura global (°C) | Mensual | 1740 | 12 meses (2025) |
| D | Serie mensual | Mensual | 132 | 10 meses (Ene–Oct 1991) |

## Modelos seleccionados

| Serie | Modelo (submission) | RMSE val. |
|-------|---------------------|-----------|
| A | SARIMA(2,0,1)(1,1,1,5) | 6.16 |
| B | SARIMA(2,0,1)(1,1,1,7) + Fourier K=5 | 710.10 |
| C | SARIMA(1,0,0)(0,1,1,12) + tendencia lineal (últimos 40 años) | 0.287 |
| D | SARIMA(0,0,0)(1,0,0,12) | 186.24 |

Puntuación Kaggle: **0.78** (primer puesto del leaderboard).

## Estructura

```
.
├── main.py                      # Ejecuta todas las series y genera submission.csv
├── serie_a.py                   # Serie A — precios bursátiles (Holt 252d)
├── serie_b.py                   # Serie B — nacimientos diarios (HistGBM)
├── serie_c.py                   # Serie C — temperatura global (STL+ETS 40y)
├── serie_d.py                   # Serie D — serie mensual (ETS(A,A,A))
├── helpers.py                   # Funciones auxiliares (ADF, plots, RMSE, guardado)
├── investigacion_adicional.py   # Experimentos y modelos alternativos explorados
├── validar_gbm_ridge.py         # Validación rolling GBM (B) y Ridge (C)
├── train_series_A/B/C/D.csv     # Datos de entrenamiento
├── test_serie_A/B/C/D.csv       # Períodos a predecir
├── pred_A/B/C/D.csv             # Predicciones por serie (generados)
├── submission.csv               # Fichero final de entrega (generado)
├── figures/                     # Gráficos generados
├── informe.tex                  # Informe en LaTeX
└── informe.pdf                  # Informe compilado
```

## Uso

```bash
# Ejecutar todas las series y generar submission.csv
python3 main.py

# Ejecutar solo una serie
python3 main.py --only B   # A | B | C | D

# Solo combinar pred_*.csv ya existentes
python3 main.py --combine
```

Cada serie se ejecuta como subproceso independiente para evitar acumulación de memoria. Si `pred_X.csv` ya existe, esa serie se omite automáticamente.

## Dependencias

```bash
pip install pandas numpy matplotlib statsmodels pmdarima scikit-learn prophet
```

Versión de Python: 3.10+. Se requiere `pdflatex` para recompilar el informe.

## Notas metodológicas

- **Serie A**: SARIMA(2,0,1)(1,1,1,5) con m=5 días laborables. d=0, D=1.
- **Serie B**: SARIMA(2,0,1)(1,1,1,7) con términos de Fourier anuales K=5 como regresores externos. m=7 captura la estacionalidad semanal; el Fourier captura la anual sin fuga de datos.
- **Serie C**: SARIMA(1,0,0)(0,1,1,12) con regresor de tendencia lineal, entrenado solo sobre los últimos 40 años (1985-2024). Los datos de 1880 distorsionan la predicción porque la tasa de calentamiento reciente es muy distinta.
- **Serie D**: SARIMA(0,0,0)(1,0,0,12): modelo AR estacional puro, el más simple y el mejor en validación.
