"""
sarimax_v2.py — Grid search SARIMAX reducido (~15 min).

Diferencias respecto a v1:
  - d=0,1 solamente (d=2 casi nunca ayuda)
  - p,q max=2, P,Q max=1 (cubre el 95% de los casos útiles)
  - C: solo ventana 40y (la que sabemos que funciona)
  - Siempre genera submission_sarimax.csv con lo mejor encontrado
  - Output con flush inmediato para monitorear progreso

Ejecutar con:
  python3 -u sarimax_v2.py 2>&1 | tee sarimax_v2_resultados.txt

Benchmarks actuales (Kaggle 0.76, meta: bajar de 0.82):
  A=6.60  B=514.16  C=0.3818  D=234.91
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os, gc, itertools, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats

from helpers import DATA_DIR, FIG_DIR, rmse

def pr(*args):
    print(*args, flush=True)


# ── Grids reducidos ───────────────────────────────────────────
# d=0,1 | p,q=0..2 | P,Q=0..1 | max_pq=3 | max_PQ=2
# → 8 combos (p,q) × 2d × 4 combos (P,Q) × 2D = 128 modelos
# ─────────────────────────────────────────────────────────────

BENCH = {"A": 6.60, "B": 514.16, "C": 0.3818, "D": 234.91}
best_sarimax = {}  # serie -> {"rmse", "pred", "order"}


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
    return [(p,d,q,P,D,Q)
            for p,d,q,P,D,Q in itertools.product(p_r,d_r,q_r,P_r,D_r,Q_r)
            if p+q <= max_pq and P+Q <= max_PQ]


def run_grid(y_tr, y_val, combos, m_s, n_fc, exog_tr=None, exog_fc=None, tag=""):
    rows, N, t0 = [], len(combos), time.time()
    for i, (p,d,q,P,D,Q) in enumerate(combos):
        if (i+1) % 20 == 0:
            pr(f"    [{tag}] {i+1}/{N}  ({time.time()-t0:.0f}s)")
        pred, fit = fit_safe(y_tr, (p,d,q), (P,D,Q,m_s), n_fc,
                             exog_tr=exog_tr, exog_fc=exog_fc)
        if pred is None:
            continue
        rows.append(dict(p=p,d=d,q=q,P=P,D=D,Q=Q,
                         rmse=rmse(y_val, pred),
                         aic=getattr(fit,"aic",np.nan)))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def fourier_annual(idx, K):
    t = idx.dayofyear / 365.25
    cols = {f"sin{k}": np.sin(2*np.pi*k*t) for k in range(1, K+1)}
    cols.update({f"cos{k}": np.cos(2*np.pi*k*t) for k in range(1, K+1)})
    return pd.DataFrame(cols, index=idx).values


def residual_analysis(fit, label, path, m=12):
    resid = fit.resid.dropna()
    resid_plot = resid.iloc[-min(500, len(resid)):]
    max_lags = min(48, len(resid)//4)

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    ax = fig.add_subplot(gs[0,0])
    ax.plot(resid_plot.values, lw=0.6, color="steelblue")
    ax.axhline(0, color="r", lw=0.8, ls="--")
    ax.set_title(f"Residuos ({len(resid_plot)} obs)"); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0,1])
    ax.hist(resid, bins=40, density=True, alpha=0.7)
    xr = np.linspace(resid.min(), resid.max(), 200)
    ax.plot(xr, stats.norm.pdf(xr, resid.mean(), resid.std()), "r-", lw=2)
    ax.set_title("Histograma + Normal"); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0,2])
    stats.probplot(resid, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot"); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1,0])
    plot_acf(resid, lags=max_lags, ax=ax, title="ACF residuos")

    ax = fig.add_subplot(gs[1,1])
    plot_pacf(resid, lags=max_lags, ax=ax, method="ywm", title="PACF residuos")

    ax = fig.add_subplot(gs[1,2])
    ax.axis("off")
    lb_lags = sorted({m, 2*m, min(3*m, len(resid)//4)})
    lb_lags = [l for l in lb_lags if 2 <= l <= len(resid)//3]
    try:
        lb = acorr_ljungbox(resid, lags=lb_lags, return_df=True)
        txt = "Ljung-Box:\n"
        for lag, row in lb.iterrows():
            ok = "ok" if row["lb_pvalue"] > 0.05 else "FALLA"
            txt += f"  lag={int(lag):3d}  p={row['lb_pvalue']:.4f} {ok}\n"
    except Exception:
        txt = "Ljung-Box: no disponible\n"
    jb, jp = stats.jarque_bera(resid)
    txt += f"\nJarque-Bera p={jp:.4f} {'ok' if jp>0.05 else 'FALLA'}"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top",
            fontsize=9, fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    plt.suptitle(f"Residuos — {label}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    pr(f"  Residuos -> {path}")


GRID_SMALL = build_grid(range(3), [0,1], range(3), range(2), [0,1], range(2))
pr(f"Grid reducido: {len(GRID_SMALL)} combinaciones por serie")


# ──────────────────────────────────────────────────────────────
# SERIE A
# ──────────────────────────────────────────────────────────────
pr("\n" + "="*60)
pr("SERIE A — Grid SARIMAX (m=5)")
pr("="*60)
t_A = time.time()

df_A = pd.read_csv(os.path.join(DATA_DIR, "train_series_A.csv"), parse_dates=["Date"])
serie_A = df_A.sort_values("Date").set_index("Date")["value"].astype(float)
test_A  = pd.read_csv(os.path.join(DATA_DIR, "test_serie_A.csv"))
test_A["timestamp"] = pd.to_datetime(test_A["timestamp"].str.strip())
n_pred_A = len(test_A)

train_A, val_A = serie_A.iloc[:-20], serie_A.iloc[-20:]
pr(f"  {len(GRID_SMALL)} modelos | benchmark={BENCH['A']}")

df_Ar = run_grid(train_A, val_A.values, GRID_SMALL, 5, 20, tag="A")

if not df_Ar.empty:
    pr(f"\n  Top 10 Serie A ({time.time()-t_A:.0f}s):")
    pr(df_Ar.head(10).to_string())
    row = df_Ar.iloc[0]
    p,d,q,P,D,Q = int(row.p),int(row.d),int(row.q),int(row.P),int(row.D),int(row.Q)
    lbl_A = f"SARIMA({p},{d},{q})({P},{D},{Q},5)"
    pr(f"\n  Mejor A: {lbl_A}  RMSE={row.rmse:.4f}  bench={BENCH['A']}")

    _, fit_A_val   = fit_safe(train_A, (p,d,q), (P,D,Q,5), 20)
    pred_A_sub, _  = fit_safe(serie_A, (p,d,q), (P,D,Q,5), n_pred_A)
    if fit_A_val:
        residual_analysis(fit_A_val, lbl_A,
                          os.path.join(FIG_DIR, "sarimax_v2_residuos_A.png"), m=5)
    if pred_A_sub is not None:
        best_sarimax["A"] = {"rmse": row.rmse, "pred": pred_A_sub, "order": lbl_A}
        if row.rmse < BENCH["A"]:
            pr(f"  *** MEJORA en A ***")

del df_A, serie_A, train_A, val_A, df_Ar; gc.collect()


# ──────────────────────────────────────────────────────────────
# SERIE B
# ──────────────────────────────────────────────────────────────
pr("\n" + "="*60)
pr("SERIE B — Grid SARIMAX+Fourier (m=7)")
pr("="*60)

df_B = pd.read_csv(os.path.join(DATA_DIR, "train_series_B.csv"))
df_B["Date"] = pd.to_datetime(df_B[["year","month","day"]])
df_B = df_B.sort_values("Date").set_index("Date")
serie_B_full = df_B["births"].astype(float)
test_B = pd.read_csv(os.path.join(DATA_DIR, "test_serie_B.csv"))
test_B["Date"] = pd.to_datetime(test_B[["year","month","day"]])
pred_dates_B = pd.DatetimeIndex(test_B["Date"].values)
n_pred_B = len(test_B)

windows_B = {
    "4y": serie_B_full[serie_B_full.index.year >= 1999],
    "all": serie_B_full,
}

# Fase 1: grid con K=3, ventana 4y
t_B = time.time()
w4 = windows_B["4y"]
tr1, va1 = w4[w4.index.year < 2002], w4[w4.index.year == 2002]
pr(f"  Fase 1: {len(GRID_SMALL)} modelos con K=3, ventana 4y | bench={BENCH['B']}")

K3_tr, K3_va = fourier_annual(tr1.index, 3), fourier_annual(va1.index, 3)
df_B1 = run_grid(tr1, va1.values, GRID_SMALL, 7, len(va1),
                 exog_tr=K3_tr, exog_fc=K3_va, tag="B-f1")

top_structs = []
if not df_B1.empty:
    pr(f"\n  Top 10 Fase 1 ({time.time()-t_B:.0f}s):")
    pr(df_B1.head(10).to_string())
    top_structs = [tuple(df_B1.iloc[i][["p","d","q","P","D","Q"]].astype(int))
                   for i in range(min(5, len(df_B1)))]
del tr1, va1, K3_tr, K3_va; gc.collect()

# Fase 2: top-5 × K=3,4,5 × ventanas 4y,all
pr(f"\n  Fase 2: {len(top_structs)} estructuras × K=3,4,5 × 2 ventanas ({len(top_structs)*3*2} fits)")
all_B2 = []
for wname, wdata in windows_B.items():
    tr_w = wdata[wdata.index.year < 2002]
    va_w = wdata[wdata.index.year == 2002]
    for K in [3, 4, 5]:
        exog_tr_w = fourier_annual(tr_w.index, K)
        exog_va_w = fourier_annual(va_w.index, K)
        for (p,d,q,P,D,Q) in top_structs:
            pred, _ = fit_safe(tr_w, (p,d,q), (P,D,Q,7), len(va_w),
                               exog_tr=exog_tr_w, exog_fc=exog_va_w)
            if pred is None:
                continue
            all_B2.append(dict(p=p,d=d,q=q,P=P,D=D,Q=Q,K=K,window=wname,
                               rmse=rmse(va_w.values, pred)))
    del tr_w, va_w; gc.collect()

if all_B2:
    df_B2 = pd.DataFrame(all_B2).sort_values("rmse").reset_index(drop=True)
    pr(f"\n  Top 10 Fase 2 ({time.time()-t_B:.0f}s total):")
    pr(df_B2[["p","d","q","P","D","Q","K","window","rmse"]].head(10).to_string())

    best_B = df_B2.iloc[0]
    p,d,q = int(best_B.p),int(best_B.d),int(best_B.q)
    P,D,Q = int(best_B.P),int(best_B.D),int(best_B.Q)
    K_best, w_best = int(best_B.K), best_B.window
    lbl_B = f"SARIMA({p},{d},{q})({P},{D},{Q},7)+F{K_best}[{w_best}]"
    pr(f"\n  Mejor B: {lbl_B}  RMSE={best_B.rmse:.2f}  bench={BENCH['B']}")

    # refit sobre todos los datos (1994-2002) para predicciones finales
    exog_all  = fourier_annual(serie_B_full.index, K_best)
    exog_test = fourier_annual(pred_dates_B, K_best)
    pred_B_sub, _ = fit_safe(serie_B_full, (p,d,q), (P,D,Q,7), n_pred_B,
                              exog_tr=exog_all, exog_fc=exog_test)

    # residuos sobre ventana ganadora
    wdata_b  = windows_B[w_best]
    tr_b, va_b = wdata_b[wdata_b.index.year < 2002], wdata_b[wdata_b.index.year == 2002]
    _, fit_B_val = fit_safe(tr_b, (p,d,q), (P,D,Q,7), len(va_b),
                            exog_tr=fourier_annual(tr_b.index, K_best),
                            exog_fc=fourier_annual(va_b.index, K_best))
    if fit_B_val:
        residual_analysis(fit_B_val, lbl_B,
                          os.path.join(FIG_DIR, "sarimax_v2_residuos_B.png"), m=7)
    if pred_B_sub is not None:
        best_sarimax["B"] = {"rmse": best_B.rmse, "pred": pred_B_sub, "order": lbl_B}
        if best_B.rmse < BENCH["B"]:
            pr(f"  *** MEJORA en B ***")

del df_B, serie_B_full, all_B2; gc.collect()


# ──────────────────────────────────────────────────────────────
# SERIE C
# ──────────────────────────────────────────────────────────────
pr("\n" + "="*60)
pr("SERIE C — Grid SARIMAX (m=12, ventana 40y)")
pr("="*60)
t_C = time.time()

df_C_wide = pd.read_csv(os.path.join(DATA_DIR, "train_series_C.csv"))
mnames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
df_C_long = df_C_wide.melt(id_vars="Year", value_vars=mnames,
                            var_name="Month", value_name="temp")
df_C_long["Month_num"] = df_C_long["Month"].map({m: i+1 for i,m in enumerate(mnames)})
df_C_long["Date"] = pd.to_datetime(
    df_C_long["Year"].astype(str)+"-"+df_C_long["Month_num"].astype(str)+"-01")
serie_C_full = df_C_long.sort_values("Date").set_index("Date")["temp"].astype(float)
del df_C_wide, df_C_long; gc.collect()

serie_C40 = serie_C_full.iloc[-480:]
n_pred_C  = len(pd.read_csv(os.path.join(DATA_DIR, "test_serie_C.csv")))
train_C, val_C = serie_C40.iloc[:-24], serie_C40.iloc[-24:]
pr(f"  {len(GRID_SMALL)} modelos | bench={BENCH['C']}")

df_Cr = run_grid(train_C, val_C.values, GRID_SMALL, 12, 24, tag="C")

# también probar top-5 con tendencia lineal
all_C_extra = []
if not df_Cr.empty:
    pr(f"\n  Top 10 Serie C ({time.time()-t_C:.0f}s):")
    pr(df_Cr.head(10).to_string())
    n_tr_c = len(train_C)
    t_tr = np.arange(n_tr_c, dtype=float).reshape(-1,1)
    t_va = np.arange(n_tr_c, n_tr_c+24, dtype=float).reshape(-1,1)
    pr("  Probando top-5 con tendencia lineal exog...")
    for _, r_c in df_Cr.head(5).iterrows():
        p_,d_,q_ = int(r_c.p),int(r_c.d),int(r_c.q)
        P_,D_,Q_ = int(r_c.P),int(r_c.D),int(r_c.Q)
        pred_t, _ = fit_safe(train_C, (p_,d_,q_), (P_,D_,Q_,12), 24,
                             exog_tr=t_tr, exog_fc=t_va)
        if pred_t is None:
            continue
        all_C_extra.append(dict(p=p_,d=d_,q=q_,P=P_,D=D_,Q=Q_,
                                rmse=rmse(val_C.values, pred_t),
                                aic=np.nan, window="40y+trend"))

# combinar y elegir mejor
df_Cr["window"] = "40y"
rows_C = [df_Cr]
if all_C_extra:
    rows_C.append(pd.DataFrame(all_C_extra))
df_C_all = pd.concat(rows_C, ignore_index=True).sort_values("rmse").reset_index(drop=True)
pr(f"\n  Top 10 Serie C incluyendo tendencia:")
pr(df_C_all[["p","d","q","P","D","Q","window","rmse"]].head(10).to_string())

best_C  = df_C_all.iloc[0]
p,d,q   = int(best_C.p),int(best_C.d),int(best_C.q)
P,D,Q   = int(best_C.P),int(best_C.D),int(best_C.Q)
use_trend_C = "trend" in str(best_C.get("window",""))
lbl_C = f"SARIMA({p},{d},{q})({P},{D},{Q},12)[{'40y+trend' if use_trend_C else '40y'}]"
pr(f"\n  Mejor C: {lbl_C}  RMSE={best_C.rmse:.4f}  bench={BENCH['C']}")

n_w = len(serie_C40)
if use_trend_C:
    exog_tr_C  = np.arange(len(train_C), dtype=float).reshape(-1,1)
    exog_va_C  = np.arange(len(train_C), len(train_C)+24, dtype=float).reshape(-1,1)
    exog_full_C = np.arange(n_w, dtype=float).reshape(-1,1)
    exog_tst_C  = np.arange(n_w, n_w+12, dtype=float).reshape(-1,1)
else:
    exog_tr_C = exog_va_C = exog_full_C = exog_tst_C = None

_, fit_C_val   = fit_safe(train_C, (p,d,q), (P,D,Q,12), 24,
                           exog_tr=exog_tr_C, exog_fc=exog_va_C)
pred_C_sub, _  = fit_safe(serie_C40, (p,d,q), (P,D,Q,12), 12,
                           exog_tr=exog_full_C, exog_fc=exog_tst_C)
if fit_C_val:
    residual_analysis(fit_C_val, lbl_C,
                      os.path.join(FIG_DIR, "sarimax_v2_residuos_C.png"), m=12)
if pred_C_sub is not None:
    best_sarimax["C"] = {"rmse": best_C.rmse, "pred": pred_C_sub, "order": lbl_C}
    if best_C.rmse < BENCH["C"]:
        pr(f"  *** MEJORA en C ***")

del serie_C_full, serie_C40, train_C, val_C; gc.collect()


# ──────────────────────────────────────────────────────────────
# SERIE D
# ──────────────────────────────────────────────────────────────
pr("\n" + "="*60)
pr("SERIE D — Grid SARIMAX (m=12)")
pr("="*60)
t_D = time.time()

df_D = pd.read_csv(os.path.join(DATA_DIR, "train_series_D.csv"))
df_D["Date"] = pd.to_datetime(df_D["Date"], format="%d-%b-%Y")
serie_D = df_D.sort_values("Date").set_index("Date")["value"].astype(float)
test_D  = pd.read_csv(os.path.join(DATA_DIR, "test_series_D.csv"))
test_D["Date"] = pd.to_datetime(test_D[" Date"].str.strip(), format="%d-%b-%Y")
n_pred_D = len(test_D)
train_D, val_D = serie_D.iloc[:-12], serie_D.iloc[-12:]
pr(f"  {len(GRID_SMALL)} modelos | bench={BENCH['D']}")

df_Dr = run_grid(train_D, val_D.values, GRID_SMALL, 12, 12, tag="D")

if not df_Dr.empty:
    pr(f"\n  Top 10 Serie D ({time.time()-t_D:.0f}s):")
    pr(df_Dr.head(10).to_string())
    row = df_Dr.iloc[0]
    p,d,q,P,D,Q = int(row.p),int(row.d),int(row.q),int(row.P),int(row.D),int(row.Q)
    lbl_D = f"SARIMA({p},{d},{q})({P},{D},{Q},12)"
    pr(f"\n  Mejor D: {lbl_D}  RMSE={row.rmse:.2f}  bench={BENCH['D']}")

    _, fit_D_val   = fit_safe(train_D, (p,d,q), (P,D,Q,12), 12)
    pred_D_sub, _  = fit_safe(serie_D, (p,d,q), (P,D,Q,12), n_pred_D)
    if fit_D_val:
        residual_analysis(fit_D_val, lbl_D,
                          os.path.join(FIG_DIR, "sarimax_v2_residuos_D.png"), m=12)
    if pred_D_sub is not None:
        best_sarimax["D"] = {"rmse": row.rmse, "pred": pred_D_sub, "order": lbl_D}
        if row.rmse < BENCH["D"]:
            pr(f"  *** MEJORA en D ***")

del df_D, serie_D, train_D, val_D, df_Dr; gc.collect()


# ──────────────────────────────────────────────────────────────
# RESUMEN Y SUBMISSION
# ──────────────────────────────────────────────────────────────
pr("\n" + "="*60)
pr("RESUMEN FINAL")
pr("="*60)
pr(f"\n  {'Serie':<6} {'Benchmark':>12} {'SARIMAX v2':>12} {'Delta':>9}  Modelo")
pr("  " + "-"*70)
for s in ["A","B","C","D"]:
    bench = BENCH[s]
    if s in best_sarimax:
        sr    = best_sarimax[s]["rmse"]
        delta = (sr - bench) / bench * 100
        order = best_sarimax[s]["order"]
        mark  = " ***" if sr < bench else ""
    else:
        sr, delta, order, mark = float("nan"), float("nan"), "(sin resultado)", ""
    sr_s = f"{sr:.4f}" if np.isfinite(sr) else "  —  "
    dl_s = f"{delta:+.1f}%" if np.isfinite(delta) else "  —  "
    pr(f"  {s:<6} {bench:>12.4f} {sr_s:>12} {dl_s:>9}  {order}{mark}")

# Generar submission con lo mejor encontrado (meta: bajar de 0.82, no necesariamente de 0.76)
pr("\n  Generando submission_sarimax.csv con los mejores modelos SARIMAX...")
pred_files = {s: os.path.join(DATA_DIR, f"pred_{s}.csv") for s in ["A","B","C","D"]}
pred_lengths = {"A": 10, "B": 365, "C": 12, "D": 10}

frames = []
for s in ["A","B","C","D"]:
    if s in best_sarimax:
        ids = [f"{s}{i}" for i in range(1, len(best_sarimax[s]["pred"])+1)]
        df_out = pd.DataFrame({"id": ids, "value": best_sarimax[s]["pred"]})
        df_out.to_csv(os.path.join(DATA_DIR, f"pred_sarimax_{s}.csv"), index=False)
        pr(f"  Guardado pred_sarimax_{s}.csv  (RMSE={best_sarimax[s]['rmse']:.4f})")
        frames.append(df_out)
    else:
        pr(f"  Serie {s}: usando predicciones existentes (pred_{s}.csv)")
        frames.append(pd.read_csv(pred_files[s]))

sub = pd.concat(frames, ignore_index=True)
out_sub = os.path.join(DATA_DIR, "submission_sarimax.csv")
sub.to_csv(out_sub, index=False)
pr(f"\n  -> submission_sarimax.csv guardado ({len(sub)} filas)")
pr("  Sube a Kaggle y compara con el score actual (0.76).")
pr("  Meta: cualquier score < 0.82 es exito para demostrar que SARIMAX funciona.")

pr("\n" + "="*60)
pr("DONE")
pr("="*60)
