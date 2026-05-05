"""
Ejecuta todas las series y genera el submission final.

Uso:
  python main.py          # corre todo (A, B, C, D) y combina
  python main.py --only A # corre solo la serie indicada (A|B|C|D)
  python main.py --combine # solo combina pred_*.csv ya existentes
"""

import argparse
import os
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

SERIES = ["A", "B", "C", "D"]
SCRIPTS = {s: os.path.join(DATA_DIR, f"serie_{s.lower()}.py") for s in SERIES}
PRED_FILES = {s: os.path.join(DATA_DIR, f"pred_{s}.csv") for s in SERIES}


def run_series(series):
    import subprocess, sys
    pred_path = PRED_FILES[series]
    if os.path.exists(pred_path):
        print(f"\n{'='*60}")
        print(f"Serie {series}: pred_{series}.csv ya existe, se omite.")
        print(f"{'='*60}")
        return
    print(f"\n{'='*60}")
    print(f"Ejecutando serie_{series.lower()}.py ...")
    print(f"{'='*60}")
    subprocess.run([sys.executable, SCRIPTS[series]], check=True)


def combine():
    frames = []
    for s in SERIES:
        path = PRED_FILES[s]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Falta {path}. Ejecuta primero serie_{s.lower()}.py")
        frames.append(pd.read_csv(path))
    submission = pd.concat(frames, ignore_index=True)
    out = os.path.join(DATA_DIR, "submission.csv")
    submission.to_csv(out, index=False)
    print(f"\nSubmission guardado en: {out}  ({len(submission)} filas)")
    print(submission.groupby(submission["id"].str[0])["id"].count().rename("predicciones").to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=SERIES, default=None,
                        help="Ejecutar solo esta serie")
    parser.add_argument("--combine", action="store_true",
                        help="Solo combinar pred_*.csv existentes")
    args = parser.parse_args()

    if args.combine:
        combine()
    elif args.only:
        run_series(args.only)
        combine()
    else:
        for s in SERIES:
            run_series(s)
        combine()
