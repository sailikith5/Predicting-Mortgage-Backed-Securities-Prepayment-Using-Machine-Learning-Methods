"""Create tiny FAKE Freddie-format files + macro for a dry run (not real data).
  python tests/make_synthetic.py
  python src/data_prep.py --raw data/raw_synth --macro data/macro_synth.csv
"""
import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
os.makedirs("data/raw_synth", exist_ok=True)
N = 4000
orig_rows, perf_rows = [], []
for i in range(N):
    lid = f"F05Q1{i:07d}"
    rate = rng.normal(5.8, 0.6); upb0 = float(rng.integers(80, 400) * 1000)
    fico = int(rng.normal(740, 40))
    start = pd.Timestamp("2005-02-01") + pd.DateOffset(months=int(rng.integers(0, 24)))
    orig_rows.append([fico, start.strftime("%Y%m"), "N", "203501", "", 0, 1, "P", 80, 35, upb0, 80,
                      round(rate, 3), "R", "N", "FRM", "CA", "SF", "90000", lid, "P", 360, 2, "X", "Y",
                      "N", "", "F", "N", "9", "N", "7"])
    L = int(rng.integers(6, 60)); upb = upb0
    for t in range(L):
        per = (start + pd.DateOffset(months=t)).strftime("%Y%m")
        end = (t == L - 1) and rng.random() < 0.35 + 0.3 * (rate > 6.0)
        if end:
            perf_rows.append([lid, per, "", "0", t, 360 - t, "", "N", "01", per, round(rate, 3)] + [""] * 19)
            break
        perf_rows.append([lid, per, round(upb, 2), "0", t, 360 - t, "", "N", "", "", round(rate, 3)] + [""] * 19)
        upb *= 0.997
pd.DataFrame(orig_rows).to_csv("data/raw_synth/sample_orig_2005.txt", sep="|", header=False, index=False)
pd.DataFrame(perf_rows).to_csv("data/raw_synth/sample_svcg_2005.txt", sep="|", header=False, index=False)
idx = pd.date_range("2000-01-01", "2012-12-01", freq="MS")
pd.DataFrame({"period": idx, "mkt_rate": 5 + rng.normal(0, .3, len(idx)),
              "unemp": 6 + rng.normal(0, .3, len(idx)), "hpi": 100.0,
              "hpa_yoy": 2 + rng.normal(0, .5, len(idx))}).to_csv("data/macro_synth.csv", index=False)
print("synthetic data written")
