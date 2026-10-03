"""Merge Freddie Mac origination + performance (+ macro) into one loan-month table.

Label: label=1 if the loan prepaid (zero balance code 01) in that month, else 0.
Rows of loans that ended for other reasons (default etc.) are dropped at their last row.
Split is by loan_id (60/20/20) so a loan never appears in two splits.

Supports both file layouts under data/raw (searched recursively):
  sample_orig_YYYY.txt        + sample_svcg_YYYY.txt (or sample_perf_YYYY.txt)
  historical_data_YYYYQn.txt  + historical_data_time_YYYYQn.txt
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

ORIG_COLS = [
    "credit_score", "first_payment_date", "first_time_homebuyer", "maturity_date", "msa",
    "mi_pct", "num_units", "occupancy_status", "orig_cltv", "orig_dti", "orig_upb",
    "orig_ltv", "orig_interest_rate", "channel", "ppm_flag", "amortization_type",
    "property_state", "property_type", "postal_code", "loan_id", "loan_purpose",
    "orig_loan_term", "num_borrowers", "seller_name", "servicer_name", "super_conforming",
    "pre_harp_loan_id", "program_indicator", "harp_indicator", "property_valuation",
    "io_indicator", "mi_cancellation",
]
ORIG_KEEP = [
    "loan_id", "credit_score", "first_time_homebuyer", "mi_pct", "num_units",
    "occupancy_status", "orig_cltv", "orig_dti", "orig_upb", "orig_ltv",
    "orig_interest_rate", "channel", "ppm_flag", "property_state", "property_type",
    "loan_purpose", "orig_loan_term", "num_borrowers",
]
# position in performance file -> name
PERF_USE = {0: "loan_id", 1: "period", 2: "current_upb", 3: "delinq", 4: "loan_age",
            5: "remaining_months", 7: "mod_flag", 8: "zb_code", 10: "current_rate"}
CAT = ["first_time_homebuyer", "occupancy_status", "channel", "ppm_flag",
       "property_state", "property_type", "loan_purpose"]
SENTINELS = {"credit_score": 9999, "mi_pct": 999, "num_units": 99, "orig_cltv": 999,
             "orig_dti": 999, "orig_ltv": 999, "num_borrowers": 99}


def find_pairs(raw_dir):
    pairs = []
    # sample dataset: performance file is named sample_svcg_YYYY.txt or sample_perf_YYYY.txt
    for o in glob.glob(os.path.join(raw_dir, "**", "sample_orig_*.txt"), recursive=True):
        for tag in ("sample_svcg_", "sample_perf_"):
            s = os.path.join(os.path.dirname(o), os.path.basename(o).replace("sample_orig_", tag))
            if os.path.exists(s):
                pairs.append((o, s))
                break
    for s in glob.glob(os.path.join(raw_dir, "**", "historical_data_time_*.txt"), recursive=True):
        o = s.replace("historical_data_time_", "historical_data_")
        if os.path.exists(o):
            pairs.append((o, s))
    return sorted(pairs)


def build_pair(o, s, n_loans, seed):
    # Freddie Mac added columns over the years (older files have 31 fields, newer 32+): size the names per file
    with open(o, "r") as f:
        n_fields = len(f.readline().rstrip("\n").split("|"))
    names = (ORIG_COLS + [f"extra_{i}" for i in range(n_fields - len(ORIG_COLS))])[:n_fields]
    orig = pd.read_csv(o, sep="|", header=None, names=names, usecols=ORIG_KEEP,
                       index_col=False, dtype={"loan_id": str}, low_memory=False)
    orig = orig.sample(min(n_loans, len(orig)), random_state=seed)
    ids = set(orig["loan_id"])
    parts = []
    reader = pd.read_csv(s, sep="|", header=None, names=list(PERF_USE.values()),
                         usecols=list(PERF_USE.keys()), index_col=False,
                         dtype={"loan_id": str, "delinq": str, "mod_flag": str},
                         chunksize=2_000_000)
    for ch in reader:
        parts.append(ch[ch["loan_id"].isin(ids)])
    perf = pd.concat(parts)
    perf["period"] = pd.to_datetime(perf["period"].astype(int).astype(str), format="%Y%m")
    perf["zb_code"] = pd.to_numeric(perf["zb_code"], errors="coerce")
    perf = perf.sort_values(["loan_id", "period"])
    perf["prev_upb"] = perf.groupby("loan_id")["current_upb"].shift(1)
    perf["label"] = (perf["zb_code"] == 1).astype(np.int8)
    perf = perf[perf["zb_code"].isna() | (perf["zb_code"] == 1)]
    return perf.merge(orig, on="loan_id", how="inner")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--macro", default="data/macro.csv")
    ap.add_argument("--out", default="data/processed/data.parquet")
    ap.add_argument("--loans-per-file", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=["loan", "vintage"], default="loan",
                    help="loan: random 60/20/20 by loan ID (paper). "
                         "vintage: oldest 60%% of loans -> train, next 20%% -> val, newest 20%% -> test "
                         "(out-of-time test, no loan or origination period shared across splits)")
    a = ap.parse_args()

    pairs = find_pairs(a.raw)
    if not pairs:
        raise SystemExit(f"No Freddie Mac file pairs found under {a.raw}")
    frames = []
    for i, (o, s) in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] {os.path.basename(o)}")
        frames.append(build_pair(o, s, a.loans_per_file, a.seed + i))
    df = pd.concat(frames, ignore_index=True)

    for c, v in SENTINELS.items():
        df.loc[df[c] == v, c] = np.nan

    # balance features (current_upb is the paper's feature; prev_upb is the leak-free variant)
    df["prev_upb"] = df["prev_upb"].fillna(df["orig_upb"])
    df["current_upb"] = df["current_upb"].fillna(0)
    df["upb_ratio_prev"] = df["prev_upb"] / df["orig_upb"]
    df["upb_ratio_cur"] = df["current_upb"] / df["orig_upb"]
    df["delinq_num"] = pd.to_numeric(df["delinq"], errors="coerce").fillna(0)
    df["modified"] = (df["mod_flag"] == "Y").astype(int)

    mac = pd.read_csv(a.macro, parse_dates=["period"]).sort_values("period")
    mac["mkt_rate_chg3"] = mac["mkt_rate"].diff(3)          # recent rate momentum (info known at month t)
    df = df.merge(mac, on="period", how="left").dropna(subset=["mkt_rate", "unemp", "hpa_yoy"])
    df["mkt_rate_chg3"] = df["mkt_rate_chg3"].fillna(0)
    df["rate_incentive_orig"] = df["orig_interest_rate"] - df["mkt_rate"]
    df["rate_incentive_cur"] = df["current_rate"] - df["mkt_rate"]
    df["refi_ratio"] = df["current_rate"] / df["mkt_rate"].where(df["mkt_rate"] > 0)  # >1 => refinancing is attractive
    month = df["period"].dt.month                            # seasonality of home sales / prepayments
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    df = df.drop(columns=["delinq", "mod_flag", "zb_code"])
    df = pd.get_dummies(df, columns=CAT, drop_first=True, dtype="int8")

    if a.split == "vintage":
        first = df.groupby("loan_id")["period"].min().sort_values(kind="stable")
        rank = np.arange(len(first)) / len(first)
        mp = pd.Series(np.where(rank < 0.6, "train", np.where(rank < 0.8, "val", "test")), index=first.index)
    else:
        rng = np.random.default_rng(a.seed)
        ids = df["loan_id"].unique()
        r = rng.random(len(ids))
        mp = pd.Series(np.where(r < 0.6, "train", np.where(r < 0.8, "val", "test")), index=ids)
    df["split"] = df["loan_id"].map(mp)

    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_parquet(a.out, index=False)
    print("rows", len(df), "loans", df["loan_id"].nunique(), "model features (leak-free set)",
          df.shape[1] - 6, "prepay rate", round(df["label"].mean(), 5))
    print(df.groupby("split")["label"].agg(["size", "sum"]))
    print("saved", a.out)


if __name__ == "__main__":
    main()
