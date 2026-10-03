"""Download macro series from FRED and save data/macro.csv (monthly).
mkt_rate : 30y fixed mortgage rate (Freddie Mac PMMS, MORTGAGE30US)
unemp    : unemployment rate (BLS, UNRATE)
hpi      : All-transactions house price index (FHFA, USSTHPI) -> hpa_yoy (% YoY)
"""
import os
import pandas as pd

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def fred(series_id):
    df = pd.read_csv(URL.format(series_id))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"]


def main():
    mkt = fred("MORTGAGE30US").resample("MS").mean()
    unemp = fred("UNRATE").resample("MS").mean()
    hpi = fred("USSTHPI").resample("MS").ffill()
    mac = pd.concat({"mkt_rate": mkt, "unemp": unemp, "hpi": hpi}, axis=1)
    mac["hpi"] = mac["hpi"].ffill()
    mac["hpa_yoy"] = mac["hpi"].pct_change(12) * 100
    mac = mac.dropna()
    mac.index.name = "period"
    os.makedirs("data", exist_ok=True)
    mac.to_csv("data/macro.csv")
    print(mac.tail(), "\nsaved data/macro.csv", mac.shape)


if __name__ == "__main__":
    main()
