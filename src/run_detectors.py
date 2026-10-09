import pandas as pd
from common import load_data, DATA_DIR
from detect_invoice_mismatch import detect_invoice_mismatch
from detect_shell import detect_shell
from detect_bid_rigging import detect_bid_rigging
from detect_price_deviation import detect_price_deviation
from detect_delivery_timing import detect_delivery_timing

DETECTORS = [detect_invoice_mismatch, detect_shell, detect_bid_rigging,
             detect_price_deviation, detect_delivery_timing]


def run_all(data):
    parts = [d(data) for d in DETECTORS]
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    data = load_data()
    signals = run_all(data)
    signals.to_csv(DATA_DIR / "signals.csv", index=False)

    print(f"Total signals: {len(signals)}\n")
    print(signals.groupby("signal_type").size().to_string(), "\n")

    summary = (signals.groupby("supplier_id")["signal_type"]
               .apply(lambda s: ", ".join(sorted(s)))
               .reset_index())
    print(f"{len(summary)} suppliers with at least one signal:\n")
    for _, r in summary.iterrows():
        print(f"  Supplier {r.supplier_id}: {r.signal_type}")