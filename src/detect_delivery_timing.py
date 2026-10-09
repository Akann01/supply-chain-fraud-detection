import pandas as pd
from common import load_data, SIGNAL_COLUMNS

MIN_PREMATURE = 2    # invoices dated before delivery
LATE_DAYS = 7
MIN_LATE = 3         # deliveries after the expected date
MIN_LATE_SHARE = 0.30


def detect_delivery_timing(data):
    df = (data["purchase_orders"]
          .merge(data["deliveries"], on="po_id")
          .merge(data["invoices"], on="po_id"))
    for c in ["expected_delivery", "delivered_date", "invoice_date"]:
        df[c] = pd.to_datetime(df[c])

    df["premature"] = df["invoice_date"] < df["delivered_date"]
    df["late_days"] = (df["delivered_date"] - df["expected_delivery"]).dt.days
    df["late"] = df["late_days"] > LATE_DAYS
    df["gap_days"] = (df["delivered_date"] - df["invoice_date"]).dt.days

    rows = []
    for seller, g in df.groupby("seller_id"):
        prem = g[g["premature"]]
        late = g[g["late"]]
        late_share = len(late) / len(g)

        if len(prem) >= MIN_PREMATURE:
            rows.append({
                "supplier_id": seller, "signal_type": "invoice_before_delivery",
                "strength": round(min(1.0, 0.5 + 0.1 * len(prem)), 2),
                "evidence": (f"{len(prem)} of {len(g)} invoices were dated before the goods "
                            f"arrived (up to {prem['gap_days'].max()} days early)")})

        if len(late) >= MIN_LATE and late_share >= MIN_LATE_SHARE:
            rows.append({
                "supplier_id": seller, "signal_type": "late_delivery",
                "strength": round(min(1.0, 0.3 + late_share), 2),
                "evidence": (f"{len(late)} of {len(g)} deliveries ({late_share:.0%}) arrived more than "
                        f"{LATE_DAYS} days late, on average {late['late_days'].mean():.0f} days "
                        f"after the expected date")})

    return pd.DataFrame(rows, columns=SIGNAL_COLUMNS)


if __name__ == "__main__":
    data = load_data()
    sig = detect_delivery_timing(data)

    print(f"{len(sig)} signal(s)\n")
    for _, r in sig.iterrows():
        print(f"Supplier {r.supplier_id} | {r.signal_type} | strength {r.strength}")
        print(f"  {r.evidence}\n")

    flagged = set(sig["supplier_id"])
    prem = set(sig[sig.signal_type == "invoice_before_delivery"].supplier_id)
    print("Supplier 18 invoice-before-delivery:", "PASS" if 18 in prem else "CHECK: missing")
    print("Supplier 18 flagged at all:", "PASS" if 18 in flagged else "CHECK: missing")
    print("Decoy 5 quiet:", "PASS" if 5 not in flagged else "CHECK: decoy 5 flagged")
    print("Other flagged suppliers:", sorted(flagged - {18}))