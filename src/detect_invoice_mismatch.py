import pandas as pd
from common import load_data, SIGNAL_COLUMNS

TOLERANCE = 0.05     # ignore differences under 5%
MIN_FLAGGED = 2      # one-off slips are not a pattern


def detect_invoice_mismatch(data):
    df = data["purchase_orders"].merge(data["invoices"], on="po_id")

    df["qty_excess"] = (df["invoiced_qty"] - df["quantity"]) / df["quantity"]
    df["price_excess"] = (df["invoiced_unit_price"] - df["unit_price"]) / df["unit_price"]
    df["excess_amount"] = df["invoice_amount"] - df["quantity"] * df["unit_price"]
    df["flag"] = (df["qty_excess"] > TOLERANCE) | (df["price_excess"] > TOLERANCE)

    rows = []
    for seller, g in df.groupby("seller_id"):
        bad = g[g["flag"]]
        if len(bad) < MIN_FLAGGED:
            continue
        share = len(bad) / len(g)
        if share < 0.20:
            continue
        avg_over = bad[["qty_excess", "price_excess"]].max(axis=1).mean()
        evidence = (f"{len(bad)} of {len(g)} purchase orders ({share:.0%}) were invoiced "
                    f"above the PO quantity or price, on average {avg_over:.0%} over; "
                    f"about {bad['excess_amount'].sum():,.0f} billed beyond what was ordered")
        rows.append({"supplier_id": seller,
                    "signal_type": "invoice_po_mismatch",
                     "strength": round(min(1.0, share * 2), 2),
                    "evidence": evidence})
    return pd.DataFrame(rows, columns=SIGNAL_COLUMNS)


if __name__ == "__main__":
    data = load_data()
    signals = detect_invoice_mismatch(data)

    print(f"{len(signals)} supplier(s) flagged\n")
    for _, r in signals.iterrows():
        print(f"Supplier {r.supplier_id} | strength {r.strength}")
        print(f"  {r.evidence}\n")

    flagged = set(signals["supplier_id"])
    print("Expected flagged suppliers: {2}")
    print("PASS" if flagged == {2} else f"CHECK: got {flagged}")