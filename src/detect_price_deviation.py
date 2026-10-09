import pandas as pd
from common import load_data, SIGNAL_COLUMNS

DEVIATION = 0.15     # a PO priced 15%+ above the item's market median is "overpriced"
MIN_FLAGGED = 3      # a single spike is not a pattern
MIN_SHARE = 0.30     # at least 30% of the seller's POs must be overpriced


def detect_price_deviation(data):
    po = data["purchase_orders"].copy()

    # "Market price" per item = median unit price across all sellers
    po["market_price"] = po.groupby("item")["unit_price"].transform("median")
    po["dev"] = (po["unit_price"] - po["market_price"]) / po["market_price"]
    po["over"] = po["dev"] > DEVIATION
    po["extra_cost"] = (po["unit_price"] - po["market_price"]) * po["quantity"]

    rows = []
    for seller, g in po.groupby("seller_id"):
        bad = g[g["over"]]
        share = len(bad) / len(g)
        if len(bad) < MIN_FLAGGED or share < MIN_SHARE:
            continue
        evidence = (f"{len(bad)} of {len(g)} purchase orders ({share:.0%}) were priced "
                    f"above the market median for the same item, on average "
                    f"{bad['dev'].mean():.0%} higher; about {bad['extra_cost'].sum():,.0f} "
                    f"paid above market")
        rows.append({"supplier_id": seller,
                     "signal_type": "price_deviation",
                     "strength": round(min(1.0, bad["dev"].mean() * 1.5 * share + 0.2), 2),
                     "evidence": evidence})
    return pd.DataFrame(rows, columns=SIGNAL_COLUMNS)


if __name__ == "__main__":
    data = load_data()
    sig = detect_price_deviation(data)

    print(f"{len(sig)} supplier(s) flagged\n")
    for _, r in sig.iterrows():
        print(f"Supplier {r.supplier_id} | strength {r.strength}")
        print(f"  {r.evidence}\n")

    flagged = set(sig["supplier_id"])
    print("Expected flagged: {30, 41, 52, 12} (Case 1 shells and their Tier-2 buyer)")
    print("Must NOT be flagged: 3 (one price spike)")
    print("Case 1 shells found:", "PASS" if {30, 41, 52} <= flagged else
          f"CHECK missing {({30, 41, 52} - flagged)}")
    print("Supplier 12 found:", "PASS" if 12 in flagged else "CHECK: 12 not flagged")
    print("Decoy 3 quiet:", "PASS" if 3 not in flagged else "CHECK: decoy 3 flagged")
    print("Other flagged suppliers:", sorted(flagged - {30, 41, 52, 12}))