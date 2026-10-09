import itertools
import pandas as pd
from common import load_data, SIGNAL_COLUMNS

CLOSE_BID_PCT = 0.03    # bids within 3% of each other count as "near-identical"
MIN_TENDERS = 4        # a pair must be close in at least this many tenders


def detect_bid_rigging(data):
    bids = data["bids"]
    tenders = data["tenders"].set_index("tender_id")

    # Step 1: find pairs of suppliers whose bids are near-identical in the same tender
    pair_tenders = {}
    for tid, g in bids.groupby("tender_id"):
        for (_, a), (_, b) in itertools.combinations(g.iterrows(), 2):
            gap = abs(a["bid_amount"] - b["bid_amount"]) / max(a["bid_amount"], b["bid_amount"])
            if gap <= CLOSE_BID_PCT:
                key = tuple(sorted((int(a["supplier_id"]), int(b["supplier_id"]))))
                pair_tenders.setdefault(key, set()).add(tid)

    # Step 2: keep only pairs that repeat across several tenders
    strong = {k: v for k, v in pair_tenders.items() if len(v) >= MIN_TENDERS}
    if not strong:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

    # Step 3: merge overlapping pairs into rings (connected components)
    rings = []
    for pair in strong:
        merged = set(pair)
        rings = [r for r in rings if not (r & merged) or merged.update(r)]
        rings.append(merged)

    rows = []
    for ring in rings:
        ring_tenders = set()
        for pair, tids in strong.items():
            if set(pair) <= ring:
                ring_tenders |= tids
        ring_bids = bids[bids["tender_id"].isin(ring_tenders) & bids["supplier_id"].isin(ring)]
        winners = ring_bids[ring_bids["won"]]["supplier_id"].tolist()
        n_winners = len(set(winners))
        issuers = tenders.loc[sorted(ring_tenders), "issuer_id"].unique().tolist()
        members = sorted(ring)

        # rotation: how evenly the wins are spread among ring members
        rotation = n_winners / len(ring)
        strength = round(min(1.0, 0.5 + 0.3 * rotation + 0.02 * len(ring_tenders)), 2)

        for s in members:
            evidence = (f"Bids were within {CLOSE_BID_PCT:.0%} of suppliers "
                        f"{[m for m in members if m != s]} in {len(ring_tenders)} tenders "
                        f"(issued by supplier(s) {issuers}); the win rotated across "
                        f"{n_winners} of {len(ring)} ring members")
            rows.append({"supplier_id": s, "signal_type": "bid_rigging",
                         "strength": strength, "evidence": evidence})

    return pd.DataFrame(rows, columns=SIGNAL_COLUMNS)


if __name__ == "__main__":
    data = load_data()
    sig = detect_bid_rigging(data)

    print(f"{len(sig)} signal(s)\n")
    for _, r in sig.iterrows():
        print(f"Supplier {r.supplier_id} | strength {r.strength}")
        print(f"  {r.evidence}\n")

    flagged = set(sig["supplier_id"])
    print("Expected flagged suppliers: {15, 20, 25}")
    print("PASS" if flagged == {15, 20, 25} else f"CHECK: got {flagged}")