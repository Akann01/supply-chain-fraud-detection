import pandas as pd
from itertools import combinations
from collections import Counter
from common import load_data, DATA_DIR
from run_detectors import run_all
from fusion import supplier_risk, noisy_or

W_UP, W_DOWN = 0.5, 0.2     # supply links: from its suppliers / from its buyers
W_BANK, W_ADDR = 0.8, 0.3   # shared bank account is strong, shared address weak
W_COBID = 0.35              # weak: honest suppliers also bid on the same tenders
MIN_COBID = 4               # must co-bid at least this many times to count
FLAG_AT = 0.35


def cobid_counts(data):
    cob = Counter()
    for _, g in data["bids"].groupby("tender_id"):
        for a, c in combinations(sorted(g["supplier_id"].astype(int).unique()), 2):
            cob[(a, c)] += 1
    return cob


def build_edges(data):
    sup = data["suppliers"]
    edges = {int(n): [] for n in sup["supplier_id"]}
    for s, b in data["supply_links"][["supplier_id", "buyer_id"]].itertuples(index=False):
        s, b = int(s), int(b)
        if s == 0 or b == 0:
            continue
        edges[b].append((s, W_UP, "buys from"))
        edges[s].append((b, W_DOWN, "sells to"))
    for col, w, text in [("bank_account", W_BANK, "shares a bank account with"),
                         ("address", W_ADDR, "shares an address with")]:
        for _, g in sup.groupby(col):
            ids = g["supplier_id"].astype(int).tolist()
            for a in ids:
                for c in ids:
                    if a != c:
                        edges[a].append((c, w, text))
    for (a, c), n in cobid_counts(data).items():
        if n >= MIN_COBID:
            edges[a].append((c, W_COBID, f"co-bids in {n} tenders with"))
            edges[c].append((a, W_COBID, f"co-bids in {n} tenders with"))
    return edges


def predict(edges, base, hidden=()):
    own = {n: (0.0 if n in hidden else base.get(n, 0.0)) for n in edges}
    rows = []
    for n in edges:
        notes = []
        for m, w, text in edges[n]:
            if own[m] >= FLAG_AT:                      # only flagged suppliers push risk
                notes.append((w * own[m], f"{text} flagged supplier {m}"))
        notes.sort(reverse=True)
        best = notes[0][0] if notes else 0.0           # strongest single link, no stacking
        rows.append({"supplier_id": n, "own_risk": round(own[n], 2),
                     "predicted_risk": round(noisy_or([own[n], best]), 2),
                     "reason": "; ".join(t for _, t in notes[:3]) if own[n] < FLAG_AT else ""})
    return pd.DataFrame(rows).sort_values("predicted_risk", ascending=False)


if __name__ == "__main__":
    data = load_data()
    risk = supplier_risk(run_all(data))
    base = dict(zip(risk.supplier_id, risk.risk))
    edges = build_edges(data)

    pred = predict(edges, base)
    pred.to_csv(DATA_DIR / "predicted_risk.csv", index=False)
    unflagged = pred[(pred.own_risk < FLAG_AT) & (pred.predicted_risk > 0)]
    print("UNFLAGGED SUPPLIERS WITH ELEVATED PREDICTED RISK (top 12)")
    print(unflagged.head(12).to_string(index=False))

    p = pred.set_index("supplier_id")["predicted_risk"]
    print("\nDecoys (should stay below", FLAG_AT, "):",
          {d: float(p.get(d, 0.0)) for d in (3, 5, 40, 55)})

    print("\nHIDE-AND-RECOVER TEST (rank among suppliers not otherwise flagged)")
    for s in [52, 30, 25, 15, 18, 2]:
        q = predict(edges, base, hidden={s}).set_index("supplier_id")["predicted_risk"]
        pool = [n for n in q.index if n == s or base.get(n, 0.0) < FLAG_AT]
        rank = q[pool].rank(ascending=False, method="average")[s]
        print(f"  Hide {s}: predicted risk {q[s]}, rank {rank:.1f} of {len(pool)}")

    print("\nCO-BID PAIRS USED AS EDGES")
    for (a, c), n in sorted(cobid_counts(data).items(), key=lambda x: -x[1]):
        if n >= MIN_COBID:
            print(f"  {a} & {c}: {n} tenders")