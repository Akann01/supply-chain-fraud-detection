import itertools
import pandas as pd
import networkx as nx
from common import load_data, DATA_DIR
from run_detectors import run_all
from detect_bid_rigging import CLOSE_BID_PCT

# How much each signal type counts toward risk. A shared address alone is weak.
WEIGHT = {"shared_bank_account": 0.9, "bid_rigging": 1.0, "price_deviation": 0.8,
          "invoice_po_mismatch": 1.0, "invoice_before_delivery": 1.0,
          "late_delivery": 0.6, "shared_address": 0.3, "new_registration": 0.5}
HIGH, MEDIUM = 0.70, 0.35


def noisy_or(values):
    p = 1.0
    for v in values:
        p *= (1 - v)
    return 1 - p


def level(r):
    return "HIGH" if r >= HIGH else "MEDIUM" if r >= MEDIUM else "LOW"


def supplier_risk(signals):
    s = signals.copy()
    s["eff"] = s["strength"] * s["signal_type"].map(WEIGHT)
    out = s.groupby("supplier_id").agg(
        risk=("eff", noisy_or),
        signals=("signal_type", lambda x: ", ".join(sorted(x)))).reset_index()
    out["risk"] = out["risk"].round(2)
    out["level"] = out["risk"].apply(level)
    return out.sort_values("risk", ascending=False)


def fuse(data, signals):
    sup = data["suppliers"].set_index("supplier_id")
    links = data["supply_links"]
    risk = supplier_risk(signals).set_index("supplier_id")["risk"].to_dict()
    sig_by = signals.groupby("supplier_id")["signal_type"].apply(set).to_dict()
    tier = sup["tier"].to_dict()
    hyps = []

    def add(kind, members, narrative):
        members = sorted(members)
        hyps.append({"hypothesis_id": f"H{len(hyps) + 1}", "root_cause": kind,
                     "suppliers": ",".join(map(str, members)),
                     "confidence": round(max(risk[m] for m in members), 2),
                     "explanation": narrative})

    # 1. Shell ring: suppliers sharing a bank account, plus what corroborates it
    bank_ids = {s for s, t in sig_by.items() if "shared_bank_account" in t}
    shell_members = set()
    if bank_ids:
        groups = (sup.loc[sorted(bank_ids)].groupby("bank_account").apply(
            lambda g: list(g.index), include_groups=False))
        for acct, members in groups.items():
            shell_members |= set(members)
            seen = set().union(*[sig_by[m] for m in members])
            extras = []
            if "shared_address" in seen:
                extras.append("some share a registered address")
            if "new_registration" in seen:
                extras.append("all are newly registered")
            if "price_deviation" in seen:
                extras.append("their prices are far above market")
            buyers = sorted(set(links[links.supplier_id.isin(members)].buyer_id)
                            - set(members))
            tiers = sorted({tier[m] for m in members})
            text = (f"Shell-company ring of {len(members)} suppliers "
                    f"(Tier {', '.join(map(str, tiers))}) sharing bank account {acct}")
            if extras:
                text += "; " + "; ".join(extras)
            if buyers:
                text += f". They sell to supplier(s) {buyers}"
            add("shell_ring", members, text)

    # 2. Bid-rigging cluster, with the tender issuer named
    br = {s for s, t in sig_by.items() if "bid_rigging" in t}
    if br:
        bids, tenders = data["bids"], data["tenders"].set_index("tender_id")
        g = nx.Graph()
        g.add_nodes_from(br)
        close_tenders = {}
        for tid, b in bids.groupby("tender_id"):
            rows = b[b.supplier_id.isin(br)]
            for (_, x), (_, y) in itertools.combinations(rows.iterrows(), 2):
                gap = abs(x.bid_amount - y.bid_amount) / max(x.bid_amount, y.bid_amount)
                if gap <= CLOSE_BID_PCT:
                    g.add_edge(x.supplier_id, y.supplier_id)
                    close_tenders.setdefault(x.supplier_id, set()).add(tid)
                    close_tenders.setdefault(y.supplier_id, set()).add(tid)
        for comp in nx.connected_components(g):
            tids = set().union(*[close_tenders.get(m, set()) for m in comp])
            issuers = sorted(int(i) for i in tenders.loc[sorted(tids), "issuer_id"].unique())
            tiers = sorted({tier[m] for m in comp})
            text = (f"Bid-rigging cluster of {len(comp)} suppliers "
                    f"(Tier {', '.join(map(str, tiers))}) with near-identical bids "
                    f"in {len(tids)} tenders issued by supplier(s) {issuers}")
            if sum("shared_address" in sig_by[m] for m in comp) >= 2:
                text += "; members also share a registered address"
            add("bid_rigging_cluster", comp, text)

    # 3. Over-billing and 4. timing manipulation: one supplier each
    for s, t in sig_by.items():
        if "invoice_po_mismatch" in t:
            ev = signals[(signals.supplier_id == s) &
                         (signals.signal_type == "invoice_po_mismatch")].evidence.iloc[0]
            add("over_billing", [s], f"Supplier {s} over-bills its buyer: {ev}")
        if t & {"invoice_before_delivery", "late_delivery"}:
            parts = [signals[(signals.supplier_id == s) & (signals.signal_type == x)]
                     .evidence.iloc[0] for x in sorted(t & {"invoice_before_delivery",
                                                            "late_delivery"})]
            add("timing_manipulation", [s],
                f"Supplier {s} shows delivery-timing manipulation: " + "; ".join(parts))

    # 5. Price inflation: passed through from a shell ring, or unexplained
    for s, t in sig_by.items():
        if "price_deviation" in t and s not in shell_members:
            upstream = sorted(set(links[links.buyer_id == s].supplier_id) & shell_members)
            if upstream:
                add("price_pass_through", [s],
                    f"Supplier {s} charges inflated prices and buys from shell-ring "
                    f"supplier(s) {upstream}; the inflation is likely passed through. "
                    f"Status: review (cannot tell from the data whether it knew)")
            else:
                add("unexplained_price_inflation", [s],
                    f"Supplier {s} charges well above market with no upstream cause found")

    return pd.DataFrame(hyps)


if __name__ == "__main__":
    data = load_data()
    signals = run_all(data)
    risk = supplier_risk(signals)
    hyps = fuse(data, signals)
    risk.to_csv(DATA_DIR / "supplier_risk.csv", index=False)
    hyps.to_csv(DATA_DIR / "hypotheses.csv", index=False)

    print("SUPPLIER RISK")
    print(risk.to_string(index=False), "\n")
    print("ROOT-CAUSE HYPOTHESES")
    for _, h in hyps.iterrows():
        print(f"{h.hypothesis_id} [{h.root_cause}] suppliers {h.suppliers} "
              f"| confidence {h.confidence}")
        print(f"   {h.explanation}\n")

    def members(kind):
        row = hyps[hyps.root_cause == kind]
        return set(map(int, ",".join(row.suppliers).split(","))) if len(row) else set()

    lv = risk.set_index("supplier_id")["level"].to_dict()
    print("CHECKS")
    print("shell ring == {30,41,52}:", "PASS" if members("shell_ring") == {30, 41, 52} else "CHECK")
    print("bid ring == {15,20,25}:", "PASS" if members("bid_rigging_cluster") == {15, 20, 25} else "CHECK")
    print("over-billing == {2}:", "PASS" if members("over_billing") == {2} else "CHECK")
    print("timing == {18}:", "PASS" if members("timing_manipulation") == {18} else "CHECK")
    print("pass-through == {12}:", "PASS" if members("price_pass_through") == {12} else "CHECK")
    print("decoys 40 and 55 LOW:", "PASS" if lv.get(40) == "LOW" and lv.get(55) == "LOW" else "CHECK")
    print("decoys 3 and 5 not above LOW:",
          "PASS" if lv.get(3, "LOW") == "LOW" and lv.get(5, "LOW") == "LOW" else "CHECK")
    high = {s for s, l in lv.items() if l == "HIGH"}
    print("HIGH == {2,15,18,20,25,30,41,52}:",
          "PASS" if high == {2, 15, 18, 20, 25, 30, 41, 52} else f"CHECK {sorted(high)}")