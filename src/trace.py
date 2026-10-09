import networkx as nx
import pandas as pd
from common import load_data, DATA_DIR
from run_detectors import run_all
from fusion import fuse


def build_graph(data):
    g = nx.DiGraph()
    for s, b in data["supply_links"][["supplier_id", "buyer_id"]].itertuples(index=False):
        g.add_edge(int(s), int(b))
    return g


def label(n):
    return "company" if n == 0 else str(n)


def trace(data, hyps):
    g = build_graph(data)
    links = data["supply_links"]
    tier = {int(k): int(v) for k, v in data["suppliers"].set_index("supplier_id")["tier"].items()}
    shell_set = set()
    for _, h in hyps[hyps.root_cause == "shell_ring"].iterrows():
        shell_set |= {int(x) for x in h.suppliers.split(",")}

    rows = []
    for _, h in hyps.iterrows():
        members = [int(x) for x in h.suppliers.split(",")]
        paths = []
        for m in members:
            paths += list(nx.all_simple_paths(g, m, 0, cutoff=4))
        downstream = {n for p in paths for n in p[1:-1]} - set(members)
        exposed = sorted(n for n in (downstream | set(members)) if tier[n] == 1)

        origin_members = members
        if h.root_cause == "price_pass_through":
            up = set(links[links.buyer_id.isin(members)].supplier_id.astype(int)) & shell_set
            if up:
                origin_members = sorted(up)
        origin_tier = max(tier[m] for m in origin_members)

        unique_paths = sorted({" -> ".join(label(n) for n in p) for p in paths})
        rows.append({"hypothesis_id": h.hypothesis_id, "root_cause": h.root_cause,
                     "suppliers": h.suppliers, "origin_tier": origin_tier,
                     "exposed_tier1": ",".join(map(str, exposed)),
                     "downstream_nodes": ",".join(map(str, sorted(downstream))),
                     "paths": " | ".join(unique_paths)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = load_data()
    signals = run_all(data)
    hyps = fuse(data, signals)
    tr = trace(data, hyps)
    tr.to_csv(DATA_DIR / "traces.csv", index=False)

    for _, r in tr.iterrows():
        print(f"{r.hypothesis_id} [{r.root_cause}] suppliers {r.suppliers}")
        print(f"   origin tier: {r.origin_tier} | Tier-1 vendor(s) exposed: {r.exposed_tier1}")
        for p in r.paths.split(" | "):
            print(f"   {p}")
        print()

    key = pd.read_csv(DATA_DIR / "fraud_cases.csv")
    print("CHECKS AGAINST ANSWER KEY")
    for _, k in key.iterrows():
        kset = {int(x) for x in str(k.suppliers).split(",")}
        match = tr[tr.suppliers.apply(lambda s: {int(x) for x in s.split(",")} == kset)]
        if match.empty:
            print(f"Case {k.case_id}: CHECK no hypothesis with suppliers {sorted(kset)}")
            continue
        m = match.iloc[0]
        tokens = str(k.chain).split(" -> ")
        mids = {int(t) for t in tokens[1:] if t.strip().isdigit()}
        found = {int(x) for x in m.downstream_nodes.split(",") if x}
        tier_ok = int(m.origin_tier) == int(k.origin_tier)
        chain_ok = mids <= found
        print(f"Case {k.case_id}: origin tier", "PASS" if tier_ok else
              f"CHECK (got {m.origin_tier}, key {k.origin_tier})",
              "| chain nodes", "PASS" if chain_ok else
              f"CHECK (key needs {sorted(mids)}, traced {sorted(found)})")

    p = tr[tr.root_cause == "price_pass_through"].iloc[0]
    print("Pass-through at 12 traced to origin tier 3 and Tier-1 vendor 4:",
          "PASS" if int(p.origin_tier) == 3 and "4" in p.exposed_tier1.split(",") else "CHECK")