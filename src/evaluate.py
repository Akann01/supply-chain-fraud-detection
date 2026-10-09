import contextlib
import io
import os
import random
import statistics as st
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA_DIR
import detect_bid_rigging, detect_delivery_timing, detect_price_deviation
import fusion
import propagate
from pipeline import run_pipeline

GEN = Path(__file__).resolve().parent / "generate_data.py"
TABLES = ["suppliers", "supply_links", "purchase_orders", "deliveries",
          "invoices", "tenders", "bids"]
SEEDS = list(range(21, 41))        # 20 generated datasets
SWEEP_SEEDS = SEEDS[:10]          # threshold sweep uses 10 to save time
CASE_TYPE = {1: "shell_ring", 2: "bid_rigging_cluster",
             3: "over_billing", 4: "timing_manipulation"}


# ---------- dataset generation (generator file is not modified) ----------
def generate(seed, folder):
    folder = Path(folder)
    (folder / "data").mkdir(parents=True, exist_ok=True)
    src = GEN.read_text(encoding="utf-8")
    saved = (random.seed, random.Random, np.random.seed, np.random.default_rng)
    OrigRandom, orig_seed, orig_rng = random.Random, random.seed, np.random.default_rng
    orig_np_seed = np.random.seed

    class SeededRandom(OrigRandom):
        def __init__(self, x=None):
            super().__init__(seed)

    random.seed = lambda *a, **k: orig_seed(seed)
    random.Random = SeededRandom
    np.random.seed = lambda *a, **k: orig_np_seed(seed)
    np.random.default_rng = lambda *a, **k: orig_rng(seed)
    old_cwd = os.getcwd()
    os.chdir(folder)
    try:
        orig_seed(seed)
        orig_np_seed(seed)
        fake_file = str(folder / "src" / "generate_data.py")   # keeps any path logic inside the temp folder
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src, fake_file, "exec"),
                 {"__name__": "__main__", "__file__": fake_file})
    finally:
        os.chdir(old_cwd)
        random.seed, random.Random, np.random.seed, np.random.default_rng = saved


def load_dir(folder):
    d = Path(folder) / "data"
    data = {n: pd.read_csv(d / f"{n}.csv") for n in TABLES}
    return data, pd.read_csv(d / "fraud_cases.csv")


# ---------- scoring ----------
def score(res, key, data):
    risk = res["risk"]
    flagged = {int(s) for s, l in zip(risk.supplier_id, risk.level) if l in ("HIGH", "MEDIUM")}
    pos, neutral, cases = set(), set(), {}
    for _, k in key.iterrows():
        members = {int(x) for x in str(k.suppliers).split(",")}
        if int(k.case_id) <= 4:
            pos |= members
            cases[int(k.case_id)] = (members, int(k.origin_tier))
            toks = str(k.chain).split(" -> ")
            neutral |= {int(t) for t in toks[1:] if t.strip().isdigit()}
    neutral -= pos      # intermediaries (e.g. 12, 4) are neither rewarded nor penalised
    tp = len(flagged & pos)
    fp = flagged - pos - neutral
    precision = tp / (tp + len(fp)) if (tp + len(fp)) else 1.0
    recall = tp / len(pos)

    hyps, tr = res["hypotheses"], res["traces"]
    rc_ok = tier_ok = 0
    for cid, (members, otier) in cases.items():
        for _, h in hyps[hyps.root_cause == CASE_TYPE[cid]].iterrows():
            if {int(x) for x in h.suppliers.split(",")} == members:
                rc_ok += 1
                t = tr[tr.hypothesis_id == h.hypothesis_id].iloc[0]
                tier_ok += int(int(t.origin_tier) == otier)
                break
    n = len(cases)
    return {"precision": precision, "recall": recall, "false_positives": sorted(fp),
            "rootcause_type_members": rc_ok / n, "rootcause_with_tier": tier_ok / n}


def hide_test(data, res):
    risk = res["risk"]
    base = dict(zip(risk.supplier_id, risk.risk))
    edges = propagate.build_edges(data)
    out = {}
    for s in [52, 30, 25, 15, 18, 2]:
        q = propagate.predict(edges, base, hidden={s}).set_index("supplier_id")["predicted_risk"]
        pool = [n for n in q.index if n == s or base.get(n, 0.0) < propagate.FLAG_AT]
        rank = q[pool].rank(ascending=False, method="average")[s]
        out[s] = bool(q[s] > 0 and rank <= 5)
    return out


# ---------- threshold sweep ----------
def set_param(name, value):
    if name == "LATE_DAYS":
        detect_delivery_timing.LATE_DAYS = value
    elif name == "PRICE_DEVIATION":
        detect_price_deviation.DEVIATION = value
    elif name == "CLOSE_BID_PCT":
        detect_bid_rigging.CLOSE_BID_PCT = value
        fusion.CLOSE_BID_PCT = value
    elif name == "MEDIUM_CUTOFF":
        fusion.MEDIUM = value


SWEEPS = {"LATE_DAYS": (7, [0, 3, 7, 10]),
          "PRICE_DEVIATION": (0.15, [0.05, 0.10, 0.15, 0.30]),
          "CLOSE_BID_PCT": (0.03, [0.01, 0.03, 0.05, 0.10]),
          "MEDIUM_CUTOFF": (0.35, [0.20, 0.35, 0.50])}


def main():
    real_key = DATA_DIR / "fraud_cases.csv"
    stamp = real_key.stat().st_mtime if real_key.exists() else None

    datasets = []
    with tempfile.TemporaryDirectory() as tmp:
        print(f"Generating {len(SEEDS)} datasets...")
        for seed in SEEDS:
            folder = Path(tmp) / f"s{seed}"
            generate(seed, folder)
            datasets.append((seed, *load_dir(folder)))

        if stamp is not None and real_key.stat().st_mtime != stamp:
            sys.exit("STOP: your real data folder was modified. Tell me before continuing.")

        rows, hide_rows = [], []
        print("Running full pipeline on each dataset...")
        for seed, data, key in datasets:
            t0 = time.perf_counter()
            res = run_pipeline(data)
            secs = time.perf_counter() - t0
            sc = score(res, key, data)
            sc.update({"seed": seed, "suppliers": len(data["suppliers"]),
                       "purchase_orders": len(data["purchase_orders"]),
                       "runtime_s": round(secs, 2)})
            rows.append(sc)
            hide_rows.append(hide_test(data, res))

        df = pd.DataFrame(rows)
        df["false_positives"] = df["false_positives"].astype(str)
        df.to_csv(DATA_DIR / "evaluation_results.csv", index=False)

        def ms(col):
            v = df[col].tolist()
            return f"{st.mean(v):.3f} +/- {st.pstdev(v):.3f}  (min {min(v):.2f}, max {max(v):.2f})"

        print("\n=== RESULTS ON", len(SEEDS), "SIMULATED DATASETS (injected fraud) ===")
        print("Precision:                       ", ms("precision"))
        print("Recall:                          ", ms("recall"))
        print("Root cause (type + members):     ", ms("rootcause_type_members"))
        print("Root cause incl. origin tier:    ", ms("rootcause_with_tier"))
        print("Runtime per pipeline run (s):    ", ms("runtime_s"))
        print("Datasets with any false positive:",
              int((df.false_positives != "[]").sum()), "of", len(df))
        bad = df[df.false_positives != "[]"][["seed", "false_positives"]]
        if len(bad):
            print(bad.to_string(index=False))

        print("\nHIDE-AND-RECOVER (hidden supplier lands in top 5 of the unflagged pool):")
        for s in [52, 30, 25, 15, 18, 2]:
            k = sum(h[s] for h in hide_rows)
            print(f"  supplier {s}: recovered in {k} of {len(hide_rows)} datasets")

        print("\nTHRESHOLD SENSITIVITY (mean over", len(SWEEP_SEEDS), "datasets)")
        sweep_rows = []
        for name, (default, values) in SWEEPS.items():
            for v in values:
                set_param(name, v)
                ps, rs, fps = [], [], []
                for seed, data, key in datasets:
                    if seed not in SWEEP_SEEDS:
                        continue
                    sc = score(run_pipeline(data), key, data)
                    ps.append(sc["precision"]); rs.append(sc["recall"])
                    fps.append(len(sc["false_positives"]))
                sweep_rows.append({"parameter": name, "value": v,
                                   "precision": round(st.mean(ps), 3),
                                   "recall": round(st.mean(rs), 3),
                                   "avg_false_positives": round(st.mean(fps), 2),
                                   "is_default": v == default})
            set_param(name, default)
        sw = pd.DataFrame(sweep_rows)
        sw.to_csv(DATA_DIR / "threshold_sensitivity.csv", index=False)
        print(sw.to_string(index=False))

    print("\nSaved evaluation_results.csv and threshold_sensitivity.csv in data/")
    print("Reminder: fraud is injected by your own generator, so these numbers show the "
          "pipeline finds the patterns it was built to find, not performance on real data.")


if __name__ == "__main__":
    main()