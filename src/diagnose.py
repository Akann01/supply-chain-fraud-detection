import tempfile
from pathlib import Path
from evaluate import generate, load_dir, score, run_pipeline, SEEDS

PLANTED = [30, 41, 52, 15, 20, 25, 2, 18]

with tempfile.TemporaryDirectory() as tmp:
    for seed in SEEDS:
        folder = Path(tmp) / f"s{seed}"
        generate(seed, folder)
        data, key = load_dir(folder)
        res = run_pipeline(data)
        sc = score(res, key, data)
        risk = res["risk"].set_index("supplier_id")
        sig = res["signals"]
        missed = [s for s in PLANTED
                  if s not in risk.index or risk.loc[s, "level"] == "LOW"]
        fps = sc["false_positives"]
        if not (missed or fps or sc["rootcause_type_members"] < 1):
            continue
        print(f"\n=== seed {seed}: root-cause {sc['rootcause_type_members']:.2f} ===")
        print("hypotheses:", [(h.root_cause, h.suppliers)
                              for h in res["hypotheses"].itertuples()])
        for s in missed:
            print(f"MISSED supplier {s}: risk",
                  risk.loc[s, "risk"] if s in risk.index else "no signals")
            for r in sig[sig.supplier_id == s].itertuples():
                print(f"   {r.signal_type} ({r.strength}): {r.evidence}")
        for s in fps:
            print(f"FALSE POSITIVE supplier {s}: risk {risk.loc[s, 'risk']}")
            for r in sig[sig.supplier_id == s].itertuples():
                print(f"   {r.signal_type} ({r.strength}): {r.evidence}")