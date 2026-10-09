import pandas as pd
from common import load_data, SIGNAL_COLUMNS

NEW_REG_DAYS = 540   # registered within ~18 months of the latest PO date counts as "new"


def _shared_groups(suppliers, col):
    """Return {value: [supplier_ids]} for values used by 2+ suppliers."""
    groups = suppliers.dropna(subset=[col]).groupby(col)["supplier_id"].apply(list)
    return {k: v for k, v in groups.items() if len(v) > 1}


def detect_shell(data):
    sup = data["suppliers"].copy()
    sup["registered_date"] = pd.to_datetime(sup["registered_date"])
    ref_date = pd.to_datetime(data["purchase_orders"]["po_date"]).max()
    rows = []

    # 1. Shared bank account: strong signal
    for acct, ids in _shared_groups(sup, "bank_account").items():
        for s in ids:
            others = ", ".join(str(i) for i in ids if i != s)
            rows.append({"supplier_id": s, "signal_type": "shared_bank_account",
                         "strength": 0.9,
                         "evidence": f"Receives payment into bank account {acct}, "
                                     f"which is also used by supplier(s) {others}"})

    # 2. Shared address: weak signal alone (offices do share buildings)
    for addr, ids in _shared_groups(sup, "address").items():
        for s in ids:
            others = ", ".join(str(i) for i in ids if i != s)
            rows.append({"supplier_id": s, "signal_type": "shared_address",
                         "strength": 0.4,
                         "evidence": f"Registered at '{addr}', the same address as "
                                     f"supplier(s) {others}"})

    # 3. Newly registered
    for _, r in sup.iterrows():
        age = (ref_date - r["registered_date"]).days
        if age <= NEW_REG_DAYS:
            rows.append({"supplier_id": r["supplier_id"],
                         "signal_type": "new_registration",
                         "strength": round(0.3 + 0.4 * (1 - age / NEW_REG_DAYS), 2),
                         "evidence": f"Registered only {age} days before the latest "
                                     f"order ({r['registered_date'].date()})"})

    return pd.DataFrame(rows, columns=SIGNAL_COLUMNS)


if __name__ == "__main__":
    data = load_data()
    sig = detect_shell(data)

    print(f"{len(sig)} signals\n")
    for t, g in sig.groupby("signal_type"):
        print(f"{t}: suppliers {sorted(g['supplier_id'].tolist())}")

    bank = set(sig[sig.signal_type == "shared_bank_account"].supplier_id)
    addr = set(sig[sig.signal_type == "shared_address"].supplier_id)
    new = set(sig[sig.signal_type == "new_registration"].supplier_id)

    print("\nChecks")
    print("shared bank == {30,41,52}:", "PASS" if bank == {30, 41, 52} else f"CHECK {bank}")
    print("shared address == {30,41,15,20,40,55}:",
        "PASS" if addr == {30, 41, 15, 20, 40, 55} else f"CHECK {addr}")
    print("new registration includes {30,41,52}:",
        "PASS" if {30, 41, 52} <= new else f"CHECK missing {({30,41,52} - new)}")
    print("other newly registered suppliers:", sorted(new - {30, 41, 52}))