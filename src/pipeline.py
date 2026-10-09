import pandas as pd

REQUIRED = {
    "suppliers": ["supplier_id", "name", "tier", "bank_account", "address", "registered_date"],
    "supply_links": ["supplier_id", "buyer_id"],
    "purchase_orders": ["po_id", "seller_id", "buyer_id", "item", "quantity", "unit_price",
                        "po_date", "expected_delivery"],
    "deliveries": ["po_id", "delivered_date", "delivered_qty"],
    "invoices": ["invoice_id", "po_id", "invoice_date", "invoiced_qty",
                 "invoiced_unit_price", "invoice_amount"],
    "tenders": ["tender_id", "issuer_id", "item", "quantity", "tender_date"],
    "bids": ["tender_id", "supplier_id", "bid_amount", "won"],
}


def validate(data):
    """Return a list of plain-English problems; empty list means the data is usable."""
    problems = []
    for name, cols in REQUIRED.items():
        if name not in data:
            problems.append(f"Missing file: {name}.csv")
            continue
        missing = [c for c in cols if c not in data[name].columns]
        if missing:
            problems.append(f"{name}.csv is missing column(s): {', '.join(missing)}")
    return problems


def run_pipeline(data):
    from run_detectors import run_all
    from fusion import supplier_risk, fuse
    from trace import trace
    from propagate import build_edges, predict

    signals = run_all(data)
    risk = supplier_risk(signals)
    hyps = fuse(data, signals)
    traces = trace(data, hyps)
    base = dict(zip(risk.supplier_id, risk.risk))
    pred = predict(build_edges(data), base)
    return {"signals": signals, "risk": risk, "hypotheses": hyps,
            "traces": traces, "predicted": pred}