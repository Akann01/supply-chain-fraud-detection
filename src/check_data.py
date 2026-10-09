import pandas as pd

files = ["suppliers", "supply_links", "purchase_orders", "deliveries",
        "invoices", "tenders", "bids", "fraud_cases"]

for name in files:
    df = pd.read_csv(f"data/{name}.csv")
    print(f"\n=== {name}.csv  ({len(df)} rows) ===")
    print(list(df.columns))
    print(df.head(2).to_string(index=False))