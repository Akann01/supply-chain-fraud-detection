from pathlib import Path
import pandas as pd

# Always points at E:\fraud_platform\data, wherever you run from
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SIGNAL_COLUMNS = ["supplier_id", "signal_type", "strength", "evidence"]


def load_data():
    names = ["suppliers", "supply_links", "purchase_orders", "deliveries",
            "invoices", "tenders", "bids"]
    return {n: pd.read_csv(DATA_DIR / f"{n}.csv") for n in names}