import random
from datetime import date, timedelta
import pandas as pd

random.seed(7)

START, END = date(2026, 1, 1), date(2026, 9, 30)

ITEMS = {  # item: market unit price (Rs)
    "Steel Sheet": 620, "Copper Wire": 880, "Aluminium Rod": 540,
    "Rubber Gasket": 45, "Hydraulic Pump": 9200, "Bearing Set": 1350,
    "Paint Drum": 2100, "Packaging Film": 310, "Circuit Board": 1750,
    "Plastic Granules": 95,
}
CITIES = ["Pune", "Chennai", "Indore", "Ahmedabad", "Coimbatore",
          "Nashik", "Rajkot", "Faridabad", "Hosur", "Aurangabad"]


def random_date(a, b):
    return a + timedelta(days=random.randint(0, (b - a).days))


# ---------- tiers ----------
T1 = list(range(1, 9))        # supply the company (id 0)
T2 = list(range(9, 29))       # supply Tier-1
T3 = list(range(29, 69))      # supply Tier-2
ALL = T1 + T2 + T3
tier_of = {s: (1 if s in T1 else 2 if s in T2 else 3) for s in ALL}

# ---------- planted fraud (the answer key) ----------
SHELL_T3 = [30, 41, 52]   # Case 1: shell trio at Tier-3
SHELL_BUYER = 12          # Tier-2 supplier they feed
HIDDEN_T1 = 4             # clean-looking Tier-1 that supplier 12 feeds
RING = [15, 20, 25]       # Case 2: bid-rigging ring at Tier-2
RING_ISSUER = 7           # Tier-1 supplier running the rigged tenders
OVERBILLER = 2            # Case 3: Tier-1 supplier over-invoicing the company
LATE_INVOICER = 18        # Case 4: Tier-2 invoicing before delivering
LATE_BUYER = 6            # Tier-1 supplier that 18 feeds
DECOY_ADDR = (40, 55)     # innocent: share an address only
DECOY_PRICE = 3           # innocent: one price spike only
DECOY_LATE = 5            # innocent: one late delivery only

# ---------- suppliers ----------
sells = {s: random.sample(list(ITEMS), 2) for s in ALL}
sup = {}
for s in ALL:
    sup[s] = {
        "supplier_id": s, "name": f"Supplier_{s:02d}", "tier": tier_of[s],
        "bank_account": f"ACC{5000 + s}",
        "address": f"{s} Industrial Estate, {random.choice(CITIES)}",
        "registered_date": random_date(date(2012, 1, 1), date(2024, 12, 31)).isoformat(),
        "items": ";".join(sells[s]),
    }
for s in SHELL_T3:                        # shells are newly registered
    sup[s]["registered_date"] = random_date(date(2026, 3, 15), date(2026, 4, 30)).isoformat()
sup[41]["bank_account"] = sup[30]["bank_account"]
sup[52]["bank_account"] = sup[30]["bank_account"]
sup[41]["address"] = sup[30]["address"]
sup[20]["address"] = sup[15]["address"]   # ring members 15 & 20 share an address
sup[55]["address"] = sup[40]["address"]   # decoy

# ---------- supply links (who supplies whom) ----------
links = set()
for s in T1:
    links.add((s, 0))
for i, s in enumerate(T2):
    links.add((s, T1[i % len(T1)]))
for i, s in enumerate(T3):
    links.add((s, T2[i % len(T2)]))
for s in random.sample(T2, 6):            # a few extra cross-links
    links.add((s, random.choice(T1)))
for s in random.sample(T3, 12):
    links.add((s, random.choice(T2)))


def set_link(seller, buyer):
    for l in [l for l in links if l[0] == seller]:
        links.discard(l)
    links.add((seller, buyer))


for s in SHELL_T3:
    set_link(s, SHELL_BUYER)
set_link(SHELL_BUYER, HIDDEN_T1)
for s in RING:
    set_link(s, RING_ISSUER)
set_link(LATE_INVOICER, LATE_BUYER)

# ---------- purchase orders ----------
pos = []


def add_po(seller, buyer, price_mult=1.0, start=START, end=END):
    item = random.choice(sells[seller])
    qty = random.randint(20, 400)
    po_date = random_date(start, end - timedelta(days=45))
    expected = po_date + timedelta(days=random.randint(10, 30))
    unit = round(ITEMS[item] * price_mult * random.uniform(0.95, 1.05), 2)
    pos.append({"seller_id": seller, "buyer_id": buyer, "item": item,
                "quantity": qty, "unit_price": unit,
                "po_date": po_date.isoformat(),
                "expected_delivery": expected.isoformat()})


for seller, buyer in sorted(links):
    if seller in SHELL_T3:
        continue                          # shells only appear in the burst below
    for _ in range(random.randint(4, 8)):
        if (seller, buyer) == (SHELL_BUYER, HIDDEN_T1):
            # pass-through: supplier 12 passes inflated prices to clean Tier-1 #4
            add_po(seller, buyer, price_mult=random.uniform(1.2, 1.4),
                   start=date(2026, 6, 15))
        else:
            add_po(seller, buyer)

for s in SHELL_T3:                        # Case 1: inflated-price burst
    for _ in range(6):
        add_po(s, SHELL_BUYER, price_mult=random.uniform(1.6, 2.0),
               start=date(2026, 6, 1), end=date(2026, 8, 31))

add_po(DECOY_PRICE, 0, price_mult=1.25)   # decoy: a single price spike

# ---------- deliveries and invoices ----------
deliveries, invoices = [], []
decoy_late_done = False
for i, po in enumerate(pos, start=1):
    po["po_id"] = i
    seller, buyer, qty, unit = po["seller_id"], po["buyer_id"], po["quantity"], po["unit_price"]
    expected = date.fromisoformat(po["expected_delivery"])

    delivered = expected + timedelta(days=random.randint(-3, 4))
    inv_date = delivered + timedelta(days=random.randint(0, 5))
    inv_qty = qty

    if seller == OVERBILLER and buyer == 0 and random.random() < 0.45:
        inv_qty = round(qty * random.uniform(1.15, 1.30))      # Case 3
    if seller == LATE_INVOICER:                                # Case 4
        delivered = expected + timedelta(days=random.randint(12, 30))
        if random.random() < 0.8:
            inv_date = delivered - timedelta(days=random.randint(5, 20))
        else:
            inv_date = delivered + timedelta(days=random.randint(0, 5))
    if seller == DECOY_LATE and not decoy_late_done:           # decoy
        delivered = expected + timedelta(days=20)
        inv_date = delivered + timedelta(days=2)
        decoy_late_done = True

    deliveries.append({"po_id": i, "delivered_date": delivered.isoformat(),
                       "delivered_qty": qty})
    invoices.append({"invoice_id": i, "po_id": i,
                     "invoice_date": inv_date.isoformat(),
                     "invoiced_qty": inv_qty, "invoiced_unit_price": unit,
                     "invoice_amount": round(inv_qty * unit, 2)})

# ---------- tenders and bids ----------
tenders, bids = [], []


def add_tender(issuer, item, bidders, ring_winner=None, outsider=None):
    tid = len(tenders) + 1
    qty = random.randint(50, 500)
    ref = ITEMS[item] * qty
    tenders.append({"tender_id": tid, "issuer_id": issuer, "item": item,
                    "quantity": qty,
                    "tender_date": random_date(date(2026, 1, 15), date(2026, 9, 1)).isoformat()})
    if ring_winner is not None:           # rigged: near-identical bids, rotating winner
        base = ref * random.uniform(0.97, 1.03)
        amounts = {b: base * random.uniform(0.985, 1.015) for b in bidders}
        amounts[ring_winner] = min(amounts.values()) * 0.995
        if outsider is not None:
            amounts[outsider] = base * random.uniform(1.10, 1.25)
    else:                                 # honest: bids spread widely
        amounts = {b: ref * random.uniform(0.85, 1.15) for b in bidders}
    winner = min(amounts, key=amounts.get)
    for b, amt in amounts.items():
        bids.append({"tender_id": tid, "supplier_id": b,
                     "bid_amount": round(amt, 2), "won": b == winner})


for _ in range(25):                       # honest tenders
    issuer = random.choice([0] + T1)
    pool = T1 if issuer == 0 else T2
    add_tender(issuer, random.choice(list(ITEMS)),
               random.sample(pool, random.randint(3, 5)))

for t in range(6):                        # Case 2: rigged tenders
    outsider = random.choice([s for s in T2 if s not in RING]) if t % 2 == 0 else None
    add_tender(RING_ISSUER, random.choice(list(ITEMS)), RING,
               ring_winner=RING[t % 3], outsider=outsider)

# ---------- answer key ----------
cases = [
    {"case_id": 1, "fraud_type": "Shell trio + inflated prices (passed through clean Tier-1)",
     "suppliers": "30,41,52", "origin_tier": 3,
     "chain": "30,41,52 -> 12 -> 4 -> company"},
    {"case_id": 2, "fraud_type": "Bid-rigging ring (2 members share an address)",
     "suppliers": "15,20,25", "origin_tier": 2,
     "chain": "15,20,25 -> 7 -> company"},
    {"case_id": 3, "fraud_type": "Invoice over-billing",
     "suppliers": "2", "origin_tier": 1, "chain": "2 -> company"},
    {"case_id": 4, "fraud_type": "Invoicing before delivery + late deliveries",
     "suppliers": "18", "origin_tier": 2, "chain": "18 -> 6 -> company"},
    {"case_id": 5, "fraud_type": "DECOY: shared address only (innocent)",
     "suppliers": "40,55", "origin_tier": 3, "chain": ""},
    {"case_id": 6, "fraud_type": "DECOY: one price spike only (innocent)",
     "suppliers": "3", "origin_tier": 1, "chain": ""},
    {"case_id": 7, "fraud_type": "DECOY: one late delivery only (innocent)",
     "suppliers": "5", "origin_tier": 1, "chain": ""},
]

# ---------- save ----------
pd.DataFrame(list(sup.values())).to_csv("data/suppliers.csv", index=False)
pd.DataFrame(sorted(links), columns=["supplier_id", "buyer_id"]).to_csv("data/supply_links.csv", index=False)
pd.DataFrame(pos).to_csv("data/purchase_orders.csv", index=False)
pd.DataFrame(deliveries).to_csv("data/deliveries.csv", index=False)
pd.DataFrame(invoices).to_csv("data/invoices.csv", index=False)
pd.DataFrame(tenders).to_csv("data/tenders.csv", index=False)
pd.DataFrame(bids).to_csv("data/bids.csv", index=False)
pd.DataFrame(cases).to_csv("data/fraud_cases.csv", index=False)

print(f"Suppliers: {len(sup)} (Tier-1: {len(T1)}, Tier-2: {len(T2)}, Tier-3: {len(T3)})")
print(f"Supply links: {len(links)}")
print(f"Purchase orders: {len(pos)}")
print(f"Tenders: {len(tenders)}, bids: {len(bids)}")
print("Answer key saved in data/fraud_cases.csv")