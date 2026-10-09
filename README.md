# Multi-tier Supply Chain Fraud Detection

Most fraud tools look at one fraud type at a time and only at Tier-1 suppliers. They can say a supplier looks odd, but not why or where the problem started. This project runs five detectors, fuses their signals into named root causes, traces each one back through the supply tiers, and scores suppliers that sit close to flagged ones.

All data is simulated. The fraud patterns are injected by my own generator, so the results below show that the pipeline finds the patterns it was built to find. They are not a measure of accuracy on real procurement data.

![Dashboard](docs/dashboard.png)
![Network view](docs/network.png)

## How it works

1. **Detectors** (invoice vs PO mismatch, shell-company patterns, bid rigging, price deviation, delivery timing) each output signals with a strength and a plain-English reason.
2. **Fusion** combines signals per supplier and per cluster into root-cause hypotheses, such as "shell ring of 3 Tier-3 suppliers feeding one Tier-2 vendor".
3. **Tracing** follows each hypothesis downstream to the company and names the Tier-1 vendors exposed, even when they have no signals of their own.
4. **Propagation** scores unflagged suppliers by their links to flagged ones (supply links, shared bank account or address, repeated co-bidding) and states the reason.
5. **A Streamlit app** lets you upload CSVs (with column validation) and see the risk table, root causes, a tiered network view and a watchlist.

## Data

The generator creates 68 suppliers across three tiers, about 500 purchase orders and 31 tenders. It plants four fraud cases and three innocent decoys:

- Case 1: a shell trio at Tier-3 (shared bank account, newly registered, inflated prices) feeding a Tier-2 supplier that passes the inflation to a clean-looking Tier-1 vendor
- Case 2: a bid-rigging ring at Tier-2 in tenders issued by a Tier-1 supplier
- Case 3: a Tier-1 supplier over-invoicing the company
- Case 4: a Tier-2 supplier invoicing before delivery and delivering late
- Decoys: a shared address only, one price spike, one late delivery

The answer key is saved in data/fraud_cases.csv.

## Evaluation

I generated 20 datasets per run, each with its own answer key. I tuned thresholds on seeds 1-20, then ran seeds 21-40 once as a held-out test, with no changes in between. I had looked at which suppliers were missed on seeds 21-40 before that run, but I changed no threshold afterwards.

| | Tuning (seeds 1-20) | Held-out (seeds 21-40) |
|---|---|---|
| Precision | 1.000 | 1.000 |
| Recall | 0.981 | 0.981 |
| Root cause correct (type and members) | 0.963 | 0.963 |
| Datasets with a false positive | 0 of 20 | 0 of 20 |

A full pipeline run takes about 0.23 seconds. Results are in data/evaluation_results.csv.

Threshold sensitivity (held-out seeds 21-30, data/threshold_sensitivity.csv):

- Delivery lateness is the most sensitive setting. Counting any late delivery gives precision 0.16 (about 41 innocent suppliers flagged per dataset); a 3-day cutoff gives 0.79; 7 days or more gives 1.00.
- A 1% bid-closeness cutoff misses part of the ring (recall 0.83); 3-10% finds it with no false positives, given a minimum of 4 shared tenders.
- Price-deviation and risk-cutoff settings had no effect across the tested range.
- Recall counts any flag on a planted supplier, not necessarily the right signal.

Hide-and-recover test: I hid a known fraud supplier and checked whether its connections still rank it near the top. Shell-ring members (30, 52) and bid-ring members (15, 25) were recovered in 20 of 20 datasets. Suppliers 18 and 2 were almost never recovered.

## Limitations

- Supplier 2 (over-billing, about 7 orders) is missed on about 15% of datasets because too few orders show the pattern.
- Propagation can only find suppliers connected to flagged ones, so it fails for suppliers 18 and 2, which have no flagged neighbours.
- The shell-ring recovery is partly circular, since a shared bank account is both a detector signal and a graph edge.
- The bid-ring recovery works because the generator puts the ring in the same tenders. Real rings may be less visible.
- Thresholds, including the co-bid and bid-ring minimums, were chosen after seeing results on the tuning seeds.
- Prices are compared against the median of the dataset itself, not an external market price.
- Everything was tested on one generator's idea of fraud.

## Run it

    pip install -r requirements.txt
    python src/generate_data.py
    streamlit run src/app.py
    python src/evaluate.py

Run commands from the project root. Note that evaluate.py overwrites data/evaluation_results.csv and data/threshold_sensitivity.csv.

## Project layout

- src/detect_*.py: the five detectors
- src/fusion.py: root-cause hypotheses and supplier risk
- src/trace.py: upstream tracing
- src/propagate.py: risk propagation and the watchlist
- src/pipeline.py: validation and the full pipeline
- src/app.py: Streamlit app
- src/evaluate.py: multi-seed evaluation
- data/: sample data and results