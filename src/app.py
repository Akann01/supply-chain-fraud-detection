import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from common import load_data
from pipeline import REQUIRED, validate, run_pipeline

st.set_page_config(page_title="Supply Chain Fraud Platform", layout="wide")
st.title("Multi-tier Supply Chain Fraud Detection")
st.caption("Upload your data to see which suppliers look risky, why, and where it started.")

source = st.sidebar.radio("Data source", ["Use sample data", "Upload my CSVs"])
data = None

if source == "Use sample data":
    data = load_data()
else:
    st.sidebar.write("Upload one CSV per table:")
    uploaded = {}
    for name in REQUIRED:
        f = st.sidebar.file_uploader(f"{name}.csv", type="csv", key=name)
        if f is not None:
            uploaded[name] = pd.read_csv(f)
    problems = validate(uploaded)
    if problems:
        st.info("Waiting for a complete, valid upload.")
        for p in problems:
            st.write("- " + p)
        st.stop()
    data = uploaded


@st.cache_data(show_spinner="Running detectors...")
def cached_run(_data, key):
    return run_pipeline(_data)


key = source + str(sum(len(v) for v in data.values()))
res = cached_run(data, key)
risk, hyps, traces, pred = res["risk"], res["hypotheses"], res["traces"], res["predicted"]
signals = res["signals"]
sup = data["suppliers"].set_index("supplier_id")
names, tiers = sup["name"].to_dict(), sup["tier"].to_dict()
risk_of = dict(zip(risk.supplier_id, risk.risk))
level_of = dict(zip(risk.supplier_id, risk.level))

# ---------- business impact ----------
po, inv = data["purchase_orders"], data["invoices"]
merged = po.merge(inv, on="po_id")
over_bill = (merged["invoice_amount"] - merged["quantity"] * merged["unit_price"]).clip(lower=0)
merged["over_bill"] = over_bill
market = po.groupby("item")["unit_price"].transform("median")
po = po.assign(above_market=((po["unit_price"] - market) * po["quantity"]).clip(lower=0))
flagged = set(risk[risk.level == "HIGH"].supplier_id) | set(
    risk[risk.level == "MEDIUM"].supplier_id)
overbilled = merged[merged.seller_id.isin(flagged)]["over_bill"].sum()
above_mkt = po[po.seller_id.isin(flagged)]["above_market"].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Suppliers analysed", len(sup))
c2.metric("High-risk suppliers", int((risk.level == "HIGH").sum()))
c3.metric("Root-cause hypotheses", len(hyps))
c4.metric("Estimated excess spend", f"{overbilled + above_mkt:,.0f}")
st.caption("Excess spend = billed beyond purchase orders plus paid above the item's median "
           "price, for medium/high-risk suppliers. An estimate, not proof of loss.")

tab1, tab2, tab3, tab4 = st.tabs(["Risk table", "Root causes", "Network view", "Watchlist"])

with tab1:
    view = risk.copy()
    view.insert(1, "name", view.supplier_id.map(names))
    view.insert(2, "tier", view.supplier_id.map(tiers))
    st.dataframe(view, use_container_width=True, hide_index=True)
    pick = st.selectbox("Evidence for supplier", risk.supplier_id.tolist())
    for _, r in signals[signals.supplier_id == pick].iterrows():
        st.write(f"**{r.signal_type.replace('_', ' ')}** (strength {r.strength}): {r.evidence}")

with tab2:
    for _, h in hyps.iterrows():
        t = traces[traces.hypothesis_id == h.hypothesis_id].iloc[0]
        with st.expander(f"{h.hypothesis_id}: {h.root_cause.replace('_', ' ')} "
                         f"(suppliers {h.suppliers}, confidence {h.confidence})"):
            st.write(h.explanation)
            st.write(f"**Origin tier:** {t.origin_tier}  |  "
                     f"**Tier-1 vendor(s) exposed:** {t.exposed_tier1}")
            for p in t.paths.split(" | "):
                st.code(p)

with tab3:
    g = nx.DiGraph()
    for s, b in data["supply_links"][["supplier_id", "buyer_id"]].itertuples(index=False):
        g.add_edge(int(s), int(b))
    pred_of = dict(zip(pred.supplier_id, pred.predicted_risk))
    hi_nodes = set()
    for _, t in traces.iterrows():
        hi_nodes |= {int(x) for x in t.suppliers.split(",")}
        hi_nodes |= {int(x) for x in str(t.downstream_nodes).split(",") if x}
    only = st.checkbox("Show only flagged suppliers and their paths to the company", True)
    nodes = [n for n in g.nodes if (not only) or n == 0 or n in hi_nodes]
    # layout: x spreads nodes within a tier, y is the tier (3 at top, company at bottom)
    by_tier = {0: [0], 1: [], 2: [], 3: []}
    for n in nodes:
        if n != 0:
            by_tier[int(tiers[n])].append(n)
    pos = {}
    for t_, ns in by_tier.items():
        ns = sorted(ns)
        for i, n in enumerate(ns):
            pos[n] = ((i + 1) / (len(ns) + 1), -t_ if t_ else 0.0)
            pos[n] = (pos[n][0], {0: 0, 1: 1, 2: 2, 3: 3}[t_])
    ex, ey = [], []
    for a, b in g.edges:
        if a in pos and b in pos:
            ex += [pos[a][0], pos[b][0], None]
            ey += [pos[a][1], pos[b][1], None]
    color = {"HIGH": "#d62728", "MEDIUM": "#ff9f1c", "LOW": "#8da0b3"}
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines",
                               line=dict(color="#556", width=1), hoverinfo="none"))
    xs, ys, cs, txt, hov = [], [], [], [], []
    for n in nodes:
        xs.append(pos[n][0]); ys.append(pos[n][1])
        lv = "LOW" if n == 0 else level_of.get(n, "LOW")
        cs.append("#2ca02c" if n == 0 else color[lv])
        txt.append("Company" if n == 0 else str(n))
        hov.append("Company" if n == 0 else
                   f"Supplier {n} ({names[n]})<br>Tier {tiers[n]}<br>"
                   f"Risk {risk_of.get(n, 0)} ({lv})<br>Predicted {pred_of.get(n, 0)}")
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=txt,
                             textposition="top center", hovertext=hov, hoverinfo="text",
                             marker=dict(size=22, color=cs)))
    fig.update_layout(height=620, showlegend=False,
                      yaxis=dict(tickvals=[0, 1, 2, 3],
                                 ticktext=["Company", "Tier 1", "Tier 2", "Tier 3"],
                                 autorange="reversed", showgrid=False),
                      xaxis=dict(visible=False), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red = high risk, orange = medium, grey = no strong signal, green = your company. "
               "Tier-1 vendors with no signals of their own still appear when fraud flows through them.")

with tab4:
    st.write("Suppliers with no strong signal of their own, ranked by closeness to flagged "
             "suppliers. A watchlist for review, not an accusation.")
    w = pred[(pred.own_risk < 0.35) & (pred.predicted_risk > 0)].head(15).copy()
    w.insert(1, "name", w.supplier_id.map(names))
    st.dataframe(w, use_container_width=True, hide_index=True)