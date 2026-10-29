import json, sqlite3
from pathlib import Path
import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent; ART = ROOT / "artifacts"
st.set_page_config(page_title="CredenceAI", page_icon="📈", layout="wide")
st.markdown("""<style>
[data-testid="stAppViewContainer"]{background:#f6f6f3}
[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.6rem;max-width:1500px}
[data-testid="stSidebar"]{background:#0f3b2d}
[data-testid="stSidebar"] *{color:#d7e8df}
[data-testid="stSidebar"] [role="radiogroup"]{gap:4px}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:10px 14px;border-radius:10px;width:100%;cursor:pointer}
[data-testid="stSidebar"] [role="radiogroup"] label>div:first-child{display:none}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:#16513d;outline:1px solid #2fb582}
.brand{font-size:24px;font-weight:700;color:#fff;margin:4px 0 22px}.brand span{color:#2fb582}
.ws{font-size:11px;letter-spacing:.14em;color:#8fb3a3!important;margin:0 0 8px 6px}
.eb{font-size:11px;letter-spacing:.14em;color:#6b7771;font-weight:600}
.hh{font-size:30px;font-weight:600;color:#14201a;margin:2px 0 4px}
.sb{color:#68756e;font-size:14px;margin:0 0 14px}
.top{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;align-items:flex-end}
.ms{font-size:12px;color:#68756e;text-align:right}.ms b{color:#14201a;font-size:13px}
.ms i{display:inline-block;width:8px;height:8px;border-radius:50%;background:#2fb582;margin-right:6px}
div[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border-radius:14px;border-color:#e6e6df}
.ch{display:flex;gap:10px;align-items:center;font-size:17px;color:#14201a;margin-bottom:2px}
.num{background:#eef1ee;color:#6b7771;border-radius:6px;padding:2px 7px;font-size:12px}
.chip{display:inline-block;padding:3px 10px;border-radius:99px;font-size:11px;font-weight:600}
.chip.g{background:#dff3ea;color:#127a52}.chip.a{background:#fdebd3;color:#a2600a}.chip.r{background:#fbdcdc;color:#b03030}.chip.n{background:#ecefec;color:#68756e}
.stat{padding:6px 4px}.stat span{font-size:12px;color:#68756e;display:block}.stat b{font-size:28px;color:#14201a;font-weight:600}.stat i{display:block;font-size:11px;color:#2c8f66;font-style:normal}
.row{display:flex;justify-content:space-between;align-items:center;padding:12px 0;border-bottom:1px solid #eef0ec;gap:10px}
.row:last-child{border-bottom:0}.row b{font-size:14px;color:#14201a;display:block}.row small{color:#8a958f;font-size:11.5px}
.ic{width:30px;height:30px;border-radius:8px;display:inline-flex;align-items:center;justify-content:center;font-weight:700;flex:none}
.ic.g{background:#dff3ea;color:#127a52}.ic.a{background:#fdebd3;color:#a2600a}
.why{background:#eef7f2;border-radius:10px;padding:12px 14px;font-size:13px;color:#2b3b33;margin-top:10px}
.bar{height:8px;background:#e8ece9;border-radius:6px;overflow:hidden}.bar i{display:block;height:100%;background:#2fb582}
.strip{display:flex;background:#f6f7f4;border-radius:10px;padding:10px 14px;gap:22px;margin:10px 0;flex-wrap:wrap}
.strip span{font-size:11px;color:#8a958f;display:block}.strip b{font-size:15px}
.foot{font-size:12px;color:#8a958f;border-top:1px solid #e6e6df;margin-top:26px;padding-top:12px}
.stButton>button,.stFormSubmitButton>button,.stDownloadButton>button{background:#0f4a37;color:#fff;border:0;border-radius:10px;padding:.55rem 1.1rem}
.stButton>button:hover,.stFormSubmitButton>button:hover{background:#16603f;color:#fff}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load():
    meta = json.loads((ART / "meta.json").read_text())
    m = CatBoostClassifier(); m.load_model(str(ART / "model.cbm"))
    return meta, m, pd.read_csv(ART / "sample_applicants.csv")
meta, model, sample = load()
FE, CATS, DEF, THR, LOW, LAB = meta["features"], meta["cat_features"], meta["defaults"], meta["threshold"], meta["low_cut"], meta["labels"]
ACTION = {"Low": "Proceed to standard processing", "Medium": "Manual underwriting review", "High": "Detailed assessment and additional verification"}
HEAD = {"Low": "Suitable for fast-track review", "Medium": "Refer for manual review", "High": "Refer for detailed assessment"}
COL = {"Low": "#2fb582", "Medium": "#e0962a", "High": "#d24b4b"}
lab = lambda f: LAB.get(f, f.replace("_", " "))
band = lambda p: "High" if p >= THR else ("Medium" if p >= LOW else "Low")
ALT = [c for c in FE if c.startswith(("inst_", "pos_", "cc_", "bureau_", "prev_"))]

def prep(df):
    X = pd.DataFrame(index=df.index)
    for c in FE:
        v = df[c] if c in df.columns else pd.Series(DEF[c], index=df.index)
        X[c] = v.fillna("Unknown").astype(str) if c in CATS else pd.to_numeric(v, errors="coerce").astype("float32")
    return X
score = lambda X: model.predict_proba(X)[:, 1]
shap = lambda X: model.get_feature_importance(Pool(X, cat_features=CATS), type="ShapValues")[:, :-1]
reasons = lambda s, k=3: "; ".join(lab(FE[i]) for i in np.argsort(-s)[:k] if s[i] > 0)

@st.cache_data
def scored_sample():
    X = prep(sample); p = score(X); sv = shap(X); return p, [reasons(s) for s in sv]
REF, REFR = scored_sample(); AVG = float(REF.mean())

def psi(p):
    e = np.array(meta["pd_edges"]); e[0], e[-1] = -1, 2
    a = np.clip(np.histogram(p, bins=e)[0] / len(p), 1e-4, None); b = np.full(10, .1)
    return float(((a - b) * np.log(a / b)).sum())
def db():
    c = sqlite3.connect(ROOT / "audit.db")
    c.execute("create table if not exists audit(ts text, applicant_ref text, pd real, band text, model text, action text, note text, analyst text)"); return c
def derive(r):
    inc = max(float(r["AMT_INCOME_TOTAL"]), 1)
    r["credit_income_ratio"] = r["AMT_CREDIT"] / inc; r["annuity_income_ratio"] = r["AMT_ANNUITY"] / inc
    r["credit_goods_ratio"] = r["AMT_CREDIT"] / r["AMT_GOODS_PRICE"] if r["AMT_GOODS_PRICE"] > 0 else np.nan
    r["annuity_credit_ratio"] = r["AMT_ANNUITY"] / r["AMT_CREDIT"] if r["AMT_CREDIT"] > 0 else np.nan
    r["employed_to_age"] = r["employed_years"] / max(r["age_years"], 1)
    if "CNT_FAM_MEMBERS" in r.index: r["income_per_person"] = inc / max(float(r["CNT_FAM_MEMBERS"]), 1)
    return r

H = st.markdown
def head(title, sub, eb="RISK INTELLIGENCE PLATFORM"):
    H(f'<div class="top"><div><div class="eb">{eb}</div><h1 class="hh">{title}</h1><p class="sb">{sub}</p></div>'
      f'<div class="ms"><i></i><b>Model operational</b><br>{meta["model_version"]} · trained {meta["trained"]}</div></div>', unsafe_allow_html=True)
ch = lambda n, t, s="": H(f'<div class="ch"><span class="num">{n}</span><b>{t}</b></div><p class="sb">{s}</p>', unsafe_allow_html=True)
chip = lambda t, c="g": f'<span class="chip {c}">{t}</span>'
stat = lambda l, v, s="": f'<div class="stat"><span>{l}</span><b>{v}</b><i>{s}</i></div>'
def stats(items):
    with st.container(border=True):
        for col, (l, v, s) in zip(st.columns(len(items)), items): col.markdown(stat(l, v, s), unsafe_allow_html=True)
def ring(p):
    C = 2 * 3.1416 * 60; f = min(p / .5, 1); t = f"{p * 100:.0f}%" if p >= .1 else f"{p * 100:.1f}%"
    return (f'<svg width="150" height="150" viewBox="0 0 150 150"><circle cx="75" cy="75" r="60" fill="none" stroke="#e8ece9" stroke-width="12"/>'
            f'<circle cx="75" cy="75" r="60" fill="none" stroke="{COL[band(p)]}" stroke-width="12" stroke-linecap="round" stroke-dasharray="{C * f:.1f} {C:.1f}" transform="rotate(-90 75 75)"/>'
            f'<text x="75" y="80" text-anchor="middle" font-size="30" font-weight="600" fill="#14201a">{t}</text><text x="75" y="100" text-anchor="middle" font-size="10" fill="#8a958f">Default risk</text></svg>')

st.sidebar.markdown('<div class="brand">Credence<span>AI</span></div><div class="ws">WORKSPACE</div>', unsafe_allow_html=True)
page = st.sidebar.radio("nav", ["Overview", "Applicants", "Model insights", "Data sources", "Settings"], label_visibility="collapsed")
FOOT = '<div class="foot">CredenceAI provides decision support, not automated lending decisions.</div>'

if page == "Overview":
    head("Assess default risk with a fuller picture.", "Combine traditional credit signals with alternative payment and cash-flow data for faster, fairer lending decisions.")
    c1, c2, c3 = st.columns(3, gap="medium"); b1, b2, b3 = c1.container(border=True), c2.container(border=True), c3.container(border=True)
    with b1:
        ch("01", "Applicant profile", "Enter the requested terms and applicant details.")
        sel = st.selectbox("Start from", ["Portfolio average"] + sample.SK_ID_CURR.tolist()[:300])
        base, ref = (pd.Series(DEF), "avg") if sel == "Portfolio average" else (sample[sample.SK_ID_CURR == sel].iloc[0], str(sel))
        row = pd.Series({c: (base[c] if c in base.index and pd.notna(base[c]) else DEF[c]) for c in FE}); k = lambda n: f"{n}_{ref}"
        nz = lambda f, d=0: float(np.nan_to_num(row[f], nan=DEF.get(f, d) if DEF.get(f) is not None else d))
        row["AMT_CREDIT"] = st.number_input("Loan amount", 0.0, 1e8, nz("AMT_CREDIT"), 1000.0, key=k("cr"))
        row["AMT_INCOME_TOTAL"] = 12 * st.number_input("Monthly income", 0.0, 1e7, nz("AMT_INCOME_TOTAL") / 12, 500.0, key=k("inc"))
        row["AMT_ANNUITY"] = st.number_input("Annual repayment", 0.0, 1e8, nz("AMT_ANNUITY"), 1000.0, key=k("an"))
        row["AMT_GOODS_PRICE"] = row["AMT_CREDIT"] if pd.isna(row["AMT_GOODS_PRICE"]) else row["AMT_GOODS_PRICE"]
        with st.expander("More applicant details"):
            row["age_years"] = st.slider("Age", 20, 70, int(nz("age_years", 35)), key=k("age"))
            row["employed_years"] = st.number_input("Years employed", 0.0, 50.0, nz("employed_years", 5), key=k("emp"))
            for f, t in [("NAME_EDUCATION_TYPE", "Education"), ("NAME_INCOME_TYPE", "Income type"), ("NAME_FAMILY_STATUS", "Family status")]:
                o = meta["cat_values"][f]; row[f] = st.selectbox(t, o, o.index(row[f]) if row[f] in o else 0, key=k(f))
            for f in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]: row[f] = st.slider(lab(f), 0.0, 1.0, min(nz(f, .5), 1.0), .01, key=k(f))
    with b3:
        ch("03", "Alternative signals", "Non-traditional data contributing to this assessment.")
        with st.expander("Adjust signals"):
            thin = st.checkbox("No bureau history (thin file)", value=bool(row["bureau_cnt"] == 0), key=k("thin"))
            row["inst_late_share"] = st.slider("Share of late instalments", 0.0, 1.0, min(nz("inst_late_share"), 1.0), .01, key=k("ils"))
            row["inst_days_late_mean"] = st.slider("Average days late", 0.0, 30.0, min(nz("inst_days_late_mean"), 30.0), .5, key=k("idl"))
            row["inst_paid_ratio_mean"] = st.slider("Payment to instalment ratio", 0.0, 2.0, min(nz("inst_paid_ratio_mean", 1), 2.0), .01, key=k("ipr"))
            row["cc_util_mean"] = st.slider("Card utilisation", 0.0, 1.5, min(nz("cc_util_mean"), 1.5), .01, key=k("ccu"))
            row["prev_refused_cnt"] = st.number_input("Previous refusals", 0, 30, int(nz("prev_refused_cnt")), key=k("prf"))
            row["phone_change_years"] = st.slider("Years since phone change", 0.0, 10.0, min(nz("phone_change_years", 1), 10.0), .1, key=k("pch"))
            if not thin: row["bureau_overdue_sum"] = st.number_input("Bureau overdue amount", 0.0, 1e8, nz("bureau_overdue_sum"), 500.0, key=k("bov"))
    if thin:
        for f in [x for x in FE if x.startswith("bureau_")]: row[f] = 0 if f.endswith("_cnt") else np.nan
    row["thin_file"] = int(thin); row = derive(row)
    X1 = prep(pd.DataFrame([row])); p = float(score(X1)[0]); sv = shap(X1)[0]; bd = band(p); rel = (p - AVG) / AVG * 100
    with b2:
        ch("02", "Risk prediction", "Estimated likelihood of default on this loan.")
        a, b = st.columns([1, 1.3])
        a.markdown(ring(p), unsafe_allow_html=True)
        b.markdown(f'{chip(bd.upper() + " RISK", {"Low": "g", "Medium": "a", "High": "r"}[bd])}<h4 style="margin:8px 0 4px">{HEAD[bd]}</h4>'
                   f'<small style="color:#68756e">Risk is {abs(rel):.0f}% {"lower" if rel < 0 else "higher"} than the portfolio average ({AVG:.1%}).</small>', unsafe_allow_html=True)
        H(f'<div class="strip"><div><span>Suggested action</span><b>{ACTION[bd]}</b></div><div><span>Riskier than</span><b>{(REF < p).mean() * 100:.0f}% of portfolio</b></div></div>', unsafe_allow_html=True)
        H('<b style="font-size:14px">Top decision factors</b>', unsafe_allow_html=True)
        for i in np.argsort(-np.abs(sv))[:5]:
            lo = sv[i] < 0; v = X1.iloc[0, i]; v = v if FE[i] in CATS else round(float(v), 2)
            H(f'<div class="row"><div style="display:flex;gap:10px;align-items:center"><span class="ic {"g" if lo else "a"}">{"+" if lo else "−"}</span><div><b>{lab(FE[i])}</b><small>{"Lowers" if lo else "Raises"} risk · value {v}</small></div></div><small>{sv[i]:+.2f}</small></div>', unsafe_allow_html=True)
        up, dn = [lab(FE[i]) for i in np.argsort(-sv)[:2] if sv[i] > 0], [lab(FE[i]) for i in np.argsort(sv)[:2] if sv[i] < 0]
        H(f'<div class="why"><b>Why this score?</b><br>{("Raising risk: " + ", ".join(up) + ". ") if up else "No factor raises risk materially. "}{("Lowering risk: " + ", ".join(dn) + ".") if dn else ""}</div>', unsafe_allow_html=True)
    with b3:
        cov = float(pd.Series([row[c] for c in ALT]).notna().mean()); lv = lambda v, a, b: "g" if v < a else ("a" if v < b else "r")
        late = float(np.nan_to_num(row["inst_late_share"])); util = row["cc_util_mean"]; rs = float(np.nan_to_num(row["prev_refused_share"]))
        H(f'<div style="background:#f6f7f4;border-radius:10px;padding:12px;margin-bottom:6px"><div style="display:flex;justify-content:space-between;font-size:13px"><span>Alternative data coverage</span><b>{cov * 100:.0f}%</b></div><div class="bar"><i style="width:{cov * 100:.0f}%"></i></div></div>', unsafe_allow_html=True)
        sig = [("Payment punctuality", f"{(1 - late) * 100:.0f}% of instalments on time", ["Strong", "Moderate", "Weak"][["g", "a", "r"].index(lv(late, .1, .3))], lv(late, .1, .3)),
               ("Card and cash-flow activity", "No card history" if pd.isna(util) else f"Utilisation {util * 100:.0f}%", "No data" if pd.isna(util) else ["Healthy", "Elevated", "High"][["g", "a", "r"].index(lv(util, .3, .7))], "n" if pd.isna(util) else lv(util, .3, .7)),
               ("Bureau history", "No bureau records" if thin else f"{int(row['bureau_cnt'])} credits", "Thin file" if thin else ("Clean" if float(np.nan_to_num(row["bureau_overdue_sum"])) == 0 else "Overdue"), "n" if thin else ("g" if float(np.nan_to_num(row["bureau_overdue_sum"])) == 0 else "r")),
               ("Previous applications", f"{int(row['prev_cnt'])} previous · {int(row['prev_refused_cnt'])} refused", ["Stable", "Mixed", "Weak"][["g", "a", "r"].index(lv(rs, .1, .3))], lv(rs, .1, .3))]
        for t, s, st_, c in sig: H(f'<div class="row"><div><b>{t}</b><small>{s}</small></div>{chip(st_, c)}</div>', unsafe_allow_html=True)
    with st.container(border=True):
        ch("04", "Analyst decision", "Record the outcome for the audit trail. Overrides need a rationale.")
        with st.form("dec"):
            x1, x2 = st.columns(2); act = x1.selectbox("Action", list(ACTION.values()) + ["Request additional documents"]); who = x2.text_input("Analyst ID")
            note = st.text_area("Rationale")
            if st.form_submit_button("Save to audit log"):
                if not who or (act != ACTION[bd] and not note): st.error("Analyst ID and a rationale for overrides are required.")
                else:
                    c = db(); c.execute("insert into audit values(datetime('now'),?,?,?,?,?,?,?)", (ref, p, bd, meta["model_version"], act, note, who)); c.commit(); st.success("Saved.")
        st.caption("Decision-support estimate, not a final lending decision. Human judgement and institutional policy apply.")
    H(FOOT, unsafe_allow_html=True)

elif page == "Applicants":
    head("Applicants", "Review and manage scored loan applications.", "PORTFOLIO MANAGEMENT")
    up = st.file_uploader("Score a new portfolio (CSV with model feature names; missing columns use defaults)", type="csv")
    if up:
        df = pd.read_csv(up).head(20000); X = prep(df); p = score(X); r = [reasons(s) for s in shap(X)]
    else: df, p, r = sample, REF, REFR
    out = pd.DataFrame({"Applicant": df["SK_ID_CURR"] if "SK_ID_CURR" in df else df.index, "Requested": df["AMT_CREDIT"] if "AMT_CREDIT" in df else DEF["AMT_CREDIT"],
                        "Default probability": p * 100, "Risk level": [band(v) for v in p], "Recommended action": [ACTION[band(v)] for v in p], "Main drivers": r})
    ps = psi(p); stats([("Total applicants", f"{len(out):,}", ""), ("Awaiting manual review", f"{(out['Risk level'] != 'Low').sum():,}", f"{(out['Risk level'] == 'High').sum():,} high priority"),
                        ("Average default risk", f"{p.mean():.1%}", ""), ("Score drift (PSI)", f"{ps:.3f}", "stable" if ps < .1 else ("monitor" if ps < .25 else "retrain"))])
    with st.container(border=True):
        s1, s2 = st.columns([3, 1]); q = s1.text_input("Search applicants", placeholder="Search by applicant ID"); lv = s2.multiselect("Risk level", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
        o = out[out["Risk level"].isin(lv)]
        if q: o = o[o.Applicant.astype(str).str.contains(q)]
        st.dataframe(o, hide_index=True, column_config={"Default probability": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100), "Requested": st.column_config.NumberColumn(format="%,.0f")})
        if "TARGET" in df: st.caption(f"Back-test ROC-AUC on this file: {roc_auc_score(df.TARGET, p):.3f}")
        st.download_button("Download scored list", out.to_csv(index=False), "scored_applicants.csv")
    H(FOOT, unsafe_allow_html=True)

elif page == "Model insights":
    head("Model insights", "Monitor performance, fairness, and the signals driving decisions.", "MODEL INTELLIGENCE")
    t = meta["test"]; stats([("ROC-AUC", f"{t['roc_auc']:.3f}", "held-out test set"), ("Recall", f"{t['recall']:.1%}", "defaulters caught"), ("Precision", f"{t['precision']:.1%}", "of flagged applicants"), ("KS statistic", f"{t['ks']:.3f}", "")])
    l, r = st.columns([1.3, 1], gap="medium")
    with l.container(border=True):
        ch("", "Risk distribution", "Predicted default probability across the holdout portfolio.")
        f = go.Figure(go.Histogram(x=REF * 100, nbinsx=30, marker_color="#2fb582")); f.add_vline(x=THR * 100, line_dash="dash", line_color="#d24b4b", annotation_text="flag threshold")
        f.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10), plot_bgcolor="white", xaxis_title="Default probability (%)"); st.plotly_chart(f)
        st.caption(f"Alternative data lifts ROC-AUC from {meta['ablation_auc']['traditional']:.3f} to {meta['ablation_auc']['with_alternative']:.3f}. Thin-file AUC {meta['thin_thick_auc']['thin_file']:.3f}, thick-file {meta['thin_thick_auc']['thick_file']:.3f}.")
    with r.container(border=True):
        ch("", "Feature importance", "Top signals across all predictions (mean |SHAP|).")
        imp = pd.read_csv(ART / "importance.csv", index_col=0).iloc[:8, 0]; tot = imp.sum()
        for n, v in imp.items(): H(f'<div style="margin:9px 0"><div style="display:flex;justify-content:space-between;font-size:13px"><span>{lab(n)}</span><b>{v / tot * 100:.0f}%</b></div><div class="bar"><i style="width:{v / imp.max() * 100:.0f}%"></i></div></div>', unsafe_allow_html=True)
    with st.container(border=True):
        fr = pd.read_csv(ART / "fairness.csv"); gaps = fr.groupby("attribute").flag_rate.agg(lambda s: (s.max() - s.min()) * 100)
        ch("", "Fairness audit", "Flag-rate gap between groups at the production threshold. Gender is excluded from the model and used only for auditing.")
        H(" ".join(chip(f"{a}: {g:.1f} pp gap", "g" if g < 5 else "a") for a, g in gaps.items()), unsafe_allow_html=True); st.dataframe(fr, hide_index=True)
    H(FOOT, unsafe_allow_html=True)

elif page == "Data sources":
    head("Data sources", "The alternative data used to enrich credit assessments.", "ALTERNATIVE DATA NETWORK")
    S = [("Instalment payments", "installments_payments", "Utility and bill-payment behaviour (proxy)", "inst_", sample.inst_cnt > 0),
         ("Card and cash-flow activity", "credit_card_balance, POS_CASH_balance", "Transaction and account activity (UPI proxy)", ("cc_", "pos_"), (sample.cc_cnt > 0) | (sample.pos_cnt > 0)),
         ("External credit history", "bureau", "Credit bureau records", "bureau_", sample.bureau_cnt > 0),
         ("Loan history", "previous_application", "Previous application outcomes", "prev_", sample.prev_cnt > 0)]
    for col, (t, src, d, pre, m) in zip(st.columns(4, gap="medium"), S):
        with col.container(border=True):
            H(f'{chip("In model")}<h4 style="margin:12px 0 2px">{t}</h4><small style="color:#8a958f">{src}</small><p class="sb" style="margin-top:8px">{d}</p>'
              f'<div class="row"><small>Coverage (holdout)</small><b>{m.mean() * 100:.0f}%</b></div><div class="row"><small>Model features</small><b>{sum(c.startswith(pre) for c in FE)}</b></div>', unsafe_allow_html=True)
    st.info("This build uses the public Home Credit dataset. In production, these sources would be replaced by consented feeds such as UPI transactions, utility and telecom payments.")
    H(FOOT, unsafe_allow_html=True)

else:
    head("Settings", "Model configuration and audit trail.", "ADMINISTRATION")
    with st.container(border=True):
        ch("", "Model configuration")
        H(f'<div class="row"><b>Model version</b><span>{meta["model_version"]}</span></div><div class="row"><b>Flag threshold (High risk)</b><span>{THR:.3f}</span></div><div class="row"><b>Low-risk cut-off</b><span>{LOW:.3f}</span></div><div class="row"><b>Training applicants</b><span>{meta["n_train"]:,}</span></div><div class="row"><b>Features used</b><span>{len(FE)}</span></div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(meta["bands"]).set_index("band"))
    with st.container(border=True):
        ch("", "Audit log", "Every analyst decision, with model version and rationale.")
        d = pd.read_sql("select * from audit order by ts desc", db()); st.dataframe(d, hide_index=True); st.download_button("Export audit log", d.to_csv(index=False), "audit_log.csv")
    H(FOOT, unsafe_allow_html=True)
