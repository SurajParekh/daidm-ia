"""
VerifAI Analytics Dashboard — Streamlit App
============================================
Modules:
  A — Association Rule Mining (Apriori from scratch)
  B — K-Means Customer Persona Clustering
  C — Subscription Prediction (Logistic Regression + Random Forest)
  D — MRR Revenue Regression (Gradient Boosting)

Run:  streamlit run app.py
"""

import io
import os
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & THEME
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VerifAI Analytics Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background: #0A0E1A; }
  [data-testid="stSidebar"] { background: #111827; border-right: 1px solid #2D3748; }
  [data-testid="stSidebar"] * { color: #F0F4FF !important; }

  .metric-card {
    background: #1A2235; border: 1px solid #2D3748; border-radius: 10px;
    padding: 16px 20px; text-align: center; margin: 4px 0;
  }
  .metric-val { font-size: 2rem; font-weight: 700; color: #6366F1; line-height: 1.1; }
  .metric-lbl { font-size: 0.78rem; color: #8892A4; margin-top: 4px; letter-spacing: .05em; text-transform: uppercase; }

  .section-header {
    background: linear-gradient(90deg, #6366F1 0%, #2DD4BF 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 1.5rem; font-weight: 700; margin: 8px 0 4px;
  }
  .insight-box {
    background: #1A2235; border-left: 3px solid #6366F1;
    border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0;
    font-size: 0.9rem; color: #E2E8F0; line-height: 1.6;
  }
  .insight-box.teal  { border-color: #2DD4BF; }
  .insight-box.rose  { border-color: #FB7185; }
  .insight-box.amber { border-color: #FCD34D; }
  .tier-pill {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600; margin: 2px;
  }
  .stDataFrame { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
  div[data-testid="stExpander"] { background: #111827; border: 1px solid #2D3748; border-radius: 8px; }
  .stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 8px; gap: 4px; }
  .stTabs [data-baseweb="tab"] { background: #1A2235; color: #8892A4; border-radius: 6px; }
  .stTabs [aria-selected="true"] { background: #6366F1 !important; color: #fff !important; }
  h1, h2, h3 { color: #F0F4FF !important; }
  p, li, label { color: #CBD5E0; }
  hr { border-color: #2D3748; }
</style>
""", unsafe_allow_html=True)

# ── Shared matplotlib style ───────────────────────────────────────────────────
C_DARK, C_PANEL, C_CARD  = "#0A0E1A", "#111827", "#1A2235"
C_BORDER, C_TEXT, C_MUTED = "#2D3748", "#F0F4FF", "#8892A4"
C_ACCENT, C_TEAL, C_ROSE  = "#6366F1", "#2DD4BF", "#FB7185"
C_AMBER, C_VIOLET, C_GREEN = "#FCD34D", "#A78BFA", "#4ADE80"
CLUSTER_COLORS = [C_ACCENT, C_TEAL, C_ROSE, C_AMBER, C_VIOLET, C_GREEN, "#F97316", "#38BDF8"]

plt.rcParams.update({
    "figure.facecolor": C_DARK, "axes.facecolor": C_PANEL,
    "axes.edgecolor": C_BORDER, "axes.labelcolor": C_TEXT,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED,
    "text.color": C_TEXT, "grid.color": C_BORDER,
    "font.family": "DejaVu Sans", "font.size": 11,
})

def fig_to_st(fig):
    """Render a matplotlib figure in Streamlit."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR & DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔬 VerifAI Analytics")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload survey CSV",
        type=["csv"],
        help="Upload verifai_survey_data.csv (or any compatible dataset)",
    )

    # Fallback: look for the file next to app.py (repo-bundled)
    FALLBACK_PATHS = [
        os.path.join(os.path.dirname(__file__), "verifai_survey_data.csv"),
        "verifai_survey_data.csv",
        os.path.join(os.path.dirname(__file__), "data", "verifai_survey_data.csv"),
    ]

    df = None
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df):,} rows from upload")
    else:
        for fp in FALLBACK_PATHS:
            if os.path.exists(fp):
                df = pd.read_csv(fp)
                st.info(f"📂 Using bundled dataset ({len(df):,} rows)")
                break

    if df is None:
        st.warning("⬆️  Please upload `verifai_survey_data.csv` to begin.")
        st.markdown("""
        **Expected columns include:**
        - `user_type`, `industry`, `region`, `age_group`
        - `monthly_spend_on_ai_usd`, `wtp_monthly_usd`
        - `will_subscribe`, `estimated_mrr_usd`
        - 7 binary `uses_*` tool columns
        """)
        st.stop()

    st.markdown("---")
    st.markdown("**Navigate modules:**")
    module = st.radio(
        "",
        ["🏠 Overview",
         "🔗 Module A — Association Rules",
         "👥 Module B — Clustering",
         "🎯 Module C — Subscription Model",
         "💰 Module D — MRR Regression"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Settings**")
    MIN_SUPPORT    = st.slider("Min Support (Apriori)",    0.10, 0.40, 0.15, 0.01)
    MIN_CONFIDENCE = st.slider("Min Confidence (Apriori)", 0.40, 0.90, 0.50, 0.05)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
TOOL_COLS = ["uses_llm_writing_tools","uses_image_gen_tools","uses_video_gen_tools",
             "uses_code_assistant_tools","uses_data_analytics_tools",
             "uses_voice_audio_tools","uses_productivity_tools"]
TOOL_LABELS = {
    "uses_llm_writing_tools":    "LLM Writing",
    "uses_image_gen_tools":      "Image Gen",
    "uses_video_gen_tools":      "Video Gen",
    "uses_code_assistant_tools": "Code Assistant",
    "uses_data_analytics_tools": "Data Analytics",
    "uses_voice_audio_tools":    "Voice / Audio",
    "uses_productivity_tools":   "Productivity",
}
ALL_ORDINALS = {
    "difficulty_finding_tools":    {"Very Easy":0,"Easy":1,"Neutral":2,"Difficult":3,"Very Difficult":4},
    "trust_concern_frequency":     {"Never":0,"Rarely":1,"Sometimes":2,"Often":3,"Always":4},
    "subscription_fatigue":        {"Strongly Disagree":0,"Disagree":1,"Neutral":2,"Agree":3,"Strongly Agree":4},
    "preferred_subscription_tier": {"Free":0,"Silver":1,"Gold":2,"Platinum":3},
    "preferred_billing_cadence":   {"Monthly":0,"Half-Yearly":1,"Yearly":2},
    "bundle_interest":             {"Not Interested":0,"Neutral":1,"Interested":2,"Very Interested":3},
    "age_group":                   {"18-24":0,"25-34":1,"35-44":2,"45-54":3,"55+":4},
}
LABEL_REMAP = {
    "wtp_monthly_usd":"WTP Monthly (USD)","preferred_subscription_tier":"Preferred Tier",
    "monthly_spend_on_ai_usd":"Monthly AI Spend","num_ai_tools_currently_used":"# Tools Used",
    "trust_concern_frequency":"Trust Concern","difficulty_finding_tools":"Discovery Difficulty",
    "subscription_fatigue":"Sub Fatigue","bundle_interest":"Bundle Interest",
    "preferred_billing_cadence":"Billing Cadence","age_group":"Age Group",
}
TIER_ORDER     = ["Free","Silver","Gold","Platinum"]
TIER_COLOR_MAP = {"Free":C_MUTED,"Silver":C_TEAL,"Gold":C_AMBER,"Platinum":C_ROSE}

# ── Preprocess df ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def preprocess(df_raw):
    df = df_raw.copy()
    for col, mp in ALL_ORDINALS.items():
        if col in df.columns:
            df[col + "_enc"] = df[col].map(mp)
    return df

df = preprocess(df)
N  = len(df)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: build ML feature matrix
# ══════════════════════════════════════════════════════════════════════════════
def build_feature_matrix(df_in, drop_cols):
    Xm = df_in.drop(columns=[c for c in drop_cols if c in df_in.columns]).copy()
    for col, mp in ALL_ORDINALS.items():
        if col in Xm.columns:
            Xm[col] = Xm[col].map(mp)
    enc_cols = [c for c in Xm.columns if c.endswith("_enc")]
    Xm = Xm.drop(columns=enc_cols, errors="ignore")
    Xm = pd.get_dummies(Xm, columns=[c for c in ["user_type","industry","region"] if c in Xm.columns], drop_first=True)
    Xm = Xm.apply(pd.to_numeric, errors="coerce").fillna(0)
    return Xm


# ══════════════════════════════════════════════════════════════════════════════
#  OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if module == "🏠 Overview":
    st.markdown('<h1 style="margin-bottom:0">VerifAI Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8892A4;margin-top:0">AI Marketplace · Customer Intelligence · Revenue Forecasting</p>', unsafe_allow_html=True)
    st.markdown("---")

    c1,c2,c3,c4,c5 = st.columns(5)
    cards = [
        (c1, f"{N:,}", "Total Respondents"),
        (c2, f"{df['will_subscribe'].mean():.1%}", "Subscription Rate"),
        (c3, f"${df['wtp_monthly_usd'].mean():.0f}", "Avg WTP / Month"),
        (c4, f"${df['estimated_mrr_usd'].mean():.0f}", "Avg MRR / User"),
        (c5, f"${df['monthly_spend_on_ai_usd'].mean():.0f}", "Avg AI Spend"),
    ]
    for col, val, lbl in cards:
        col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### User Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=C_DARK)
        for ax, col_name, title in [
            (axes[0], "user_type", "By User Type"),
            (axes[1], "preferred_subscription_tier", "By Preferred Tier"),
        ]:
            vc = df[col_name].value_counts()
            colors_ = CLUSTER_COLORS[:len(vc)]
            ax.pie(vc.values, labels=vc.index, colors=colors_, autopct="%1.0f%%",
                   textprops={"color":C_TEXT,"fontsize":8}, pctdistance=0.75,
                   wedgeprops={"edgecolor":C_DARK,"linewidth":1.5})
            ax.set_title(title, color=C_TEXT, fontsize=10, fontweight="bold")
            ax.set_facecolor(C_DARK)
        fig.patch.set_facecolor(C_DARK)
        fig_to_st(fig)

    with col_r:
        st.markdown("#### MRR by Subscription Tier")
        fig, ax = plt.subplots(figsize=(6, 4), facecolor=C_DARK)
        tier_mrr = df.groupby("preferred_subscription_tier")["estimated_mrr_usd"].mean().reindex(TIER_ORDER)
        bars = ax.bar(TIER_ORDER, tier_mrr, color=[TIER_COLOR_MAP[t] for t in TIER_ORDER], alpha=0.85, edgecolor=C_BORDER)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"${bar.get_height():.0f}", ha="center", va="bottom", fontsize=10, color=C_TEXT)
        ax.set_ylabel("Avg MRR (USD)", color=C_TEXT)
        ax.set_title("Average MRR per User by Tier", color=C_TEXT, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.15)
        fig_to_st(fig)

    st.markdown("---")
    st.markdown("#### Quick Data Preview")
    st.dataframe(df.head(10), use_container_width=True, height=280)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE A — ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════════════════
elif module == "🔗 Module A — Association Rules":
    st.markdown('<div class="section-header">Module A — Association Rule Mining</div>', unsafe_allow_html=True)
    st.caption(f"Apriori algorithm · min support={MIN_SUPPORT} · min confidence={MIN_CONFIDENCE}")

    @st.cache_data(show_spinner="Mining frequent itemsets…")
    def run_apriori(df_hash, min_sup, min_conf):
        basket = df[TOOL_COLS].rename(columns=TOOL_LABELS)
        items  = list(TOOL_LABELS.values())
        txns   = [frozenset(i for i in items if row[i]==1) for _, row in basket.iterrows()]

        def sup(iset): return sum(1 for t in txns if iset.issubset(t)) / len(txns)

        freq = {}
        L1 = {frozenset([i]): sup(frozenset([i])) for i in items}
        L1 = {k:v for k,v in L1.items() if v >= min_sup}
        freq.update(L1); Lk = L1; k = 2
        while Lk:
            cands = set()
            ll = list(Lk.keys())
            for i in range(len(ll)):
                for j in range(i+1,len(ll)):
                    u = ll[i]|ll[j]
                    if len(u)==k: cands.add(u)
            Lk = {c:sup(c) for c in cands if sup(c)>=min_sup}
            freq.update(Lk); k+=1

        rules = []
        for iset, s_is in freq.items():
            if len(iset)<2: continue
            for sz in range(1,len(iset)):
                for ant in map(frozenset, itertools.combinations(iset,sz)):
                    con = iset-ant
                    s_a = freq.get(ant, sup(ant)); s_c = freq.get(con, sup(con))
                    conf = s_is/s_a if s_a>0 else 0
                    lift = conf/s_c if s_c>0 else 0
                    if conf>=min_conf:
                        rules.append({"antecedent":", ".join(sorted(ant)),
                                      "consequent":", ".join(sorted(con)),
                                      "support":round(s_is,4),"confidence":round(conf,4),"lift":round(lift,4)})

        rdf = pd.DataFrame(rules).drop_duplicates().sort_values("lift",ascending=False).reset_index(drop=True)

        # Co-occurrence matrix
        labels = list(TOOL_LABELS.values())
        co = pd.DataFrame(0.0, index=labels, columns=labels)
        for t in txns:
            for a,b in itertools.product(list(t),list(t)): co.loc[a,b]+=1
        co /= len(txns)
        return rdf, co, txns

    rules_df, co_matrix, txns = run_apriori(hash(str(df.shape)), MIN_SUPPORT, MIN_CONFIDENCE)

    # ── KPIs ────────────────────────────────────────────────────────────────
    k1,k2,k3 = st.columns(3)
    for col, val, lbl in [
        (k1, len([x for x in rules_df.iterrows()]), "Total Rules"),
        (k2, f"{rules_df['lift'].max():.3f}", "Max Lift"),
        (k3, f"{rules_df['confidence'].mean():.3f}", "Avg Confidence"),
    ]:
        col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["🔥 Co-occurrence Heatmap", "📊 Rules Chart", "📋 Rules Table"])

    with tab1:
        fig, ax = plt.subplots(figsize=(9,7), facecolor=C_DARK)
        cmap_h = LinearSegmentedColormap.from_list("v",[C_PANEL,C_ACCENT,C_TEAL],N=256)
        im = ax.imshow(co_matrix.values, cmap=cmap_h, aspect="auto", vmin=0, vmax=co_matrix.values.max())
        tks = range(len(co_matrix))
        ax.set_xticks(tks); ax.set_xticklabels(co_matrix.columns, rotation=35, ha="right", fontsize=10)
        ax.set_yticks(tks); ax.set_yticklabels(co_matrix.index, fontsize=10)
        for i in range(len(co_matrix)):
            for j in range(len(co_matrix)):
                v = co_matrix.values[i,j]; br = v/(co_matrix.values.max()+1e-9)
                ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=8.5,
                        color=C_DARK if br>0.5 else C_TEXT, fontweight="bold" if i==j else "normal")
        plt.colorbar(im,ax=ax,fraction=0.03,pad=0.01).set_label("Co-occurrence Rate",color=C_MUTED,fontsize=9)
        ax.set_title("Tool Co-occurrence Heatmap",color=C_TEXT,fontsize=13,fontweight="bold")
        fig.tight_layout()
        fig_to_st(fig)

    with tab2:
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,6),facecolor=C_DARK)
        # Lift bar
        top_n = min(15, len(rules_df))
        pr = rules_df.head(top_n).copy()
        pr["lbl"] = pr.apply(lambda r: f"{r['antecedent'][:20]} → {r['consequent'][:12]}", axis=1)
        bc = [CLUSTER_COLORS[i%len(CLUSTER_COLORS)] for i in range(top_n)]
        ax1.barh(range(top_n), pr["lift"], color=bc, alpha=0.85, height=0.7)
        ax1.set_yticks(range(top_n)); ax1.set_yticklabels(pr["lbl"],fontsize=8)
        ax1.invert_yaxis(); ax1.axvline(1.0,color=C_ROSE,lw=1.5,ls="--",alpha=0.7)
        ax1.set_xlabel("Lift"); ax1.set_title(f"Top {top_n} Rules by Lift",color=C_TEXT,fontsize=12,fontweight="bold")
        ax1.grid(True,axis="x",alpha=0.15)
        # Bubble
        sc = ax2.scatter(rules_df["support"],rules_df["confidence"],c=rules_df["lift"],
                         cmap="plasma",s=rules_df["lift"]*100,alpha=0.75,edgecolors=C_BORDER,linewidths=0.5)
        plt.colorbar(sc,ax=ax2,fraction=0.04).set_label("Lift",color=C_MUTED)
        ax2.axhline(MIN_CONFIDENCE,color=C_ROSE,lw=1.2,ls="--",alpha=0.6)
        ax2.axvline(MIN_SUPPORT,color=C_AMBER,lw=1.2,ls="--",alpha=0.6)
        ax2.set_xlabel("Support"); ax2.set_ylabel("Confidence")
        ax2.set_title("Support vs Confidence (size=Lift)",color=C_TEXT,fontsize=12,fontweight="bold")
        ax2.grid(True,alpha=0.15)
        fig.tight_layout(); fig_to_st(fig)

    with tab3:
        st.dataframe(rules_df.style.format({"support":"{:.3f}","confidence":"{:.3f}","lift":"{:.3f}"}),
                     use_container_width=True, height=400)
        csv = rules_df.to_csv(index=False).encode()
        st.download_button("⬇️  Download Rules CSV", csv, "verifai_rules.csv", "text/csv")

    # ── Bundle recommendations ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🎁 Top 5 Bundle Recommendations by Lift")
    BUNDLES = [
        {"name":"🖥️ Full-Stack Dev Suite","tools":["LLM Writing","Code Assistant","Data Analytics"],
         "persona":"Software & data engineers","pitch":"Every tool in your daily dev loop — one verified stack."},
        {"name":"🎨 Creator Studio","tools":["Image Gen","Video Gen","Productivity"],
         "persona":"Designers & visual creators","pitch":"From concept to publish — verified creative tools."},
        {"name":"📊 Analyst Intelligence Pack","tools":["Data Analytics","LLM Writing","Code Assistant"],
         "persona":"Data analysts & growth marketers","pitch":"Query it. Model it. Explain it. One price."},
        {"name":"✍️ Content & Copy Engine","tools":["LLM Writing","Productivity","Image Gen"],
         "persona":"Freelancers & marketers","pitch":"Write faster, stay organised, make it visual."},
        {"name":"🎙️ Media Production Bundle","tools":["Voice / Audio","Video Gen","Image Gen"],
         "persona":"Podcast & video creators","pitch":"Studio-quality AI tools — curated & trusted."},
    ]
    cols = st.columns(len(BUNDLES))
    for i, (col, b) in enumerate(zip(cols, BUNDLES)):
        tool_cols_in = [k for k,v in TOOL_LABELS.items() if v in b["tools"]]
        valid_cols   = [c for c in tool_cols_in if c in df.columns]
        if valid_cols:
            mask   = (df[valid_cols]==1).all(axis=1)
            n_u    = mask.sum(); pct = n_u/N*100
            avg_wtp = df.loc[mask,"wtp_monthly_usd"].mean() if n_u>0 else 0
        else:
            n_u=0; pct=0; avg_wtp=0
        rule = rules_df.iloc[i] if i < len(rules_df) else None
        lift_str = f"Lift {rule['lift']:.3f}" if rule is not None else "—"
        col.markdown(f"""
        <div class="metric-card" style="text-align:left;padding:12px">
          <div style="font-weight:700;font-size:0.95rem;color:#F0F4FF">{b['name']}</div>
          <div style="font-size:0.75rem;color:#6366F1;margin:4px 0">{lift_str}</div>
          <div style="font-size:0.78rem;color:#8892A4">{b['persona']}</div>
          <div style="font-size:0.78rem;color:#CBD5E0;margin:6px 0">{b['pitch']}</div>
          <hr style="border-color:#2D3748;margin:6px 0">
          <div style="font-size:0.78rem;color:#2DD4BF"><b>{n_u}</b> users ({pct:.1f}%)</div>
          <div style="font-size:0.78rem;color:#FCD34D">Avg WTP ${avg_wtp:.0f}/mo</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE B — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif module == "👥 Module B — Clustering":
    st.markdown('<div class="section-header">Module B — Customer Persona Clustering</div>', unsafe_allow_html=True)

    CLUSTER_FEATURES = ["monthly_spend_on_ai_usd","wtp_monthly_usd","num_ai_tools_currently_used",
                        "difficulty_finding_tools_enc","trust_concern_frequency_enc",
                        "subscription_fatigue_enc","preferred_subscription_tier_enc"]
    avail_feat = [f for f in CLUSTER_FEATURES if f in df.columns]

    @st.cache_data(show_spinner="Running K-Means sweep…")
    def run_clustering(df_hash):
        Xc = df[avail_feat].dropna(); cidx = Xc.index
        sc = StandardScaler(); Xs = sc.fit_transform(Xc)
        K  = range(2,9); iner=[]; sils=[]
        for k in K:
            km=KMeans(n_clusters=k,random_state=42,n_init=20)
            lb=km.fit_predict(Xs); iner.append(km.inertia_); sils.append(silhouette_score(Xs,lb))
        bk_math = K.start+int(np.argmax(sils))
        bk = max([(k,s) for k,s in zip(K,sils) if k>=4],key=lambda x:x[1])[0]
        km_best=KMeans(n_clusters=bk,random_state=42,n_init=30)
        labs=km_best.fit_predict(Xs)
        df_c=df.loc[cidx].copy(); df_c["cluster"]=labs
        prof=df_c.groupby("cluster").agg(
            size=("cluster","count"),avg_spend=("monthly_spend_on_ai_usd","mean"),
            avg_wtp=("wtp_monthly_usd","mean"),avg_tools=("num_ai_tools_currently_used","mean"),
            sub_rate=("will_subscribe","mean"),avg_mrr=("estimated_mrr_usd","mean")).round(2)
        pca=PCA(n_components=2,random_state=42); Xp=pca.fit_transform(Xs)
        return df_c,prof,list(K),iner,sils,bk_math,bk,Xs,labs,Xp,pca.explained_variance_ratio_

    df_c,profile,k_list,inertias,sils,bk_math,best_k,Xs,cluster_labels,Xp,vexp = run_clustering(hash(str(df.shape)))

    def name_persona(row, cid):
        fat = df_c[df_c["cluster"]==cid]["subscription_fatigue_enc"].mean() if "subscription_fatigue_enc" in df_c else 2
        if row["avg_wtp"]>80 and row["sub_rate"]>0.95: return ("High-Intent Power Buyers","💎",CLUSTER_COLORS[0])
        if row["avg_spend"]>300 and row["avg_wtp"]<50: return ("Heavy Spenders, Low Commitment","🏢",CLUSTER_COLORS[1])
        if fat>2.8 and row["avg_wtp"]<35:              return ("Subscription-Fatigued Skeptics","😤",CLUSTER_COLORS[2])
        if row["avg_tools"]>9 and row["sub_rate"]>0.55:return ("AI Power Users","⚡",CLUSTER_COLORS[3])
        if row["avg_tools"]<4:                         return ("Cautious Early Adopters","🌱",CLUSTER_COLORS[4])
        if row["sub_rate"]>0.80 and row["avg_wtp"]>30: return ("Trust-Driven Converters","🔐",CLUSTER_COLORS[5])
        if row["avg_spend"]>100 and row["sub_rate"]>0.65: return ("Growth-Stage Adopters","🚀",CLUSTER_COLORS[6])
        return ("Budget-Conscious Explorers","🧭",CLUSTER_COLORS[7])

    persona_info = {cid: name_persona(profile.loc[cid],cid) for cid in sorted(df_c["cluster"].unique())}
    profile["persona"]=[ persona_info[c][0] for c in profile.index]
    profile["icon"]   =[ persona_info[c][1] for c in profile.index]
    profile["color"]  =[ persona_info[c][2] for c in profile.index]

    # ── KPI row ─────────────────────────────────────────────────────────────
    kc1,kc2,kc3 = st.columns(3)
    kc1.markdown(f'<div class="metric-card"><div class="metric-val">{best_k}</div><div class="metric-lbl">Business Optimal k</div></div>',unsafe_allow_html=True)
    kc2.markdown(f'<div class="metric-card"><div class="metric-val">{sils[best_k-k_list[0]]:.3f}</div><div class="metric-lbl">Silhouette Score</div></div>',unsafe_allow_html=True)
    kc3.markdown(f'<div class="metric-card"><div class="metric-val">{best_k}</div><div class="metric-lbl">Distinct Personas</div></div>',unsafe_allow_html=True)

    st.markdown("---")
    tab1,tab2,tab3 = st.tabs(["📊 Cluster Selection","🗺️ Persona Map","📋 Profile Table"])

    with tab1:
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4),facecolor=C_DARK)
        bc=[C_TEAL if k==best_k else C_BORDER for k in k_list]
        bars_s=ax1.bar(k_list,sils,color=bc,edgecolor=C_BORDER,width=0.6)
        for bar,sc_,k in zip(bars_s,sils,k_list):
            ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.003,f"{sc_:.3f}",
                     ha="center",va="bottom",fontsize=9,color=C_TEXT,fontweight="bold" if k==best_k else "normal")
        ax1.set_xlabel("k"); ax1.set_ylabel("Silhouette Score")
        ax1.set_title("Silhouette Scores",color=C_TEXT,fontsize=12,fontweight="bold")
        ax1.set_xticks(k_list); ax1.grid(True,axis="y",alpha=0.15)
        ax2.plot(k_list,inertias,color=C_ACCENT,lw=2.5,marker="o",markersize=7,
                 markerfacecolor=C_DARK,markeredgecolor=C_ACCENT,markeredgewidth=2)
        ax2.scatter([best_k],[inertias[best_k-k_list[0]]],color=C_TEAL,s=120,zorder=5)
        ax2.set_xlabel("k"); ax2.set_ylabel("Inertia")
        ax2.set_title("Elbow Curve",color=C_TEXT,fontsize=12,fontweight="bold")
        ax2.set_xticks(k_list); ax2.grid(True,alpha=0.15)
        fig.tight_layout(); fig_to_st(fig)
        st.markdown(f'<div class="insight-box">Math optimal k={bk_math} (silhouette={max(sils):.3f}) is too coarse for GTM. <b>Business optimal k={best_k}</b> used — richest segmentation with interpretable personas.</div>',unsafe_allow_html=True)

    with tab2:
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6),facecolor=C_DARK)
        for cid in sorted(df_c["cluster"].unique()):
            mask=df_c["cluster"]==cid; col=persona_info[cid][2]
            ax1.scatter(df_c.loc[mask,"monthly_spend_on_ai_usd"],df_c.loc[mask,"wtp_monthly_usd"],
                        c=col,alpha=0.45,s=40,edgecolors="none",
                        label=f"{persona_info[cid][1]} {persona_info[cid][0]} (n={mask.sum()})")
            cx=df_c.loc[mask,"monthly_spend_on_ai_usd"].mean(); cy=df_c.loc[mask,"wtp_monthly_usd"].mean()
            ax1.scatter(cx,cy,c=col,s=220,marker="D",edgecolors=C_TEXT,linewidths=1.5,zorder=10)
            ax1.annotate(f"{persona_info[cid][1]}",xy=(cx,cy),ha="center",va="center",fontsize=13,zorder=11)
        ax1.set_xlabel("Monthly AI Spend (USD)"); ax1.set_ylabel("WTP / Month (USD)")
        ax1.set_title("Spend vs WTP by Persona (◆=centroid)",color=C_TEXT,fontsize=11,fontweight="bold")
        ax1.legend(fontsize=7.5,facecolor=C_PANEL,edgecolor=C_BORDER,labelcolor=C_TEXT,loc="upper left")
        ax1.grid(True,alpha=0.12)
        for cid in sorted(df_c["cluster"].unique()):
            mask=(cluster_labels==cid); col=persona_info[cid][2]
            ax2.scatter(Xp[mask,0],Xp[mask,1],c=col,alpha=0.45,s=35,edgecolors="none")
            cx,cy=Xp[mask,0].mean(),Xp[mask,1].mean()
            ax2.scatter(cx,cy,c=col,s=180,marker="D",edgecolors=C_TEXT,linewidths=1.5,zorder=10)
            ax2.text(cx,cy+0.15,persona_info[cid][1],ha="center",fontsize=12)
        ax2.set_xlabel(f"PC1 ({vexp[0]:.1%})"); ax2.set_ylabel(f"PC2 ({vexp[1]:.1%})")
        ax2.set_title("PCA Projection",color=C_TEXT,fontsize=11,fontweight="bold")
        ax2.grid(True,alpha=0.12)
        fig.tight_layout(); fig_to_st(fig)

    with tab3:
        display = profile[["icon","persona","size","avg_spend","avg_wtp","avg_tools","sub_rate","avg_mrr"]].copy()
        display.columns=["Icon","Persona","Users","Avg Spend $","Avg WTP $","Avg Tools","Sub Rate","Avg MRR $"]
        display["Sub Rate"]=display["Sub Rate"].apply(lambda x:f"{x:.0%}")
        st.dataframe(display, use_container_width=True, height=260)
        for cid in sorted(df_c["cluster"].unique()):
            row=profile.loc[cid]; pi=persona_info[cid]
            st.markdown(f'<div class="insight-box">'
                        f'<b>{pi[1]} {pi[0]}</b> · n={int(row["size"])} · '
                        f'Avg WTP ${row["avg_wtp"]:.0f} · Sub rate {row["sub_rate"]:.0%} · Avg MRR ${row["avg_mrr"]:.0f}'
                        f'</div>',unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE C — SUBSCRIPTION PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif module == "🎯 Module C — Subscription Model":
    st.markdown('<div class="section-header">Module C — Subscription Prediction</div>', unsafe_allow_html=True)

    @st.cache_data(show_spinner="Training classifiers…")
    def run_classification(df_hash):
        drop = ["respondent_id","primary_discovery_method","churn_risk_score","estimated_mrr_usd"]
        Xm   = build_feature_matrix(df.drop(columns=[c for c in drop if c in df.columns]),["will_subscribe"])
        y_   = df["will_subscribe"]
        Xtr,Xte,ytr,yte = train_test_split(Xm,y_,test_size=0.2,random_state=42,stratify=y_)
        lr=Pipeline([("sc",StandardScaler()),("clf",LogisticRegression(max_iter=1000,C=1.0,random_state=42,class_weight="balanced"))])
        lr.fit(Xtr,ytr); lp=lr.predict(Xte); lb=lr.predict_proba(Xte)[:,1]
        rf=RandomForestClassifier(n_estimators=300,min_samples_leaf=5,random_state=42,class_weight="balanced",n_jobs=-1)
        rf.fit(Xtr,ytr); rp=rf.predict(Xte); rb=rf.predict_proba(Xte)[:,1]
        fi=pd.DataFrame({"feature":Xm.columns,"importance":rf.feature_importances_})
        fi=fi.sort_values("importance",ascending=False).head(20).reset_index(drop=True)
        fi["label"]=fi["feature"].apply(lambda x:LABEL_REMAP.get(x,x.replace("user_type_","User: ").replace("industry_","Ind: ").replace("region_","Reg: ").replace("_"," ").title()))
        return Xte,yte,lp,lb,rp,rb,fi,Xm.columns.tolist()

    Xte,yte,lp,lb,rp,rb,fi_df,feat_names = run_classification(hash(str(df.shape)))

    def m(yt,yp,ypr):
        return {"Accuracy":accuracy_score(yt,yp),"Precision":precision_score(yt,yp),
                "Recall":recall_score(yt,yp),"F1":f1_score(yt,yp),"ROC-AUC":roc_auc_score(yt,ypr)}

    lm=m(yte,lp,lb); rm=m(yte,rp,rb)

    # KPIs
    kc=st.columns(5)
    for col,key in zip(kc,["Accuracy","Precision","Recall","F1","ROC-AUC"]):
        col.markdown(f'<div class="metric-card"><div class="metric-val" style="font-size:1.4rem">{rm[key]:.3f}</div><div class="metric-lbl">RF {key}</div></div>',unsafe_allow_html=True)

    st.markdown("---")
    tab1,tab2,tab3 = st.tabs(["🎯 Confusion Matrices","📈 ROC & Metrics","🔑 Feature Importance"])

    with tab1:
        fig,axes=plt.subplots(1,2,figsize=(10,4),facecolor=C_DARK)
        for ax,cm_,name,color in [
            (axes[0],confusion_matrix(yte,lp),"Logistic Regression",C_ACCENT),
            (axes[1],confusion_matrix(yte,rp),"Random Forest",C_TEAL)]:
            cmap_=LinearSegmentedColormap.from_list("cm",[C_PANEL,color],N=256)
            ax.imshow(cm_,cmap=cmap_,aspect="auto")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["Pred No","Pred Yes"],color=C_TEXT,fontsize=10)
            ax.set_yticklabels(["Actual No","Actual Yes"],color=C_TEXT,fontsize=10)
            for i in range(2):
                for j in range(2):
                    ax.text(j,i,f"{cm_[i,j]}\n({cm_[i,j]/cm_.sum()*100:.1f}%)",
                            ha="center",va="center",fontsize=12,fontweight="bold",
                            color=C_DARK if cm_[i,j]>cm_.max()*0.5 else C_TEXT)
            ax.set_title(name,color=C_TEXT,fontsize=12,fontweight="bold")
        fig.tight_layout(); fig_to_st(fig)

    with tab2:
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4),facecolor=C_DARK)
        for prob,label,col in [(lb,f"LR (AUC={roc_auc_score(yte,lb):.3f})",C_ACCENT),(rb,f"RF (AUC={roc_auc_score(yte,rb):.3f})",C_TEAL)]:
            fpr,tpr,_=roc_curve(yte,prob); ax1.plot(fpr,tpr,color=col,lw=2.5,label=label)
        ax1.plot([0,1],[0,1],"--",color=C_MUTED,lw=1.2); ax1.fill_between(*roc_curve(yte,rb)[:2],alpha=0.06,color=C_TEAL)
        ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title("ROC Curves",color=C_TEXT,fontsize=12,fontweight="bold")
        ax1.legend(fontsize=9,facecolor=C_PANEL,edgecolor=C_BORDER,labelcolor=C_TEXT); ax1.grid(True,alpha=0.15)
        mlabels=list(lm.keys()); x__=np.arange(len(mlabels)); w__=0.35
        for vals,color,lbl in [(list(lm.values()),C_ACCENT,"LR"),(list(rm.values()),C_TEAL,"RF")]:
            off=-w__/2 if color==C_ACCENT else w__/2
            b_=ax2.bar(x__+off,vals,w__,color=color,alpha=0.85,label=lbl)
            for bar in b_: ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,f"{bar.get_height():.3f}",ha="center",va="bottom",fontsize=8,color=C_TEXT)
        ax2.set_xticks(x__); ax2.set_xticklabels(mlabels,fontsize=10); ax2.set_ylim(0,1.14)
        ax2.set_title("Metrics Comparison",color=C_TEXT,fontsize=12,fontweight="bold")
        ax2.legend(fontsize=9,facecolor=C_PANEL,edgecolor=C_BORDER,labelcolor=C_TEXT); ax2.grid(True,axis="y",alpha=0.15)
        fig.tight_layout(); fig_to_st(fig)

    with tab3:
        fig,ax=plt.subplots(figsize=(11,7),facecolor=C_DARK)
        fi_cols=[]
        for feat in fi_df["feature"]:
            if feat in ["wtp_monthly_usd","preferred_subscription_tier","monthly_spend_on_ai_usd"]: fi_cols.append(C_AMBER)
            elif feat in ["trust_concern_frequency","difficulty_finding_tools"]: fi_cols.append(C_ROSE)
            elif feat in ["subscription_fatigue","bundle_interest","preferred_billing_cadence"]: fi_cols.append(C_VIOLET)
            elif feat.startswith("uses_"): fi_cols.append(C_TEAL)
            else: fi_cols.append(C_ACCENT)
        ax.barh(range(len(fi_df)),fi_df["importance"],color=fi_cols,alpha=0.87,height=0.72)
        ax.set_yticks(range(len(fi_df))); ax.set_yticklabels(fi_df["label"],fontsize=10)
        ax.invert_yaxis()
        for i,v in enumerate(fi_df["importance"]): ax.text(v+0.001,i,f"{v:.4f}",va="center",fontsize=8.5,color=C_MUTED)
        patches=[mpatches.Patch(color=c,label=l) for c,l in [(C_AMBER,"Monetization"),(C_ROSE,"Trust & Discovery"),(C_VIOLET,"Marketplace Fit"),(C_TEAL,"Tool Usage"),(C_ACCENT,"Demographics")]]
        ax.legend(handles=patches,loc="lower right",fontsize=9,facecolor=C_CARD,edgecolor=C_BORDER,labelcolor=C_TEXT)
        ax.set_xlabel("Feature Importance"); ax.set_title("RF Top 20 Feature Importances",color=C_TEXT,fontsize=13,fontweight="bold"); ax.grid(True,axis="x",alpha=0.15)
        fig.tight_layout(); fig_to_st(fig)
        st.markdown(f'<div class="insight-box amber"><b>Top predictor: {fi_df.iloc[0]["label"]}</b> — The ceiling-setter. Anchor value before showing price, not after. Users who self-report high WTP convert at 2× the rate of mid-WTP users and generate 4–6× the MRR.</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="insight-box teal"><b>#2: {fi_df.iloc[1]["label"]}</b> — Structural lever. Moving one tier up roughly triples MRR. Silver→Gold is the single highest-leverage upsell action.</div>',unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE D — MRR REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
elif module == "💰 Module D — MRR Regression":
    st.markdown('<div class="section-header">Module D — MRR Revenue Regression</div>', unsafe_allow_html=True)
    st.caption("Gradient Boosting Regressor · target = estimated_mrr_usd")

    @st.cache_data(show_spinner="Training Gradient Boosting model…")
    def run_regression(df_hash):
        drop=["respondent_id","primary_discovery_method","churn_risk_score","will_subscribe"]
        Xd=build_feature_matrix(df.drop(columns=[c for c in drop if c in df.columns]),["estimated_mrr_usd"])
        yd=df["estimated_mrr_usd"]
        feat=Xd.columns.tolist()
        Xtr,Xte,ytr,yte=train_test_split(Xd,yd,test_size=0.2,random_state=42)
        gbr=GradientBoostingRegressor(n_estimators=500,learning_rate=0.05,max_depth=4,
                                       subsample=0.8,min_samples_leaf=10,random_state=42)
        gbr.fit(Xtr,ytr)
        yp=gbr.predict(Xte); yp_all=gbr.predict(Xd)
        r2=r2_score(yte,yp); mae=mean_absolute_error(yte,yp)
        rmse=np.sqrt(mean_squared_error(yte,yp))
        mape_mask=yte>1.0
        mape=np.mean(np.abs((yte[mape_mask]-yp[mape_mask])/yte[mape_mask]))*100
        fi=pd.DataFrame({"feature":feat,"importance":gbr.feature_importances_})
        fi=fi.sort_values("importance",ascending=False).head(20).reset_index(drop=True)
        fi["label"]=fi["feature"].apply(lambda x:LABEL_REMAP.get(x,x.replace("user_type_","User: ").replace("industry_","Ind: ").replace("region_","Reg: ").replace("_"," ").title()))
        return Xte,yte,yp,yp_all,r2,mae,rmse,mape,fi,Xd.index

    Xte_d,yte_d,yp_d,yp_all,r2,mae,rmse,mape,fi_d,xd_idx = run_regression(hash(str(df.shape)))

    df["mrr_predicted"] = 0.0
    df.loc[xd_idx,"mrr_predicted"] = yp_all

    # ── KPIs ────────────────────────────────────────────────────────────────
    kc=st.columns(4)
    for col,val,lbl in [(kc[0],f"{r2:.4f}","R² Score"),(kc[1],f"${mae:.2f}","MAE"),(kc[2],f"${rmse:.2f}","RMSE"),(kc[3],f"{mape:.1f}%","MAPE (subscribers)")]:
        col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>',unsafe_allow_html=True)

    st.markdown("---")
    tab1,tab2,tab3,tab4 = st.tabs(["🎯 Actual vs Predicted","💡 Feature Importance","📊 Tier Forecast","🌍 10k Extrapolation"])

    with tab1:
        fig,ax=plt.subplots(figsize=(10,6),facecolor=C_DARK)
        te_tiers=df.loc[yte_d.index,"preferred_subscription_tier"]
        for tier in TIER_ORDER:
            mask_t=te_tiers==tier
            if mask_t.sum()==0: continue
            ax.scatter(yte_d[mask_t],yp_d[mask_t],c=TIER_COLOR_MAP[tier],alpha=0.65,s=55,edgecolors="none",label=f"{tier} (n={mask_t.sum()})",zorder=3)
        lims=[min(yte_d.min(),yp_d.min())-5,max(yte_d.max(),yp_d.max())+5]
        ax.plot(lims,lims,"--",color=C_ACCENT,lw=2,alpha=0.7,label="Perfect prediction",zorder=2)
        ax.text(0.02,0.96,f"R²={r2:.4f}  MAE=${mae:.2f}  RMSE=${rmse:.2f}  MAPE={mape:.1f}%",
                transform=ax.transAxes,fontsize=10,color=C_TEXT,va="top",
                bbox=dict(boxstyle="round,pad=0.4",facecolor=C_CARD,edgecolor=C_BORDER,alpha=0.9))
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual MRR (USD)"); ax.set_ylabel("Predicted MRR (USD)")
        ax.set_title("Actual vs Predicted MRR · Coloured by Tier",color=C_TEXT,fontsize=13,fontweight="bold")
        ax.legend(fontsize=9.5,facecolor=C_PANEL,edgecolor=C_BORDER,labelcolor=C_TEXT,loc="upper left")
        ax.grid(True,alpha=0.12); fig.tight_layout(); fig_to_st(fig)

    with tab2:
        fig,ax=plt.subplots(figsize=(11,7),facecolor=C_DARK)
        fi_colors=[]
        for feat in fi_d["feature"]:
            if feat in ["wtp_monthly_usd","preferred_subscription_tier","monthly_spend_on_ai_usd"]: fi_colors.append(C_AMBER)
            elif feat in ["trust_concern_frequency","difficulty_finding_tools"]: fi_colors.append(C_ROSE)
            elif feat in ["subscription_fatigue","bundle_interest","preferred_billing_cadence"]: fi_colors.append(C_VIOLET)
            elif feat.startswith("uses_"): fi_colors.append(C_TEAL)
            else: fi_colors.append(C_ACCENT)
        ax.barh(range(len(fi_d)),fi_d["importance"],color=fi_colors,alpha=0.87,height=0.72)
        ax.set_yticks(range(len(fi_d))); ax.set_yticklabels(fi_d["label"],fontsize=10)
        ax.invert_yaxis()
        for i,v in enumerate(fi_d["importance"]): ax.text(v+0.001,i,f"{v:.4f}",va="center",fontsize=8.5,color=C_MUTED)
        patches=[mpatches.Patch(color=c,label=l) for c,l in [(C_AMBER,"Monetization"),(C_ROSE,"Trust & Discovery"),(C_VIOLET,"Marketplace Fit"),(C_TEAL,"Tool Usage"),(C_ACCENT,"Demographics")]]
        ax.legend(handles=patches,loc="lower right",fontsize=9,facecolor=C_CARD,edgecolor=C_BORDER,labelcolor=C_TEXT)
        ax.set_xlabel("GB Feature Importance"); ax.set_title("Top 20 MRR Predictors · Gradient Boosting",color=C_TEXT,fontsize=13,fontweight="bold"); ax.grid(True,axis="x",alpha=0.15)
        fig.tight_layout(); fig_to_st(fig)
        for i in range(3):
            colors_map=["amber","teal",""]
            st.markdown(f'<div class="insight-box {colors_map[i] if i<2 else ""}"><b>#{i+1}: {fi_d.iloc[i]["label"]}</b> — {"WTP is the ceiling-setter. Surface value before price to avoid anchoring low." if i==0 else "Tier is a direct structural lever — one tier up ≈ 3× MRR." if i==1 else "High AI spend = budget maturity. Reframe as consolidation, not new cost."}</div>',unsafe_allow_html=True)

    with tab3:
        tier_rows=[]
        for tier in TIER_ORDER:
            mask=df["preferred_subscription_tier"]==tier
            tier_rows.append({"Tier":tier,"Users":int(mask.sum()),
                "Sub Rate":f"{df.loc[mask,'will_subscribe'].mean():.1%}",
                "Avg Pred MRR":f"${df.loc[mask,'mrr_predicted'].mean():.2f}",
                "Avg Actual MRR":f"${df.loc[mask,'estimated_mrr_usd'].mean():.2f}",
                "Proj MRR (all)":f"${df.loc[mask,'mrr_predicted'].mean()*mask.sum():,.0f}",
                "Proj MRR (real)":f"${df.loc[mask,'mrr_predicted'].mean()*mask.sum()*df.loc[mask,'will_subscribe'].mean():,.0f}",
            })
        tier_tbl=pd.DataFrame(tier_rows)
        st.dataframe(tier_tbl,use_container_width=True,height=200)

        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5),facecolor=C_DARK)
        avg_pred=[df.loc[df["preferred_subscription_tier"]==t,"mrr_predicted"].mean() for t in TIER_ORDER]
        avg_act =[df.loc[df["preferred_subscription_tier"]==t,"estimated_mrr_usd"].mean() for t in TIER_ORDER]
        x_=np.arange(4); w_=0.35
        b1=ax1.bar(x_-w_/2,avg_pred,w_,color=[TIER_COLOR_MAP[t] for t in TIER_ORDER],alpha=0.85,label="Predicted")
        b2=ax1.bar(x_+w_/2,avg_act, w_,color=[TIER_COLOR_MAP[t] for t in TIER_ORDER],alpha=0.40,label="Actual",hatch="//")
        for bar in list(b1)+list(b2): ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,f"${bar.get_height():.0f}",ha="center",va="bottom",fontsize=9,color=C_TEXT)
        ax1.set_xticks(x_); ax1.set_xticklabels(TIER_ORDER); ax1.set_ylabel("MRR / User (USD)")
        ax1.set_title("Predicted vs Actual MRR by Tier",color=C_TEXT,fontsize=11,fontweight="bold")
        ax1.legend(fontsize=9,facecolor=C_PANEL,edgecolor=C_BORDER,labelcolor=C_TEXT); ax1.grid(True,axis="y",alpha=0.15)
        real_proj=[df.loc[df["preferred_subscription_tier"]==t,"mrr_predicted"].mean()*
                   (df["preferred_subscription_tier"]==t).sum()*
                   df.loc[df["preferred_subscription_tier"]==t,"will_subscribe"].mean() for t in TIER_ORDER]
        bars_r=ax2.bar(TIER_ORDER,real_proj,color=[TIER_COLOR_MAP[t] for t in TIER_ORDER],alpha=0.85,edgecolor=C_BORDER,width=0.6)
        for bar,v in zip(bars_r,real_proj): ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+100,f"${v:,.0f}",ha="center",va="bottom",fontsize=9.5,color=C_TEXT,fontweight="bold")
        ax2.set_ylabel("Projected MRR (USD)"); ax2.set_title("Realistic Projected MRR (1k sample)",color=C_TEXT,fontsize=11,fontweight="bold"); ax2.grid(True,axis="y",alpha=0.15)
        fig.tight_layout(); fig_to_st(fig)

    with tab4:
        SCALE=10
        total_real=sum([df.loc[df["preferred_subscription_tier"]==t,"mrr_predicted"].mean()*
                        (df["preferred_subscription_tier"]==t).sum()*
                        df.loc[df["preferred_subscription_tier"]==t,"will_subscribe"].mean() for t in TIER_ORDER])
        mrr_10k=total_real*SCALE; arr_10k=mrr_10k*12

        col1,col2,col3=st.columns(3)
        col1.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#4ADE80">${mrr_10k:,.0f}</div><div class="metric-lbl">Projected MRR @ 10k users</div></div>',unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#4ADE80">${arr_10k/1e6:.2f}M</div><div class="metric-lbl">Projected ARR @ 10k users</div></div>',unsafe_allow_html=True)
        plat_avg=df.loc[df["preferred_subscription_tier"]=="Platinum","mrr_predicted"].mean()
        weighted_avg=total_real/1000
        plat_sens=mrr_10k*(1+0.08*(plat_avg/weighted_avg-1)) if weighted_avg>0 else mrr_10k
        col3.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#FCD34D">${plat_sens/1e6:.2f}M</div><div class="metric-lbl">ARR if Platinum +8pp share</div></div>',unsafe_allow_html=True)

        fig,ax=plt.subplots(figsize=(9,5),facecolor=C_DARK)
        wf_vals=[df.loc[df["preferred_subscription_tier"]==t,"mrr_predicted"].mean()*
                 (df["preferred_subscription_tier"]==t).sum()*
                 df.loc[df["preferred_subscription_tier"]==t,"will_subscribe"].mean()*SCALE for t in TIER_ORDER]
        bars_wf=ax.bar(TIER_ORDER,wf_vals,color=[TIER_COLOR_MAP[t] for t in TIER_ORDER],alpha=0.85,edgecolor=C_BORDER,width=0.6)
        for bar,v in zip(bars_wf,wf_vals): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+500,f"${v:,.0f}",ha="center",va="bottom",fontsize=10,color=C_TEXT,fontweight="bold")
        ax.axhline(mrr_10k,color=C_ACCENT,lw=2,ls="--",alpha=0.8,label=f"Total MRR = ${mrr_10k:,.0f}")
        ax.text(3.42,mrr_10k+500,f"Total\n${mrr_10k:,.0f}/mo\nARR ${arr_10k/1e6:.2f}M",ha="right",va="bottom",fontsize=9,color=C_ACCENT,fontweight="bold")
        ax.set_ylabel("Projected MRR (USD)"); ax.set_title("10,000-User MRR Projection by Tier",color=C_TEXT,fontsize=13,fontweight="bold")
        ax.legend(fontsize=9,facecolor=C_PANEL,edgecolor=C_BORDER,labelcolor=C_TEXT); ax.grid(True,axis="y",alpha=0.15)
        fig.tight_layout(); fig_to_st(fig)

        st.markdown("---")
        st.markdown("**Tier breakdown at 10,000 users:**")
        bdown=[]
        for t,v in zip(TIER_ORDER,wf_vals):
            bdown.append({"Tier":t,"MRR/month":f"${v:,.0f}","ARR":f"${v*12:,.0f}"})
        st.dataframe(pd.DataFrame(bdown),use_container_width=True,height=180)
        st.markdown(f'<div class="insight-box amber"><b>Revenue mix insight:</b> Platinum (17% of users) generates {wf_vals[3]/mrr_10k:.0%} of total MRR at 10k scale. Shifting Platinum share from 17% → 25% adds ~${(plat_sens-mrr_10k)*12/1e3:.0f}k to ARR with zero new users.</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="insight-box teal"><b>Upsell > Acquisition:</b> Converting 100 Silver users to Gold adds ~${100*(df.loc[df["preferred_subscription_tier"]=="Gold","mrr_predicted"].mean()-df.loc[df["preferred_subscription_tier"]=="Silver","mrr_predicted"].mean()):,.0f}/mo — equivalent to acquiring ~{int(100*(df.loc[df["preferred_subscription_tier"]=="Gold","mrr_predicted"].mean()-df.loc[df["preferred_subscription_tier"]=="Silver","mrr_predicted"].mean())/df.loc[df["preferred_subscription_tier"]=="Silver","mrr_predicted"].mean()):,} new Silver users.</div>',unsafe_allow_html=True)
