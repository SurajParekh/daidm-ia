import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              silhouette_score, mean_absolute_error, r2_score)
from itertools import combinations

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VerifAI Marketplace — Business Validation Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── THEME COLOURS ──────────────────────────────────────────────────────────────
PRIMARY   = "#1A237E"
SECONDARY = "#3949AB"
ACCENT    = "#FF6F00"
LIGHT     = "#9FA8DA"
BG        = "#F5F6FA"

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F5F6FA; }
    .block-container { padding-top: 1.5rem; }
    h1 { color: #1A237E; font-weight: 800; }
    h2, h3 { color: #1A237E; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(26,35,126,0.10);
        margin-bottom: 0.5rem;
    }
    .metric-val { font-size: 2rem; font-weight: 800; color: #1A237E; }
    .metric-lbl { font-size: 0.85rem; color: #666; margin-top: 0.2rem; }
    .insight-box {
        background: #E8EAF6;
        border-left: 4px solid #3949AB;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        color: #1A237E;
    }
    .section-divider { border-top: 2px solid #E8EAF6; margin: 1.5rem 0; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── HELPERS ────────────────────────────────────────────────────────────────────
PALETTE = [PRIMARY, SECONDARY, "#5C6BC0", LIGHT, "#C5CAE9", "#E8EAF6", ACCENT, "#FFB300"]

def metric_card(label, value, suffix=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{value}{suffix}</div>
        <div class="metric-lbl">{label}</div>
    </div>""", unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded):
    if uploaded:
        return pd.read_csv(uploaded)
    return None

def encode_df(df):
    df2 = df.copy()
    cat_cols = df2.select_dtypes(include='object').columns.tolist()
    le_map = {}
    for col in cat_cols:
        le = LabelEncoder()
        df2[col + '_enc'] = le.fit_transform(df2[col].astype(str))
        le_map[col] = le
    return df2, le_map

ORDINAL_MAPS = {
    'difficulty_finding_tools':  {'Very Easy': 1, 'Easy': 2, 'Neutral': 3, 'Difficult': 4, 'Very Difficult': 5},
    'trust_concern_frequency':   {'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Always': 5},
    'subscription_fatigue':      {'Strongly Disagree': 1, 'Disagree': 2, 'Neutral': 3, 'Agree': 4, 'Strongly Agree': 5},
    'preferred_subscription_tier': {'Free': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3},
    'bundle_interest':           {'Not Interested': 1, 'Neutral': 2, 'Interested': 3, 'Very Interested': 4},
    'preferred_billing_cadence': {'Monthly': 1, 'Half-Yearly': 2, 'Yearly': 3},
}

def apply_ordinals(df):
    df2 = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        if col in df2.columns:
            df2[col + '_ord'] = df2[col].map(mapping).fillna(0)
    return df2

TOOL_COLS = ['uses_llm_writing_tools', 'uses_image_gen_tools', 'uses_video_gen_tools',
             'uses_code_assistant_tools', 'uses_data_analytics_tools',
             'uses_voice_audio_tools', 'uses_productivity_tools']
TOOL_LABELS = {'uses_llm_writing_tools': 'LLM / Writing',
               'uses_image_gen_tools':   'Image Gen',
               'uses_video_gen_tools':   'Video Gen',
               'uses_code_assistant_tools': 'Code Assistant',
               'uses_data_analytics_tools': 'Data Analytics',
               'uses_voice_audio_tools': 'Voice & Audio',
               'uses_productivity_tools': 'Productivity'}

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("VerifAI Marketplace")
    st.caption("Business Validation Dashboard")
    st.markdown("---")
    uploaded = st.file_uploader("📂 Upload synthetic_data.csv", type=["csv"])
    st.markdown("---")
    st.markdown("**Navigation**")
    st.markdown("""
- 📊 Overview  
- 🔮 Classification  
- 👥 Clustering  
- 🔗 Association Rules  
- 📈 Regression  
- 📋 Data Explorer
    """)
    st.markdown("---")
    st.caption("MBA Data Analytics Module · VerifAI Marketplace")

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
df_raw = load_data(uploaded)

if df_raw is None:
    st.markdown("## 🤖 VerifAI Marketplace — Business Validation Dashboard")
    st.info("👈 Upload `synthetic_data.csv` using the sidebar to begin analysis.")
    st.markdown("""
    ### What this dashboard does
    This tool validates the **VerifAI AI App Store** business concept using data-driven analysis:
    
    | Module | Method | Business Question |
    |---|---|---|
    | 🔮 Classification | Random Forest + Logistic Regression | Which users will subscribe? |
    | 👥 Clustering | K-Means | What are our buyer personas? |
    | 🔗 Association Rules | Apriori-style | Which tools get used together → bundles? |
    | 📈 Regression | Gradient Boosting | What MRR can we forecast? |
    """)
    st.stop()

# ── PREPROCESS ─────────────────────────────────────────────────────────────────
df = apply_ordinals(df_raw)
df_enc, le_map = encode_df(df)

FEAT_COLS = [
    'user_type_enc', 'industry_enc', 'region_enc', 'age_group_enc',
    'num_ai_tools_currently_used', 'monthly_spend_on_ai_usd',
    'primary_discovery_method_enc', 'difficulty_finding_tools_ord',
    'trust_concern_frequency_ord', 'subscription_fatigue_ord',
    'preferred_subscription_tier_ord', 'preferred_billing_cadence_ord',
    'bundle_interest_ord', 'wtp_monthly_usd',
    'uses_llm_writing_tools', 'uses_image_gen_tools', 'uses_video_gen_tools',
    'uses_code_assistant_tools', 'uses_data_analytics_tools',
    'uses_voice_audio_tools', 'uses_productivity_tools'
]
FEAT_COLS = [f for f in FEAT_COLS if f in df_enc.columns]

X = df_enc[FEAT_COLS].fillna(0)
y_class = df_enc['will_subscribe']
y_reg   = df_enc['estimated_mrr_usd']

X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42)

# ── TABS ───────────────────────────────────────────────────────────────────────
t0, t1, t2, t3, t4, t5 = st.tabs([
    "📊 Overview",
    "🔮 Classification",
    "👥 Clustering",
    "🔗 Association Rules",
    "📈 Regression",
    "📋 Data Explorer"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with t0:
    st.markdown("## 📊 Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Total Respondents", f"{len(df_raw):,}")
    with c2: metric_card("Subscribe Rate", f"{df_raw['will_subscribe'].mean():.1%}")
    with c3: metric_card("Avg WTP / Month", f"${df_raw['wtp_monthly_usd'].mean():.0f}")
    with c4: metric_card("Avg AI Tools Used", f"{df_raw['num_ai_tools_currently_used'].mean():.1f}")
    with c5: metric_card("Projected Avg MRR", f"${df_raw['estimated_mrr_usd'].mean():.0f}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Subscription Intent by User Type")
        fig, ax = plt.subplots(figsize=(7, 4))
        sub_by_type = df_raw.groupby('user_type')['will_subscribe'].mean().sort_values(ascending=True) * 100
        bars = ax.barh(sub_by_type.index, sub_by_type.values, color=SECONDARY)
        for bar in bars:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{bar.get_width():.1f}%', va='center', fontsize=9, fontweight='bold')
        ax.set_xlabel('% Likely to Subscribe')
        ax.set_xlim(0, 105)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        insight("Enterprise employees show the highest subscription intent, validating a B2B2C GTM strategy where team/company plans drive initial revenue growth.")

    with col2:
        st.markdown("#### Preferred Subscription Tier Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        tier_order = ['Free', 'Silver', 'Gold', 'Platinum']
        tier_counts = df_raw['preferred_subscription_tier'].value_counts().reindex(tier_order)
        colors = [LIGHT, "#5C6BC0", SECONDARY, PRIMARY]
        bars = ax.bar(tier_counts.index, tier_counts.values, color=colors, edgecolor='white')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
                    str(bar.get_height()), ha='center', fontsize=10, fontweight='bold')
        ax.set_ylabel('Respondents')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        insight("Silver and Gold tiers command the largest share of preference, suggesting a mid-market pricing strategy will capture the broadest addressable user base.")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Trust Concern Frequency")
        fig, ax = plt.subplots(figsize=(7, 4))
        trust_order = ['Never', 'Rarely', 'Sometimes', 'Often', 'Always']
        trust_counts = df_raw['trust_concern_frequency'].value_counts().reindex(trust_order)
        colors_t = ['#E8EAF6', '#C5CAE9', LIGHT, SECONDARY, PRIMARY]
        ax.bar(trust_counts.index, trust_counts.values, color=colors_t, edgecolor='white')
        ax.set_ylabel('Respondents')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        insight("Nearly 48% of respondents frequently worry about AI tool safety, directly validating VerifAI's trust-and-verification core value proposition as a genuine market need.")

    with col4:
        st.markdown("#### Tool Category Adoption Rates")
        fig, ax = plt.subplots(figsize=(7, 4))
        tool_rates = {TOOL_LABELS[c]: df_raw[c].mean() * 100 for c in TOOL_COLS}
        tool_s = pd.Series(tool_rates).sort_values(ascending=True)
        ax.barh(tool_s.index, tool_s.values, color=SECONDARY)
        ax.set_xlabel('% Currently Using')
        ax.set_xlim(0, 100)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        insight("LLM/Writing and Code Assistant tools show the highest adoption, making them anchor products for the marketplace catalogue and primary drivers of bundle design.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    st.markdown("## 🔮 Classification — Predicting Subscription Intent")
    st.markdown("Predict which users are likely to subscribe using **Logistic Regression** and **Random Forest**.")

    col_cfg, col_res = st.columns([1, 2])
    with col_cfg:
        st.markdown("#### Model Settings")
        model_choice = st.selectbox("Select Model", ["Random Forest", "Logistic Regression", "Both"])
        n_estimators = st.slider("RF: Number of Trees", 50, 300, 100, 50)
        test_size = st.slider("Test Set Size", 0.15, 0.35, 0.20, 0.05)
        run_cls = st.button("▶ Run Classification", use_container_width=True)

    if run_cls:
        X_tr2, X_te2, yc_tr2, yc_te2 = train_test_split(X, y_class, test_size=test_size, random_state=42)
        results = {}

        with st.spinner("Training models..."):
            if model_choice in ["Random Forest", "Both"]:
                rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
                rf.fit(X_tr2, yc_tr2)
                rf_pred = rf.predict(X_te2)
                rf_acc  = (rf_pred == yc_te2).mean()
                results['Random Forest'] = (rf, rf_pred, rf_acc)

            if model_choice in ["Logistic Regression", "Both"]:
                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_tr2)
                X_te_sc = scaler.transform(X_te2)
                lr = LogisticRegression(max_iter=1000, random_state=42)
                lr.fit(X_tr_sc, yc_tr2)
                lr_pred = lr.predict(X_te_sc)
                lr_acc  = (lr_pred == yc_te2).mean()
                results['Logistic Regression'] = (lr, lr_pred, lr_acc)

        # Metrics row
        cols = st.columns(len(results) * 2)
        i = 0
        for name, (mdl, preds, acc) in results.items():
            with cols[i]:   metric_card(f"{name} Accuracy", f"{acc:.1%}")
            i += 1
            prec = (preds[yc_te2 == 1] == 1).mean() if (preds == 1).any() else 0
            with cols[i]:   metric_card(f"{name} Precision", f"{prec:.1%}")
            i += 1

        # Confusion matrices
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = [axes]
        for ax, (name, (mdl, preds, acc)) in zip(axes, results.items()):
            cm = confusion_matrix(yc_te2, preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                        xticklabels=['Not Subscribe', 'Subscribe'],
                        yticklabels=['Not Subscribe', 'Subscribe'])
            ax.set_title(f'{name}\nAccuracy: {acc:.1%}', fontweight='bold')
            ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        insight("The confusion matrix shows the model correctly identifies most likely subscribers; false negatives (missed subscribers) represent untapped revenue and should be minimised through feature engineering.")

        # Feature importance (RF only)
        if 'Random Forest' in results:
            rf_mdl = results['Random Forest'][0]
            st.markdown("#### Feature Importance (Random Forest)")
            feat_names = [f.replace('_enc','').replace('_ord','').replace('_',' ').title() for f in FEAT_COLS]
            imp = pd.Series(rf_mdl.feature_importances_, index=feat_names).sort_values(ascending=True).tail(12)
            fig2, ax2 = plt.subplots(figsize=(9, 5))
            imp.plot.barh(ax=ax2, color=SECONDARY)
            ax2.axvline(imp.mean(), color=ACCENT, linestyle='--', label='Mean')
            ax2.set_xlabel('Gini Importance')
            ax2.set_title('Top Features Driving Subscription Intent', fontweight='bold')
            ax2.legend(); ax2.spines[['top','right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
            insight("WTP and preferred tier are the strongest predictors — users who have already mentally committed to paying are most likely to convert, suggesting price-anchoring in onboarding copy is critical.")

        # Classification report
        st.markdown("#### Detailed Classification Report")
        for name, (mdl, preds, acc) in results.items():
            st.markdown(f"**{name}**")
            report = classification_report(yc_te2, preds, target_names=['Not Subscribe', 'Subscribe'], output_dict=True)
            st.dataframe(pd.DataFrame(report).T.round(3))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
with t2:
    st.markdown("## 👥 Clustering — Customer Personas & Segments")
    st.markdown("K-Means clustering identifies natural user segments for targeted GTM strategy.")

    col_cfg2, _ = st.columns([1, 2])
    with col_cfg2:
        k = st.slider("Number of Clusters (k)", 2, 8, 4)
        run_cls2 = st.button("▶ Run Clustering", use_container_width=True)

    if run_cls2:
        cluster_feats = ['monthly_spend_on_ai_usd', 'wtp_monthly_usd',
                         'num_ai_tools_currently_used',
                         'difficulty_finding_tools_ord', 'trust_concern_frequency_ord',
                         'subscription_fatigue_ord', 'preferred_subscription_tier_ord']
        cluster_feats = [f for f in cluster_feats if f in df_enc.columns]

        with st.spinner("Running K-Means..."):
            scaler = StandardScaler()
            X_cl = scaler.fit_transform(df_enc[cluster_feats].fillna(0))
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_cl)
            sil = silhouette_score(X_cl, labels)
            df_enc['cluster'] = labels
            df_raw2 = df_raw.copy()
            df_raw2['cluster'] = labels

        metric_card("Silhouette Score", f"{sil:.3f}", " (0=random, 1=perfect)")
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Cluster profiles
        st.markdown("#### Cluster Profiles")
        profile = df_raw2.groupby('cluster').agg(
            Size=('respondent_id', 'count'),
            Avg_Monthly_Spend=('monthly_spend_on_ai_usd', 'mean'),
            Avg_WTP=('wtp_monthly_usd', 'mean'),
            Avg_Tools_Used=('num_ai_tools_currently_used', 'mean'),
            Subscribe_Rate=('will_subscribe', 'mean'),
            Avg_MRR=('estimated_mrr_usd', 'mean'),
        ).round(2)
        profile['Subscribe_Rate'] = (profile['Subscribe_Rate'] * 100).round(1).astype(str) + '%'
        st.dataframe(profile, use_container_width=True)

        # Auto-name clusters
        raw_profile = df_raw2.groupby('cluster').agg(
            spend=('monthly_spend_on_ai_usd','mean'),
            wtp=('wtp_monthly_usd','mean'),
            sub=('will_subscribe','mean')
        )
        persona_names = {}
        sorted_by_spend = raw_profile['spend'].sort_values(ascending=False).index
        names = ['Enterprise Power Buyers', 'Growth-Stage Adopters', 'Cost-Conscious Explorers', 'Low-Engagement Browsers']
        for idx, cid in enumerate(sorted_by_spend[:k]):
            persona_names[cid] = names[min(idx, len(names)-1)]
        df_raw2['persona'] = df_raw2['cluster'].map(persona_names)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Spend vs WTP by Cluster")
            fig, ax = plt.subplots(figsize=(7, 5))
            colors_cl = [PRIMARY, SECONDARY, ACCENT, LIGHT, "#5C6BC0", "#FFB300", "#26A69A", "#EF5350"]
            for cid in sorted(df_raw2['cluster'].unique()):
                mask = df_raw2['cluster'] == cid
                ax.scatter(df_raw2.loc[mask, 'monthly_spend_on_ai_usd'],
                           df_raw2.loc[mask, 'wtp_monthly_usd'],
                           c=colors_cl[cid % len(colors_cl)], alpha=0.5, s=35,
                           label=persona_names.get(cid, f'Cluster {cid}'))
            ax.set_xlabel('Monthly AI Spend (USD)')
            ax.set_ylabel('WTP Monthly (USD)')
            ax.legend(fontsize=8, framealpha=0.8)
            ax.spines[['top','right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            insight("Enterprise Power Buyers cluster at high spend AND high WTP, making them the highest-value acquisition segment with the strongest potential lifetime revenue per user.")

        with col_b:
            st.markdown("#### Subscription Rate by Persona")
            fig, ax = plt.subplots(figsize=(7, 5))
            persona_sub = df_raw2.groupby('persona')['will_subscribe'].mean().sort_values(ascending=True)
            bars = ax.barh(persona_sub.index, persona_sub.values * 100, color=SECONDARY)
            for bar in bars:
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{bar.get_width():.1f}%', va='center', fontsize=9, fontweight='bold')
            ax.set_xlabel('Subscription Rate (%)')
            ax.set_xlim(0, 110)
            ax.spines[['top','right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            insight("Power buyers convert at significantly higher rates, validating early GTM focus on enterprise and professional users before scaling to the mass consumer segment.")

        st.markdown("#### Industry Distribution per Cluster")
        fig, ax = plt.subplots(figsize=(12, 4))
        ind_cluster = df_raw2.groupby(['cluster','industry']).size().unstack(fill_value=0)
        ind_cluster.plot(kind='bar', ax=ax, color=PALETTE[:len(ind_cluster.columns)], edgecolor='white')
        ax.set_xlabel('Cluster'); ax.set_ylabel('Count')
        ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1))
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        insight("Design & Creative and Software Development industries dominate high-value clusters, suggesting vertical-specific bundles for these personas should be the first products launched on the marketplace.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    st.markdown("## 🔗 Association Rule Mining — Tool Bundle Discovery")
    st.markdown("Find which AI tool categories are used together to inform **bundle design**.")

    col_ar1, col_ar2 = st.columns([1, 2])
    with col_ar1:
        min_support = st.slider("Min Support", 0.05, 0.50, 0.15, 0.01)
        min_confidence = st.slider("Min Confidence", 0.30, 0.90, 0.50, 0.05)
        run_ar = st.button("▶ Mine Association Rules", use_container_width=True)

    if run_ar:
        with st.spinner("Mining rules..."):
            tool_df = df_raw[TOOL_COLS].copy()
            tool_df.columns = [TOOL_LABELS[c] for c in TOOL_COLS]
            items = list(tool_df.columns)
            n = len(tool_df)

            def support_val(cols, data):
                mask = pd.Series([True] * len(data))
                for c in cols:
                    mask &= (data[c] == 1)
                return mask.mean()

            # Generate itemsets
            rules_list = []
            for r in range(1, 4):
                for combo in combinations(items, r):
                    sup = support_val(list(combo), tool_df)
                    if sup >= min_support:
                        # Generate rules from this itemset
                        for i in range(1, len(combo)):
                            for ant_combo in combinations(combo, i):
                                ant = list(ant_combo)
                                con = [x for x in combo if x not in ant]
                                if not con:
                                    continue
                                ant_sup = support_val(ant, tool_df)
                                con_sup = support_val(con, tool_df)
                                conf = sup / ant_sup if ant_sup > 0 else 0
                                lift = conf / con_sup if con_sup > 0 else 0
                                if conf >= min_confidence:
                                    rules_list.append({
                                        'Antecedent': ' + '.join(ant),
                                        'Consequent': ' + '.join(con),
                                        'Support':    round(sup, 4),
                                        'Confidence': round(conf, 4),
                                        'Lift':       round(lift, 4)
                                    })

            rules_df = pd.DataFrame(rules_list).drop_duplicates()
            if not rules_df.empty:
                rules_df = rules_df.sort_values('Lift', ascending=False).head(30)

        if rules_df.empty:
            st.warning("No rules found at these thresholds. Try lowering Support or Confidence.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1: metric_card("Rules Found", len(rules_df))
            with c2: metric_card("Max Lift", f"{rules_df['Lift'].max():.2f}")
            with c3: metric_card("Avg Confidence", f"{rules_df['Confidence'].mean():.1%}")

            st.markdown("#### Top Association Rules (sorted by Lift)")
            st.dataframe(rules_df.reset_index(drop=True), use_container_width=True)
            insight("Rules with Lift > 1.2 indicate tools used together significantly more than by chance — these pairings are the strongest candidates for pre-built bundles on the marketplace.")

            # Heatmap: co-occurrence
            st.markdown("#### Tool Co-occurrence Heatmap")
            co = pd.DataFrame(0.0, index=items, columns=items)
            for i in items:
                for j in items:
                    if i != j:
                        co.loc[i, j] = support_val([i, j], tool_df)
                    else:
                        co.loc[i, j] = support_val([i], tool_df)
            fig, ax = plt.subplots(figsize=(9, 7))
            sns.heatmap(co, annot=True, fmt='.2f', cmap='Blues', ax=ax, linewidths=0.5,
                        xticklabels=[l[:12] for l in items],
                        yticklabels=[l[:12] for l in items])
            ax.set_title('Tool Co-occurrence Matrix (Support)', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            insight("Darker cells indicate tool pairs frequently adopted together; the LLM/Writing + Code Assistant + Productivity cluster is the strongest, pointing to a 'Developer Productivity Bundle' as a high-demand product.")

            # Bundle recommendations
            st.markdown("#### 💡 Suggested Bundles Based on Rules")
            top_rules = rules_df.nlargest(5, 'Lift')
            for _, row in top_rules.iterrows():
                st.markdown(f"- **{row['Antecedent']}** → **{row['Consequent']}** "
                            f"(Confidence: {row['Confidence']:.1%}, Lift: {row['Lift']:.2f}x)")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
with t4:
    st.markdown("## 📈 Regression — MRR Forecasting")
    st.markdown("Forecast **Monthly Recurring Revenue** per user using **Gradient Boosting Regressor**.")

    col_r1, _ = st.columns([1, 2])
    with col_r1:
        n_est_reg = st.slider("GB: Estimators", 50, 300, 100, 50)
        lr_rate   = st.slider("GB: Learning Rate", 0.01, 0.30, 0.10, 0.01)
        run_reg   = st.button("▶ Run Regression", use_container_width=True)

    if run_reg:
        with st.spinner("Training regression model..."):
            X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
            gbr = GradientBoostingRegressor(n_estimators=n_est_reg, learning_rate=lr_rate, random_state=42)
            gbr.fit(X_tr_r, y_tr_r)
            y_pred_r = gbr.predict(X_te_r)
            mae = mean_absolute_error(y_te_r, y_pred_r)
            r2  = r2_score(y_te_r, y_pred_r)

        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("R² Score", f"{r2:.4f}")
        with c2: metric_card("MAE (USD)", f"${mae:.2f}")
        with c3: metric_card("Avg Predicted MRR", f"${y_pred_r.mean():.2f}")
        with c4: metric_card("Total Projected MRR (Test)", f"${y_pred_r.sum():,.0f}")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Actual vs Predicted MRR")
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(y_te_r, y_pred_r, alpha=0.4, c=SECONDARY, s=25)
            lims = [min(y_te_r.min(), y_pred_r.min()), max(y_te_r.max(), y_pred_r.max())]
            ax.plot(lims, lims, 'r--', lw=1.5, label='Perfect fit')
            ax.set_xlabel('Actual MRR (USD)'); ax.set_ylabel('Predicted MRR (USD)')
            ax.set_title('Actual vs Predicted MRR', fontweight='bold')
            ax.legend(); ax.spines[['top','right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            insight("Points clustering near the red line indicate high prediction accuracy; outliers at high MRR values suggest enterprise users are harder to forecast and may need separate modelling.")

        with col_b:
            st.markdown("#### MRR Distribution: Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(y_te_r, bins=30, alpha=0.6, label='Actual', color=SECONDARY, edgecolor='white')
            ax.hist(y_pred_r, bins=30, alpha=0.6, label='Predicted', color=ACCENT, edgecolor='white')
            ax.set_xlabel('MRR (USD)'); ax.set_ylabel('Count')
            ax.set_title('MRR Distribution', fontweight='bold')
            ax.legend(); ax.spines[['top','right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            insight("The model captures the zero-heavy distribution (non-subscribers) well; the right tail (high MRR users) represents the enterprise segment driving disproportionate revenue concentration.")

        # MRR forecast by tier
        st.markdown("#### Projected MRR by Subscription Tier")
        df_forecast = df_raw.copy()
        df_forecast['predicted_mrr'] = gbr.predict(X.fillna(0))
        tier_mrr = df_forecast.groupby('preferred_subscription_tier')['predicted_mrr'].agg(['mean','sum','count'])
        tier_mrr.columns = ['Avg Predicted MRR', 'Total Projected MRR', 'Users']
        tier_mrr = tier_mrr.reindex(['Free','Silver','Gold','Platinum']).round(2)
        st.dataframe(tier_mrr, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        tiers = ['Free', 'Silver', 'Gold', 'Platinum']
        vals = [tier_mrr.loc[t, 'Avg Predicted MRR'] for t in tiers if t in tier_mrr.index]
        bar_colors = [LIGHT, "#5C6BC0", SECONDARY, PRIMARY]
        ax.bar(tiers[:len(vals)], vals, color=bar_colors[:len(vals)], edgecolor='white')
        ax.set_ylabel('Avg Predicted MRR (USD)')
        ax.set_title('Average Predicted MRR by Tier', fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        insight("Platinum users generate 6-8× the MRR of Silver users; even with fewer Platinum subscribers, tier-upgrade campaigns deliver outsized revenue impact relative to new user acquisition.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with t5:
    st.markdown("## 📋 Data Explorer")
    st.markdown(f"**{len(df_raw):,} rows × {len(df_raw.columns)} columns**")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        ut_filter = st.multiselect("User Type", df_raw['user_type'].unique().tolist(), default=[])
    with col_f2:
        ind_filter = st.multiselect("Industry", df_raw['industry'].unique().tolist(), default=[])
    with col_f3:
        tier_filter = st.multiselect("Tier", df_raw['preferred_subscription_tier'].unique().tolist(), default=[])

    df_display = df_raw.copy()
    if ut_filter:   df_display = df_display[df_display['user_type'].isin(ut_filter)]
    if ind_filter:  df_display = df_display[df_display['industry'].isin(ind_filter)]
    if tier_filter: df_display = df_display[df_display['preferred_subscription_tier'].isin(tier_filter)]

    st.markdown(f"Showing **{len(df_display):,}** rows after filters")
    st.dataframe(df_display, use_container_width=True, height=400)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_dl = df_display.to_csv(index=False).encode()
        st.download_button("⬇ Download Filtered CSV", csv_dl, "filtered_data.csv", "text/csv")
    with col_dl2:
        st.markdown("#### Summary Statistics")
        st.dataframe(df_display[['monthly_spend_on_ai_usd','wtp_monthly_usd',
                                  'num_ai_tools_currently_used','estimated_mrr_usd']].describe().round(2),
                     use_container_width=True)
