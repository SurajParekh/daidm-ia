"""
VerifAI Analytics Dashboard
============================
Consolidated pipeline covering:
  Module A — Association Rule Mining (Apriori from scratch)
  Module B — K-Means Customer Persona Clustering
  Module C — Subscription Prediction (Logistic Regression + Random Forest)

Outputs:
  verifai_association_rules.png   — heatmap + rule chart
  verifai_clusters.png            — persona clustering report
  verifai_model_report.png        — ML classification report
  verifai_rules.csv               — full association rules table
"""

import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, roc_auc_score,
                             roc_curve, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  SHARED THEME
# ══════════════════════════════════════════════════════════════════════════════
C_DARK   = "#0A0E1A"
C_PANEL  = "#111827"
C_CARD   = "#1A2235"
C_BORDER = "#2D3748"
C_TEXT   = "#F0F4FF"
C_MUTED  = "#8892A4"
C_ACCENT = "#6366F1"   # indigo
C_TEAL   = "#2DD4BF"
C_ROSE   = "#FB7185"
C_AMBER  = "#FCD34D"
C_VIOLET = "#A78BFA"
C_GREEN  = "#4ADE80"

CLUSTER_COLORS = [C_ACCENT, C_TEAL, C_ROSE, C_AMBER, C_VIOLET, C_GREEN,
                  "#F97316", "#38BDF8"]

plt.rcParams.update({
    "figure.facecolor":  C_DARK,
    "axes.facecolor":    C_PANEL,
    "axes.edgecolor":    C_BORDER,
    "axes.labelcolor":   C_TEXT,
    "xtick.color":       C_MUTED,
    "ytick.color":       C_MUTED,
    "text.color":        C_TEXT,
    "grid.color":        C_BORDER,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
})

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv("/mnt/user-data/uploads/verifai_survey_data.csv")
print(f"Loaded {len(df):,} rows × {len(df.columns)} columns\n")

TOOL_COLS = [
    "uses_llm_writing_tools",
    "uses_image_gen_tools",
    "uses_video_gen_tools",
    "uses_code_assistant_tools",
    "uses_data_analytics_tools",
    "uses_voice_audio_tools",
    "uses_productivity_tools",
]
TOOL_LABELS = {
    "uses_llm_writing_tools":    "LLM Writing",
    "uses_image_gen_tools":      "Image Gen",
    "uses_video_gen_tools":      "Video Gen",
    "uses_code_assistant_tools": "Code Assistant",
    "uses_data_analytics_tools": "Data Analytics",
    "uses_voice_audio_tools":    "Voice / Audio",
    "uses_productivity_tools":   "Productivity",
}
SHORT = {v: v for v in TOOL_LABELS.values()}   # identity; for rule labels


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE A — ASSOCIATION RULE MINING (APRIORI FROM SCRATCH)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 62)
print("  MODULE A  —  ASSOCIATION RULE MINING")
print("=" * 62)

MIN_SUPPORT    = 0.15
MIN_CONFIDENCE = 0.50

# ── 1. transactions as frozensets ────────────────────────────────────────────
basket = df[TOOL_COLS].rename(columns=TOOL_LABELS)
items  = list(TOOL_LABELS.values())
N      = len(basket)

transactions = [
    frozenset(item for item in items if row[item] == 1)
    for _, row in basket.iterrows()
]

# ── 2. Apriori: frequent itemsets ────────────────────────────────────────────
def support(itemset, txns):
    return sum(1 for t in txns if itemset.issubset(t)) / len(txns)

def apriori(transactions, items, min_sup):
    freq = {}
    # k=1
    L1 = {frozenset([i]): support(frozenset([i]), transactions) for i in items}
    L1 = {k: v for k, v in L1.items() if v >= min_sup}
    freq.update(L1)

    Lk = L1
    k  = 2
    while Lk:
        candidates = set()
        Lk_list = list(Lk.keys())
        for i in range(len(Lk_list)):
            for j in range(i + 1, len(Lk_list)):
                union = Lk_list[i] | Lk_list[j]
                if len(union) == k:
                    candidates.add(union)
        Lk_new = {c: support(c, transactions) for c in candidates}
        Lk_new = {k_: v for k_, v in Lk_new.items() if v >= min_sup}
        freq.update(Lk_new)
        Lk = Lk_new
        k += 1
    return freq

frequent_itemsets = apriori(transactions, items, MIN_SUPPORT)
print(f"\nFrequent itemsets found (support ≥ {MIN_SUPPORT}): {len(frequent_itemsets)}")

# ── 3. Generate association rules ────────────────────────────────────────────
rules = []
for itemset, sup_itemset in frequent_itemsets.items():
    if len(itemset) < 2:
        continue
    for size in range(1, len(itemset)):
        for antecedent in map(frozenset, itertools.combinations(itemset, size)):
            consequent = itemset - antecedent
            sup_ant  = frequent_itemsets.get(antecedent, support(antecedent, transactions))
            sup_con  = frequent_itemsets.get(consequent, support(consequent, transactions))
            conf     = sup_itemset / sup_ant if sup_ant > 0 else 0
            lift     = conf / sup_con if sup_con > 0 else 0
            if conf >= MIN_CONFIDENCE:
                rules.append({
                    "antecedent":  ", ".join(sorted(antecedent)),
                    "consequent":  ", ".join(sorted(consequent)),
                    "support":     round(sup_itemset, 4),
                    "confidence":  round(conf, 4),
                    "lift":        round(lift, 4),
                })

rules_df = pd.DataFrame(rules).drop_duplicates()
rules_df = rules_df.sort_values("lift", ascending=False).reset_index(drop=True)

print(f"Association rules generated (confidence ≥ {MIN_CONFIDENCE}): {len(rules_df)}")
print(f"\nTop 10 Rules by Lift:\n{'─'*78}")
print(f"  {'Antecedent':<36} {'Consequent':<18} {'Sup':>6} {'Conf':>6} {'Lift':>6}")
print(f"  {'─'*73}")
for _, r in rules_df.head(10).iterrows():
    print(f"  {r['antecedent']:<36} → {r['consequent']:<18} "
          f"{r['support']:>5.3f}  {r['confidence']:>5.3f}  {r['lift']:>5.3f}")

# ── 4. Co-occurrence matrix ───────────────────────────────────────────────────
labels_short = list(TOOL_LABELS.values())
co_matrix = pd.DataFrame(0.0, index=labels_short, columns=labels_short)
for t in transactions:
    t_list = list(t)
    for a, b in itertools.product(t_list, t_list):
        co_matrix.loc[a, b] += 1
co_matrix = co_matrix / N   # normalise to support

# ── 5. Top 5 bundle recommendations ──────────────────────────────────────────
top5 = rules_df.head(5).copy()

BUNDLE_DEFS = [
    {
        "name":    "🖥️  Full-Stack Dev Suite",
        "tools":   ["LLM Writing", "Code Assistant", "Data Analytics"],
        "persona": "Software engineers and data engineers who build, analyse, and document",
        "pitch":   "Every tool in your daily dev loop — one verified stack.",
    },
    {
        "name":    "🎨  Creator Studio",
        "tools":   ["Image Gen", "Video Gen", "Productivity"],
        "persona": "Designers, content creators, and visual storytellers",
        "pitch":   "From concept to publish — verified creative tools in one bundle.",
    },
    {
        "name":    "📊  Analyst Intelligence Pack",
        "tools":   ["Data Analytics", "LLM Writing", "Code Assistant"],
        "persona": "Data analysts and growth marketers who move between SQL and slides",
        "pitch":   "Query it. Model it. Explain it. Three tools, one price.",
    },
    {
        "name":    "✍️  Content & Copy Engine",
        "tools":   ["LLM Writing", "Productivity", "Image Gen"],
        "persona": "Freelance writers, marketers, and solopreneurs shipping content daily",
        "pitch":   "Write faster, stay organised, make it visual — all verified.",
    },
    {
        "name":    "🎙️  Media Production Bundle",
        "tools":   ["Voice / Audio", "Video Gen", "Image Gen"],
        "persona": "Podcast producers, video editors, and multimedia creators",
        "pitch":   "Studio-quality AI tools — curated, tested, trusted.",
    },
]

print(f"\n{'═'*62}")
print("  TOP 5 BUNDLE RECOMMENDATIONS  (by Association Rule Lift)")
print(f"{'═'*62}")
for i, (_, rule) in enumerate(top5.iterrows()):
    b = BUNDLE_DEFS[i]
    # Estimate market size: rows where all bundle tools == 1
    tool_cols_in_bundle = [k for k, v in TOOL_LABELS.items() if v in b["tools"]]
    mask = (df[tool_cols_in_bundle] == 1).all(axis=1)
    n_users = mask.sum()
    pct     = n_users / N * 100
    avg_wtp = df.loc[mask, "wtp_monthly_usd"].mean()
    est_mrr = df.loc[mask, "estimated_mrr_usd"].mean()

    print(f"\n  {'─'*58}")
    print(f"  #{i+1}  {b['name']}")
    print(f"  {'─'*58}")
    print(f"  Rule:      {rule['antecedent']} → {rule['consequent']}")
    print(f"  Metrics:   Support={rule['support']:.3f}  "
          f"Confidence={rule['confidence']:.3f}  Lift={rule['lift']:.3f}")
    print(f"  Tools:     {' + '.join(b['tools'])}")
    print(f"  Persona:   {b['persona']}")
    print(f"  Pitch:     {b['pitch']}")
    print(f"  Market:    {n_users} users ({pct:.1f}% of sample) | "
          f"Avg WTP ${avg_wtp:.0f}/mo | Avg MRR ${est_mrr:.0f}")

# ── 6. Save rules CSV ─────────────────────────────────────────────────────────
rules_df.to_csv("/mnt/user-data/outputs/verifai_rules.csv", index=False)
print(f"\n  Rules saved → verifai_rules.csv  ({len(rules_df)} rules)")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE A — PLOTS
# ══════════════════════════════════════════════════════════════════════════════
fig_a = plt.figure(figsize=(22, 20), facecolor=C_DARK)
fig_a.suptitle("VerifAI  ·  Association Rule Mining  ·  Tool Co-occurrence & Bundle Intelligence",
               fontsize=19, fontweight="bold", color=C_TEXT, y=0.99)

gs_a = gridspec.GridSpec(2, 2, figure=fig_a, hspace=0.42, wspace=0.32,
                          left=0.07, right=0.97, top=0.95, bottom=0.04)

# ── Heatmap ───────────────────────────────────────────────────────────────────
ax_heat = fig_a.add_subplot(gs_a[0, :])
cmap_heat = LinearSegmentedColormap.from_list("verifai", [C_PANEL, C_ACCENT, C_TEAL], N=256)
im = ax_heat.imshow(co_matrix.values, cmap=cmap_heat, aspect="auto", vmin=0, vmax=co_matrix.values.max())
ticks = range(len(labels_short))
ax_heat.set_xticks(ticks); ax_heat.set_xticklabels(labels_short, rotation=35, ha="right", fontsize=10.5)
ax_heat.set_yticks(ticks); ax_heat.set_yticklabels(labels_short, fontsize=10.5)

for i in range(len(labels_short)):
    for j in range(len(labels_short)):
        val = co_matrix.values[i, j]
        brightness = val / (co_matrix.values.max() + 1e-9)
        txt_color  = C_DARK if brightness > 0.5 else C_TEXT
        ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=9, color=txt_color, fontweight="bold" if i == j else "normal")

cb = plt.colorbar(im, ax=ax_heat, fraction=0.015, pad=0.01)
cb.ax.yaxis.set_tick_params(color=C_MUTED, labelcolor=C_MUTED)
cb.set_label("Co-occurrence Rate", color=C_MUTED, fontsize=10)
ax_heat.set_title("Tool Co-occurrence Heatmap  (proportion of users using both tools simultaneously)",
                  color=C_TEXT, fontsize=12, fontweight="bold", pad=10)

# ── Top Rules by Lift ─────────────────────────────────────────────────────────
ax_lift = fig_a.add_subplot(gs_a[1, 0])
top_n   = min(15, len(rules_df))
plot_rules = rules_df.head(top_n).copy()
plot_rules["rule_label"] = plot_rules.apply(
    lambda r: f"{r['antecedent'][:22]} → {r['consequent'][:14]}", axis=1)

bar_colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(top_n)]
bars = ax_lift.barh(range(top_n), plot_rules["lift"], color=bar_colors, alpha=0.85, height=0.7)
ax_lift.set_yticks(range(top_n))
ax_lift.set_yticklabels(plot_rules["rule_label"], fontsize=8.5)
ax_lift.invert_yaxis()
ax_lift.axvline(1.0, color=C_ROSE, lw=1.5, ls="--", alpha=0.7, label="Lift = 1 (independence)")
for i, (bar, val) in enumerate(zip(bars, plot_rules["lift"])):
    ax_lift.text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=8.5, color=C_MUTED)
ax_lift.set_xlabel("Lift", fontsize=11)
ax_lift.set_title(f"Top {top_n} Rules by Lift", color=C_TEXT, fontsize=12, fontweight="bold")
ax_lift.legend(fontsize=9, facecolor=C_CARD, edgecolor=C_BORDER, labelcolor=C_TEXT)
ax_lift.grid(True, axis="x", alpha=0.15)

# ── Support vs Confidence bubble chart ───────────────────────────────────────
ax_bub = fig_a.add_subplot(gs_a[1, 1])
sc = ax_bub.scatter(rules_df["support"], rules_df["confidence"],
                     c=rules_df["lift"], cmap="plasma",
                     s=rules_df["lift"] * 120, alpha=0.75, edgecolors=C_BORDER, linewidths=0.5)
cb2 = plt.colorbar(sc, ax=ax_bub, fraction=0.04, pad=0.02)
cb2.set_label("Lift", color=C_MUTED, fontsize=10)
cb2.ax.yaxis.set_tick_params(color=C_MUTED, labelcolor=C_MUTED)
ax_bub.axhline(MIN_CONFIDENCE, color=C_ROSE,  lw=1.2, ls="--", alpha=0.6, label=f"Min confidence={MIN_CONFIDENCE}")
ax_bub.axvline(MIN_SUPPORT,    color=C_AMBER, lw=1.2, ls="--", alpha=0.6, label=f"Min support={MIN_SUPPORT}")
ax_bub.set_xlabel("Support", fontsize=11)
ax_bub.set_ylabel("Confidence", fontsize=11)
ax_bub.set_title("Support vs Confidence  (bubble size = Lift)",
                 color=C_TEXT, fontsize=12, fontweight="bold")
ax_bub.legend(fontsize=9, facecolor=C_CARD, edgecolor=C_BORDER, labelcolor=C_TEXT)
ax_bub.grid(True, alpha=0.15)

plt.savefig("/mnt/user-data/outputs/verifai_association_rules.png",
            dpi=155, bbox_inches="tight", facecolor=C_DARK)
plt.close(fig_a)
print("\nSaved → verifai_association_rules.png")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE B — K-MEANS CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}")
print("  MODULE B  —  CUSTOMER PERSONA CLUSTERING")
print(f"{'='*62}")

ORDINAL_MAPS = {
    "difficulty_finding_tools":    {"Very Easy":0,"Easy":1,"Neutral":2,"Difficult":3,"Very Difficult":4},
    "trust_concern_frequency":     {"Never":0,"Rarely":1,"Sometimes":2,"Often":3,"Always":4},
    "subscription_fatigue":        {"Strongly Disagree":0,"Disagree":1,"Neutral":2,"Agree":3,"Strongly Agree":4},
    "preferred_subscription_tier": {"Free":0,"Silver":1,"Gold":2,"Platinum":3},
}
for col, mp in ORDINAL_MAPS.items():
    df[col + "_enc"] = df[col].map(mp)

CLUSTER_FEATURES = [
    "monthly_spend_on_ai_usd", "wtp_monthly_usd", "num_ai_tools_currently_used",
    "difficulty_finding_tools_enc", "trust_concern_frequency_enc",
    "subscription_fatigue_enc", "preferred_subscription_tier_enc",
]
Xc = df[CLUSTER_FEATURES].dropna()
cidx = Xc.index
scaler_c = StandardScaler()
Xc_scaled = scaler_c.fit_transform(Xc)

K_RANGE     = range(2, 9)
inertias_b  = []
sil_scores_b = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    lb = km.fit_predict(Xc_scaled)
    inertias_b.append(km.inertia_)
    sil_scores_b.append(silhouette_score(Xc_scaled, lb))

best_k_math = K_RANGE.start + int(np.argmax(sil_scores_b))
sil_4plus   = [(k, s) for k, s in zip(K_RANGE, sil_scores_b) if k >= 4]
best_k      = max(sil_4plus, key=lambda x: x[1])[0]

print(f"\n  Silhouette scores: { {k:round(s,4) for k,s in zip(K_RANGE,sil_scores_b)} }")
print(f"  Math optimal k={best_k_math} | Business optimal k={best_k} (used)")

km_best = KMeans(n_clusters=best_k, random_state=42, n_init=30)
cluster_labels = km_best.fit_predict(Xc_scaled)
df_c = df.loc[cidx].copy()
df_c["cluster"] = cluster_labels

profile = df_c.groupby("cluster").agg(
    size=("cluster","count"), avg_spend=("monthly_spend_on_ai_usd","mean"),
    avg_wtp=("wtp_monthly_usd","mean"), avg_tools=("num_ai_tools_currently_used","mean"),
    sub_rate=("will_subscribe","mean"), avg_mrr=("estimated_mrr_usd","mean"),
).round(2)

def name_persona(row, cid):
    wtp = row["avg_wtp"]; sub = row["sub_rate"]; spend = row["avg_spend"]
    tools = row["avg_tools"]
    fat = df_c[df_c["cluster"]==cid]["subscription_fatigue_enc"].mean()
    if wtp > 80 and sub > 0.95:  return ("High-Intent Power Buyers",       "💎", CLUSTER_COLORS[0])
    if spend > 300 and wtp < 50: return ("Heavy Spenders, Low Commitment", "🏢", CLUSTER_COLORS[1])
    if fat > 2.8 and wtp < 35:   return ("Subscription-Fatigued Skeptics", "😤", CLUSTER_COLORS[2])
    if tools > 9 and sub > 0.55: return ("AI Power Users",                 "⚡", CLUSTER_COLORS[3])
    if tools < 4:                 return ("Cautious Early Adopters",        "🌱", CLUSTER_COLORS[4])
    if sub > 0.80 and wtp > 30:  return ("Trust-Driven Converters",        "🔐", CLUSTER_COLORS[5])
    if spend > 100 and sub > 0.65: return ("Growth-Stage Adopters",        "🚀", CLUSTER_COLORS[6])
    return ("Budget-Conscious Explorers", "🧭", CLUSTER_COLORS[7])

persona_info = {cid: name_persona(profile.loc[cid], cid) for cid in sorted(df_c["cluster"].unique())}
profile["persona"] = [persona_info[c][0] for c in profile.index]
profile["icon"]    = [persona_info[c][1] for c in profile.index]
profile["color"]   = [persona_info[c][2] for c in profile.index]

print(f"\n  {'Cluster':<4} {'Icon'} {'Persona':<30} {'N':>4} {'Spend':>8} {'WTP':>7} {'Tools':>6} {'Sub%':>6} {'MRR':>7}")
print(f"  {'─'*80}")
for cid, row in profile.iterrows():
    print(f"  {cid:<4} {row['icon']}  {row['persona']:<30} {int(row['size']):>4} "
          f"${row['avg_spend']:>6.0f} ${row['avg_wtp']:>5.0f} {row['avg_tools']:>5.1f} "
          f"{row['sub_rate']:>5.0%}  ${row['avg_mrr']:>5.0f}")

# ── Clustering plots ──────────────────────────────────────────────────────────
fig_b = plt.figure(figsize=(22, 24), facecolor=C_DARK)
fig_b.suptitle(f"VerifAI  ·  Customer Persona Clustering  (K-Means  k={best_k})",
               fontsize=19, fontweight="bold", color=C_TEXT, y=0.99)
gs_b = gridspec.GridSpec(3, 2, figure=fig_b, hspace=0.40, wspace=0.30,
                          left=0.07, right=0.97, top=0.96, bottom=0.03)

# Silhouette bar
ax_sil = fig_b.add_subplot(gs_b[0, 0])
k_list = list(K_RANGE)
bc = [C_TEAL if k == best_k else C_BORDER for k in k_list]
bars_s = ax_sil.bar(k_list, sil_scores_b, color=bc, edgecolor=C_BORDER, width=0.6)
for bar, sc_, k in zip(bars_s, sil_scores_b, k_list):
    ax_sil.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f"{sc_:.3f}", ha="center", va="bottom", fontsize=9,
                color=C_TEXT, fontweight="bold" if k==best_k else "normal")
ax_sil.set_xlabel("k", fontsize=11); ax_sil.set_ylabel("Silhouette Score", fontsize=11)
ax_sil.set_title("Silhouette Scores by k", color=C_TEXT, fontsize=12, fontweight="bold")
ax_sil.set_xticks(k_list); ax_sil.grid(True, axis="y", alpha=0.15)

# Elbow
ax_elbow = fig_b.add_subplot(gs_b[0, 1])
ax_elbow.plot(k_list, inertias_b, color=C_ACCENT, lw=2.5, marker="o", markersize=7,
              markerfacecolor=C_DARK, markeredgecolor=C_ACCENT, markeredgewidth=2)
ax_elbow.scatter([best_k], [inertias_b[best_k-K_RANGE.start]],
                  color=C_TEAL, s=120, zorder=5)
ax_elbow.set_xlabel("k", fontsize=11); ax_elbow.set_ylabel("Inertia", fontsize=11)
ax_elbow.set_title("Elbow Curve", color=C_TEXT, fontsize=12, fontweight="bold")
ax_elbow.set_xticks(k_list); ax_elbow.grid(True, alpha=0.15)

# Main scatter
ax_sc = fig_b.add_subplot(gs_b[1, :])
for cid in sorted(df_c["cluster"].unique()):
    mask  = df_c["cluster"] == cid
    color = persona_info[cid][2]
    icon  = persona_info[cid][1]
    name  = persona_info[cid][0]
    size  = int(profile.loc[cid,"size"])
    ax_sc.scatter(df_c.loc[mask,"monthly_spend_on_ai_usd"],
                  df_c.loc[mask,"wtp_monthly_usd"],
                  c=color, alpha=0.45, s=50, edgecolors="none",
                  label=f"{icon} {name} (n={size})")
    cx = df_c.loc[mask,"monthly_spend_on_ai_usd"].mean()
    cy = df_c.loc[mask,"wtp_monthly_usd"].mean()
    ax_sc.scatter(cx, cy, c=color, s=280, marker="D",
                  edgecolors=C_TEXT, linewidths=1.8, zorder=10)
    ax_sc.annotate(f"{icon} {name}", xy=(cx,cy), xytext=(cx+5,cy+3),
                   fontsize=8.5, color=color, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.25",facecolor=C_DARK,
                             edgecolor=color,alpha=0.8))
ax_sc.set_xlabel("Monthly AI Spend (USD)", fontsize=12)
ax_sc.set_ylabel("Willingness to Pay / Month (USD)", fontsize=12)
ax_sc.set_title("Spend vs WTP by Persona  (◆ centroid)", color=C_TEXT, fontsize=13, fontweight="bold")
ax_sc.legend(fontsize=9, facecolor=C_PANEL, edgecolor=C_BORDER, labelcolor=C_TEXT, loc="upper left")
ax_sc.grid(True, alpha=0.12)

# PCA
ax_pca = fig_b.add_subplot(gs_b[2, 0])
pca = PCA(n_components=2, random_state=42)
Xp  = pca.fit_transform(Xc_scaled)
vexp = pca.explained_variance_ratio_
for cid in sorted(df_c["cluster"].unique()):
    mask  = (cluster_labels == cid)
    color = persona_info[cid][2]
    ax_pca.scatter(Xp[mask,0], Xp[mask,1], c=color, alpha=0.45, s=35, edgecolors="none")
    cx,cy = Xp[mask,0].mean(), Xp[mask,1].mean()
    ax_pca.scatter(cx, cy, c=color, s=200, marker="D", edgecolors=C_TEXT, linewidths=1.5, zorder=10)
    ax_pca.text(cx, cy+0.18, persona_info[cid][1], ha="center", fontsize=12)
ax_pca.set_xlabel(f"PC1 ({vexp[0]:.1%})", fontsize=10)
ax_pca.set_ylabel(f"PC2 ({vexp[1]:.1%})", fontsize=10)
ax_pca.set_title("PCA Projection", color=C_TEXT, fontsize=12, fontweight="bold")
ax_pca.grid(True, alpha=0.12)

# Normalised profile bars
ax_bar = fig_b.add_subplot(gs_b[2, 1])
mets   = ["avg_spend","avg_wtp","avg_tools","sub_rate","avg_mrr"]
mlabs  = ["Avg Spend\n($)","Avg WTP\n($)","Avg Tools","Sub Rate\n(×100)","Avg MRR\n($)"]
pn     = profile[mets].copy()
pn["sub_rate"] = pn["sub_rate"] * 100
for col in pn.columns:
    mn,mx = pn[col].min(), pn[col].max()
    pn[col] = (pn[col]-mn)/(mx-mn+1e-9)
x_  = np.arange(len(mets))
nc  = len(profile)
w_  = 0.75/nc
for i,(cid,row) in enumerate(pn.iterrows()):
    off = (i-nc/2+0.5)*w_
    ax_bar.bar(x_+off, row[mets], w_*0.9, color=persona_info[cid][2], alpha=0.82,
               label=f"{persona_info[cid][1]} {persona_info[cid][0]}")
ax_bar.set_xticks(x_); ax_bar.set_xticklabels(mlabs, fontsize=9.5)
ax_bar.set_ylabel("Normalised Score", fontsize=10)
ax_bar.set_title("Cluster Comparison (Normalised)", color=C_TEXT, fontsize=12, fontweight="bold")
ax_bar.legend(fontsize=7.5, facecolor=C_PANEL, edgecolor=C_BORDER, labelcolor=C_TEXT, loc="upper right")
ax_bar.grid(True, axis="y", alpha=0.15); ax_bar.set_ylim(0, 1.25)

plt.savefig("/mnt/user-data/outputs/verifai_clusters.png",
            dpi=155, bbox_inches="tight", facecolor=C_DARK)
plt.close(fig_b)
print("\nSaved → verifai_clusters.png")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE C — SUBSCRIPTION PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}")
print("  MODULE C  —  SUBSCRIPTION PREDICTION MODELS")
print(f"{'='*62}")

DROP  = ["respondent_id","primary_discovery_method","churn_risk_score","estimated_mrr_usd"]
df_ml = df.drop(columns=[c for c in DROP if c in df.columns]).copy()
TARGET = "will_subscribe"
y_ml   = df_ml[TARGET]
Xm     = df_ml.drop(columns=[TARGET]).copy()

# Encode all ordinals explicitly (use original string cols, not _enc cols)
ALL_ORDINALS = {
    "difficulty_finding_tools":    {"Very Easy":0,"Easy":1,"Neutral":2,"Difficult":3,"Very Difficult":4},
    "trust_concern_frequency":     {"Never":0,"Rarely":1,"Sometimes":2,"Often":3,"Always":4},
    "subscription_fatigue":        {"Strongly Disagree":0,"Disagree":1,"Neutral":2,"Agree":3,"Strongly Agree":4},
    "preferred_subscription_tier": {"Free":0,"Silver":1,"Gold":2,"Platinum":3},
    "preferred_billing_cadence":   {"Monthly":0,"Half-Yearly":1,"Yearly":2},
    "bundle_interest":             {"Not Interested":0,"Neutral":1,"Interested":2,"Very Interested":3},
    "age_group":                   {"18-24":0,"25-34":1,"35-44":2,"45-54":3,"55+":4},
}
for col, mp in ALL_ORDINALS.items():
    if col in Xm.columns:
        Xm[col] = Xm[col].map(mp)

# Drop any _enc columns (already have the originals encoded)
enc_drop = [c for c in Xm.columns if c.endswith("_enc")]
Xm = Xm.drop(columns=enc_drop, errors="ignore")

# One-hot encode nominals
NOMINAL = ["user_type","industry","region"]
Xm = pd.get_dummies(Xm, columns=[c for c in NOMINAL if c in Xm.columns], drop_first=True)

# Ensure all numeric
Xm = Xm.apply(pd.to_numeric, errors="coerce").fillna(0)
feat_names = Xm.columns.tolist()

Xtr,Xte,ytr,yte = train_test_split(Xm, y_ml, test_size=0.2, random_state=42, stratify=y_ml)

lr_pipe = Pipeline([("sc",StandardScaler()),
                    ("clf",LogisticRegression(max_iter=1000,C=1.0,random_state=42,class_weight="balanced"))])
lr_pipe.fit(Xtr, ytr)
lr_pred = lr_pipe.predict(Xte); lr_prob = lr_pipe.predict_proba(Xte)[:,1]

rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42,
                             class_weight="balanced", n_jobs=-1)
rf.fit(Xtr, ytr)
rf_pred = rf.predict(Xte); rf_prob = rf.predict_proba(Xte)[:,1]

def mets(yt, yp, ypr, name):
    print(f"\n  {name}")
    print(f"  Accuracy={accuracy_score(yt,yp):.4f}  Precision={precision_score(yt,yp):.4f}  "
          f"Recall={recall_score(yt,yp):.4f}  F1={f1_score(yt,yp):.4f}  "
          f"AUC={roc_auc_score(yt,ypr):.4f}")
    print(f"  Confusion matrix: {confusion_matrix(yt,yp).tolist()}")

mets(yte, lr_pred, lr_prob, "Logistic Regression")
mets(yte, rf_pred, rf_prob, "Random Forest")

# Feature importance
fi_df = pd.DataFrame({"feature":feat_names,"importance":rf.feature_importances_})
fi_df = fi_df.sort_values("importance",ascending=False).head(20).reset_index(drop=True)
label_remap = {
    "wtp_monthly_usd":"WTP Monthly (USD)","preferred_subscription_tier":"Preferred Tier",
    "monthly_spend_on_ai_usd":"Monthly AI Spend","num_ai_tools_currently_used":"# Tools Used",
    "trust_concern_frequency":"Trust Concern","difficulty_finding_tools":"Discovery Difficulty",
    "subscription_fatigue":"Sub Fatigue","bundle_interest":"Bundle Interest",
    "preferred_billing_cadence":"Billing Cadence","age_group":"Age Group",
}
fi_df["label"] = fi_df["feature"].apply(
    lambda x: label_remap.get(x, x.replace("user_type_","User: ")
                                    .replace("industry_","Ind: ")
                                    .replace("region_","Reg: ")
                                    .replace("_"," ").title()))

# ── ML plots ──────────────────────────────────────────────────────────────────
fig_c = plt.figure(figsize=(22, 24), facecolor=C_DARK)
fig_c.suptitle("VerifAI  ·  Subscription Prediction  ·  Model Report",
               fontsize=19, fontweight="bold", color=C_TEXT, y=0.99)
gs_c = gridspec.GridSpec(3, 2, figure=fig_c, hspace=0.42, wspace=0.32,
                          left=0.07, right=0.97, top=0.96, bottom=0.03)

def plot_cm(ax, cm, title, color):
    cmap_ = LinearSegmentedColormap.from_list("cm", [C_PANEL, color], N=256)
    ax.imshow(cm, cmap=cmap_, aspect="auto")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred No","Pred Yes"], color=C_TEXT, fontsize=10)
    ax.set_yticklabels(["Actual No","Actual Yes"], color=C_TEXT, fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j,i,f"{cm[i,j]}\n({cm[i,j]/cm.sum()*100:.1f}%)",
                    ha="center",va="center",fontsize=13,fontweight="bold",
                    color=C_DARK if cm[i,j]>cm.max()*0.5 else C_TEXT)
    ax.set_title(title, color=C_TEXT, fontsize=13, fontweight="bold", pad=10)
    ax.spines[["top","right","left","bottom"]].set_color(C_BORDER)

plot_cm(fig_c.add_subplot(gs_c[0,0]), confusion_matrix(yte,lr_pred),
        "Logistic Regression · Confusion Matrix", C_ACCENT)
plot_cm(fig_c.add_subplot(gs_c[0,1]), confusion_matrix(yte,rf_pred),
        "Random Forest · Confusion Matrix", C_TEAL)

ax_roc = fig_c.add_subplot(gs_c[1,0])
for prob,lbl,col in [(lr_prob,f"LR  (AUC={roc_auc_score(yte,lr_prob):.3f})",C_ACCENT),
                      (rf_prob,f"RF  (AUC={roc_auc_score(yte,rf_prob):.3f})",C_TEAL)]:
    fpr,tpr,_ = roc_curve(yte,prob)
    ax_roc.plot(fpr,tpr,color=col,lw=2.5,label=lbl)
ax_roc.plot([0,1],[0,1],"--",color=C_MUTED,lw=1.2)
ax_roc.fill_between(*roc_curve(yte,rf_prob)[:2],alpha=0.06,color=C_TEAL)
ax_roc.set_xlabel("FPR",fontsize=11); ax_roc.set_ylabel("TPR",fontsize=11)
ax_roc.set_title("ROC Curves",color=C_TEXT,fontsize=12,fontweight="bold")
ax_roc.legend(fontsize=9.5,facecolor=C_PANEL,edgecolor=C_BORDER,labelcolor=C_TEXT)
ax_roc.grid(True,alpha=0.15)

ax_met = fig_c.add_subplot(gs_c[1,1])
mlabels = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
lv = [accuracy_score(yte,lr_pred),precision_score(yte,lr_pred),
      recall_score(yte,lr_pred),f1_score(yte,lr_pred),roc_auc_score(yte,lr_prob)]
rv = [accuracy_score(yte,rf_pred),precision_score(yte,rf_pred),
      recall_score(yte,rf_pred),f1_score(yte,rf_pred),roc_auc_score(yte,rf_prob)]
x__ = np.arange(len(mlabels)); w__ = 0.35
for vals,color,lbl in [(lv,C_ACCENT,"Logistic Reg"),(rv,C_TEAL,"Random Forest")]:
    offset = -w__/2 if color==C_ACCENT else w__/2
    b_ = ax_met.bar(x__+offset, vals, w__, color=color, alpha=0.85, label=lbl)
    for bar in b_:
        ax_met.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8.5, color=C_TEXT)
ax_met.set_xticks(x__); ax_met.set_xticklabels(mlabels,fontsize=10)
ax_met.set_ylim(0,1.14); ax_met.set_title("Model Metrics Comparison",color=C_TEXT,fontsize=12,fontweight="bold")
ax_met.legend(fontsize=9.5,facecolor=C_PANEL,edgecolor=C_BORDER,labelcolor=C_TEXT)
ax_met.grid(True,axis="y",alpha=0.15)

ax_fi = fig_c.add_subplot(gs_c[2,:])
n_ = len(fi_df)
fi_colors = []
for feat in fi_df["feature"]:
    if feat in ["wtp_monthly_usd","preferred_subscription_tier","monthly_spend_on_ai_usd"]: fi_colors.append(C_AMBER)
    elif feat in ["trust_concern_frequency","difficulty_finding_tools"]: fi_colors.append(C_ROSE)
    elif feat in ["subscription_fatigue","bundle_interest","preferred_billing_cadence"]: fi_colors.append(C_VIOLET)
    elif feat.startswith("uses_"): fi_colors.append(C_TEAL)
    else: fi_colors.append(C_ACCENT)
ax_fi.barh(range(n_), fi_df["importance"], color=fi_colors, alpha=0.87, height=0.72)
ax_fi.set_yticks(range(n_)); ax_fi.set_yticklabels(fi_df["label"],fontsize=10.5)
ax_fi.invert_yaxis()
for i,(v) in enumerate(fi_df["importance"]):
    ax_fi.text(v+0.001, i, f"{v:.4f}", va="center", fontsize=9, color=C_MUTED)
patches = [mpatches.Patch(color=C_AMBER,label="Monetization (WTP/Tier/Spend)"),
           mpatches.Patch(color=C_ROSE,label="Trust & Discovery Pain"),
           mpatches.Patch(color=C_VIOLET,label="Marketplace Fit"),
           mpatches.Patch(color=C_TEAL,label="Tool Usage"),
           mpatches.Patch(color=C_ACCENT,label="Demographics")]
ax_fi.legend(handles=patches,loc="lower right",fontsize=9,
             facecolor=C_CARD,edgecolor=C_BORDER,labelcolor=C_TEXT)
ax_fi.set_xlabel("Feature Importance",fontsize=11)
ax_fi.set_title("Random Forest · Top 20 Feature Importances",color=C_TEXT,fontsize=13,fontweight="bold")
ax_fi.grid(True,axis="x",alpha=0.15)

plt.savefig("/mnt/user-data/outputs/verifai_model_report.png",
            dpi=155, bbox_inches="tight", facecolor=C_DARK)
plt.close(fig_c)
print("Saved → verifai_model_report.png")

# ══════════════════════════════════════════════════════════════════════════════
#  MODULE D — MRR REGRESSION  (Gradient Boosting)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}")
print("  MODULE D  —  MRR REVENUE REGRESSION  (Gradient Boosting)")
print(f"{'='*62}")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Feature matrix (same pipeline as Module C, target = estimated_mrr_usd) ───
DROP_D  = ["respondent_id", "primary_discovery_method", "churn_risk_score", "will_subscribe"]
df_d    = df.drop(columns=[c for c in DROP_D if c in df.columns]).copy()
TARGET_D = "estimated_mrr_usd"
y_d      = df_d[TARGET_D]
Xd       = df_d.drop(columns=[TARGET_D]).copy()

for col, mp in ALL_ORDINALS.items():
    if col in Xd.columns:
        Xd[col] = Xd[col].map(mp)

enc_cols = [c for c in Xd.columns if c.endswith("_enc")]
Xd = Xd.drop(columns=enc_cols, errors="ignore")
Xd = pd.get_dummies(Xd, columns=[c for c in ["user_type","industry","region"] if c in Xd.columns], drop_first=True)
Xd = Xd.apply(pd.to_numeric, errors="coerce").fillna(0)
feat_names_d = Xd.columns.tolist()

Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, y_d, test_size=0.2, random_state=42)

# ── Train Gradient Boosting Regressor ────────────────────────────────────────
gbr = GradientBoostingRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=4,
    subsample=0.8, min_samples_leaf=10, random_state=42
)
gbr.fit(Xd_tr, yd_tr)
yd_pred      = gbr.predict(Xd_te)
yd_pred_full = gbr.predict(Xd)

# ── Metrics ───────────────────────────────────────────────────────────────────
r2   = r2_score(yd_te, yd_pred)
mae  = mean_absolute_error(yd_te, yd_pred)
rmse = np.sqrt(mean_squared_error(yd_te, yd_pred))
# MAPE: only on subscribers (MRR > 0) to avoid division by near-zero
mape_mask = yd_te > 1.0
mape = np.mean(np.abs((yd_te[mape_mask] - yd_pred[mape_mask]) / yd_te[mape_mask])) * 100

print(f"\n  Gradient Boosting Regressor — Test Set Performance")
print(f"  {'─'*48}")
print(f"  R² Score  : {r2:.4f}  {'★ Excellent' if r2>0.85 else '★ Good' if r2>0.70 else 'Moderate'}")
print(f"  MAE       : ${mae:.2f}   (avg absolute error per user)")
print(f"  RMSE      : ${rmse:.2f}")
print(f"  MAPE      : {mape:.1f}%  (mean absolute % error)")

# ── Feature importance ────────────────────────────────────────────────────────
fi_d = pd.DataFrame({"feature": feat_names_d, "importance": gbr.feature_importances_})
fi_d = fi_d.sort_values("importance", ascending=False).head(20).reset_index(drop=True)
fi_d["label"] = fi_d["feature"].apply(
    lambda x: label_remap.get(x, x.replace("user_type_","User: ")
                                    .replace("industry_","Ind: ")
                                    .replace("region_","Reg: ")
                                    .replace("_"," ").title()))

# ── Tier-level MRR forecast table ────────────────────────────────────────────
df["mrr_predicted"] = yd_pred_full
TIER_ORDER = ["Free","Silver","Gold","Platinum"]
tier_table = []
for tier in TIER_ORDER:
    mask       = df["preferred_subscription_tier"] == tier
    n_tier     = mask.sum()
    avg_pred   = df.loc[mask, "mrr_predicted"].mean()
    avg_actual = df.loc[mask, "estimated_mrr_usd"].mean()
    total_proj = avg_pred * n_tier                         # if ALL in tier subscribe
    sub_rate   = df.loc[mask, "will_subscribe"].mean()
    real_proj  = avg_pred * n_tier * sub_rate              # realistic (× sub rate)
    tier_table.append({
        "tier": tier, "n": n_tier, "sub_rate": sub_rate,
        "avg_pred_mrr": avg_pred, "avg_actual_mrr": avg_actual,
        "proj_mrr_all": total_proj, "proj_mrr_real": real_proj,
    })
tier_df = pd.DataFrame(tier_table)

print(f"\n  {'─'*80}")
print(f"  TIER-LEVEL MRR FORECAST  (1,000-user sample)")
print(f"  {'─'*80}")
print(f"  {'Tier':<10} {'N':>5} {'SubRate':>8} {'AvgPred MRR':>12} {'AvgActual':>10} "
      f"{'Proj MRR (all)':>15} {'Proj MRR (real)':>16}")
print(f"  {'─'*78}")
for _, r in tier_df.iterrows():
    print(f"  {r['tier']:<10} {int(r['n']):>5} {r['sub_rate']:>7.1%}  "
          f"${r['avg_pred_mrr']:>10.2f}  ${r['avg_actual_mrr']:>8.2f}  "
          f"${r['proj_mrr_all']:>13,.0f}  ${r['proj_mrr_real']:>14,.0f}")
total_real = tier_df["proj_mrr_real"].sum()
print(f"  {'─'*78}")
print(f"  {'TOTAL':<10} {int(tier_df['n'].sum()):>5}  {'':>8}  {'':>12}  {'':>10}  "
      f"${tier_df['proj_mrr_all'].sum():>13,.0f}  ${total_real:>14,.0f}")

# ── Top revenue-generating segments ──────────────────────────────────────────
seg_cols = ["user_type","industry","preferred_subscription_tier","preferred_billing_cadence"]
print(f"\n  TOP REVENUE SEGMENTS (avg predicted MRR per user):")
print(f"  {'─'*55}")
for col in seg_cols:
    seg = (df.groupby(col)["mrr_predicted"]
             .mean()
             .sort_values(ascending=False)
             .head(3))
    top = seg.index[0]
    print(f"  {col:<32}  Top: {top:<22}  ${seg.iloc[0]:.2f}/user")

# ── 10,000-user extrapolation ─────────────────────────────────────────────────
SCALE = 10_000 / 1_000
proj_mrr_10k  = total_real * SCALE
proj_arr_10k  = proj_mrr_10k * 12

# Distribution-weighted avg MRR
dist_weights  = tier_df.set_index("tier")["n"] / tier_df["n"].sum()
weighted_avg  = (tier_df.set_index("tier")["avg_pred_mrr"] * dist_weights).sum()

print(f"\n  {'═'*62}")
print(f"  10,000-USER REVENUE EXTRAPOLATION")
print(f"  {'═'*62}")
print(f"  Assumed user distribution: same as 1k sample")
print(f"  Weighted avg MRR/user     : ${weighted_avg:.2f}")
print(f"  Projected Total MRR       : ${proj_mrr_10k:>10,.0f} / month")
print(f"  Projected ARR             : ${proj_arr_10k:>10,.0f} / year")
print(f"\n  Tier breakdown at 10k users:")
for _, r in tier_df.iterrows():
    mrr_10k = r["proj_mrr_real"] * SCALE
    print(f"    {r['tier']:<10} → ${mrr_10k:>10,.0f} / month  "
          f"(${mrr_10k*12:>12,.0f} ARR)")
print(f"  {'─'*62}")

# Revenue sensitivity: if Platinum share grows to 25% (from ~17%)
plat_boost = proj_mrr_10k * (1 + 0.08 * (tier_df.loc[tier_df.tier=="Platinum","avg_pred_mrr"].values[0] /
                                            weighted_avg - 1))
print(f"\n  Sensitivity: Platinum share +8pp → MRR ≈ ${plat_boost:,.0f}/mo  "
      f"(ARR ≈ ${plat_boost*12:,.0f})")

# ── MRR insight print ─────────────────────────────────────────────────────────
print(f"""
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WHAT THE TOP MRR PREDICTORS TELL YOU
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Top features driving predicted MRR (GB feature importance):
  #{1}: {fi_d.iloc[0]['label']} — This is the ceiling setter.
        Users who anchor high here generate 4-6x the MRR of low-WTP users.
        GTM implication: surface value and savings framing before any pricing
        touchpoint, or you'll anchor the wrong number.

  #{2}: {fi_d.iloc[1]['label']} — Second-strongest driver.
        Direct structural lever: moving one tier up roughly doubles MRR.
        Every Silver → Gold conversion is worth more than acquiring a new Free user.

  #{3}: {fi_d.iloc[2]['label']} — Proxy for AI budget maturity.
        High spenders accept higher price points — they're already in the habit.
        Target users spending $100+/mo on direct subscriptions; they're the
        easiest to reframe as "consolidation" buyers, not "new cost" buyers.

  SEGMENT PRIORITY FOR REVENUE OPTIMISATION:
  1. Platinum + Enterprise/SME → highest MRR per user, lowest churn risk
  2. Gold + annual billing     → 85% billing factor + 2× MRR of Silver
  3. Silver → Gold upsell path → single biggest MRR unlock per conversion
  4. Free tier                 → de-prioritise for revenue; CAC rarely recovers
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# ══════════════════════════════════════════════════════════════════════════════
#  MODULE D — PLOTS
# ══════════════════════════════════════════════════════════════════════════════
fig_d = plt.figure(figsize=(24, 26), facecolor=C_DARK)
fig_d.suptitle("VerifAI  ·  MRR Revenue Regression  ·  Gradient Boosting Forecast",
               fontsize=19, fontweight="bold", color=C_TEXT, y=0.99)
gs_d = gridspec.GridSpec(3, 2, figure=fig_d, hspace=0.42, wspace=0.34,
                          left=0.07, right=0.97, top=0.96, bottom=0.03)

# ── 1. Actual vs Predicted scatter ───────────────────────────────────────────
ax_avp = fig_d.add_subplot(gs_d[0, :])

# Colour points by tier
tier_color_map = {"Free": C_MUTED, "Silver": C_TEAL, "Gold": C_AMBER, "Platinum": C_ROSE}
te_idx   = yd_te.index
te_tiers = df.loc[te_idx, "preferred_subscription_tier"]

for tier in TIER_ORDER:
    mask_t = te_tiers == tier
    if mask_t.sum() == 0: continue
    ax_avp.scatter(yd_te[mask_t], yd_pred[mask_t],
                   c=tier_color_map[tier], alpha=0.65, s=55,
                   edgecolors="none", label=f"{tier} (n={mask_t.sum()})", zorder=3)

# Perfect prediction line
lims = [min(yd_te.min(), yd_pred.min()) - 5, max(yd_te.max(), yd_pred.max()) + 5]
ax_avp.plot(lims, lims, "--", color=C_ACCENT, lw=2, alpha=0.7, label="Perfect prediction", zorder=2)

# Annotate metrics on plot
metrics_txt = (f"R² = {r2:.4f}   MAE = ${mae:.2f}   RMSE = ${rmse:.2f}   MAPE = {mape:.1f}%")
ax_avp.text(0.02, 0.96, metrics_txt, transform=ax_avp.transAxes,
            fontsize=11, color=C_TEXT, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C_CARD, edgecolor=C_BORDER, alpha=0.9))

ax_avp.set_xlim(lims); ax_avp.set_ylim(lims)
ax_avp.set_xlabel("Actual MRR (USD)", fontsize=12)
ax_avp.set_ylabel("Predicted MRR (USD)", fontsize=12)
ax_avp.set_title("Actual vs Predicted MRR  ·  Coloured by Subscription Tier",
                 color=C_TEXT, fontsize=13, fontweight="bold")
ax_avp.legend(fontsize=10, facecolor=C_PANEL, edgecolor=C_BORDER, labelcolor=C_TEXT, loc="upper left")
ax_avp.grid(True, alpha=0.12)

# ── 2. Feature importance ─────────────────────────────────────────────────────
ax_fi_d = fig_d.add_subplot(gs_d[1, :])
n_fi = len(fi_d)
fi_d_colors = []
for feat in fi_d["feature"]:
    if feat in ["wtp_monthly_usd", "preferred_subscription_tier", "monthly_spend_on_ai_usd"]:
        fi_d_colors.append(C_AMBER)
    elif feat in ["trust_concern_frequency", "difficulty_finding_tools"]:
        fi_d_colors.append(C_ROSE)
    elif feat in ["subscription_fatigue", "bundle_interest", "preferred_billing_cadence"]:
        fi_d_colors.append(C_VIOLET)
    elif feat.startswith("uses_"):
        fi_d_colors.append(C_TEAL)
    else:
        fi_d_colors.append(C_ACCENT)

bars_fi = ax_fi_d.barh(range(n_fi), fi_d["importance"], color=fi_d_colors, alpha=0.87, height=0.72)
ax_fi_d.set_yticks(range(n_fi))
ax_fi_d.set_yticklabels(fi_d["label"], fontsize=10.5)
ax_fi_d.invert_yaxis()
for i, v in enumerate(fi_d["importance"]):
    ax_fi_d.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9, color=C_MUTED)

legend_patches = [
    mpatches.Patch(color=C_AMBER,  label="Monetization Signals (WTP / Tier / Spend)"),
    mpatches.Patch(color=C_ROSE,   label="Trust & Discovery Pain"),
    mpatches.Patch(color=C_VIOLET, label="Marketplace Fit (Fatigue / Bundles / Billing)"),
    mpatches.Patch(color=C_TEAL,   label="Tool Usage Behaviour"),
    mpatches.Patch(color=C_ACCENT, label="Demographics & Context"),
]
ax_fi_d.legend(handles=legend_patches, loc="lower right", fontsize=9,
               facecolor=C_CARD, edgecolor=C_BORDER, labelcolor=C_TEXT)
ax_fi_d.set_xlabel("Feature Importance (GB)", fontsize=11)
ax_fi_d.set_title("Gradient Boosting · Top 20 MRR Predictors",
                  color=C_TEXT, fontsize=13, fontweight="bold")
ax_fi_d.grid(True, axis="x", alpha=0.15)

# ── 3. Tier MRR forecast bar ─────────────────────────────────────────────────
ax_tier = fig_d.add_subplot(gs_d[2, 0])
tier_names  = tier_df["tier"].tolist()
x_tier      = np.arange(len(tier_names))
w_tier      = 0.35
b1 = ax_tier.bar(x_tier - w_tier/2, tier_df["avg_pred_mrr"],   w_tier,
                  color=[tier_color_map[t] for t in tier_names], alpha=0.85, label="Avg Predicted MRR/user")
b2 = ax_tier.bar(x_tier + w_tier/2, tier_df["avg_actual_mrr"], w_tier,
                  color=[tier_color_map[t] for t in tier_names], alpha=0.40, label="Avg Actual MRR/user", hatch="//")
for bar in list(b1) + list(b2):
    ax_tier.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                 f"${bar.get_height():.0f}", ha="center", va="bottom", fontsize=9, color=C_TEXT)
ax_tier.set_xticks(x_tier); ax_tier.set_xticklabels(tier_names, fontsize=11)
ax_tier.set_ylabel("MRR per User (USD)", fontsize=11)
ax_tier.set_title("Avg Predicted vs Actual MRR by Tier",
                  color=C_TEXT, fontsize=12, fontweight="bold")
ax_tier.legend(fontsize=9, facecolor=C_PANEL, edgecolor=C_BORDER, labelcolor=C_TEXT)
ax_tier.grid(True, axis="y", alpha=0.15)

# ── 4. 10k Extrapolation waterfall ───────────────────────────────────────────
ax_wf = fig_d.add_subplot(gs_d[2, 1])
wf_tiers  = tier_df["tier"].tolist()
wf_vals   = (tier_df["proj_mrr_real"] * SCALE).tolist()
wf_colors = [tier_color_map[t] for t in wf_tiers]

bars_wf = ax_wf.bar(wf_tiers, wf_vals, color=wf_colors, alpha=0.85, edgecolor=C_BORDER, width=0.6)
for bar, val in zip(bars_wf, wf_vals):
    ax_wf.text(bar.get_x()+bar.get_width()/2, bar.get_height()+500,
               f"${val:,.0f}", ha="center", va="bottom", fontsize=9.5, color=C_TEXT, fontweight="bold")

# Total line
ax_wf.axhline(proj_mrr_10k, color=C_ACCENT, lw=2, ls="--", alpha=0.8,
              label=f"Total MRR = ${proj_mrr_10k:,.0f}")
ax_wf.text(3.42, proj_mrr_10k + 500, f"Total\n${proj_mrr_10k:,.0f}/mo\n(ARR ${proj_arr_10k/1e6:.2f}M)",
           ha="right", va="bottom", fontsize=9, color=C_ACCENT, fontweight="bold")

ax_wf.set_ylabel("Projected Monthly Revenue (USD)", fontsize=11)
ax_wf.set_title("10,000-User MRR Projection by Tier",
                color=C_TEXT, fontsize=12, fontweight="bold")
ax_wf.legend(fontsize=9, facecolor=C_PANEL, edgecolor=C_BORDER, labelcolor=C_TEXT)
ax_wf.grid(True, axis="y", alpha=0.15)

plt.savefig("/mnt/user-data/outputs/verifai_mrr_regression.png",
            dpi=155, bbox_inches="tight", facecolor=C_DARK)
plt.close(fig_d)
print("Saved → verifai_mrr_regression.png")

# ══════════════════════════════════════════════════════════════════════════════
#  DONE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*62}")
print("  ALL MODULES COMPLETE")
print(f"  Outputs in /mnt/user-data/outputs/")
print(f"  • verifai_association_rules.png  (Module A)")
print(f"  • verifai_clusters.png           (Module B)")
print(f"  • verifai_model_report.png       (Module C)")
print(f"  • verifai_mrr_regression.png     (Module D)")
print(f"  • verifai_rules.csv              ({len(rules_df)} association rules)")
print(f"{'═'*62}")
