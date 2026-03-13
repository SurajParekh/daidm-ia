# VerifAI Marketplace — Business Validation Dashboard

An interactive Streamlit dashboard for validating the **VerifAI AI App Store** business concept using data-driven analysis.

## 🚀 Live Demo
Deploy instantly on [Streamlit Community Cloud](https://streamlit.io/cloud) — free, no server required.

---

## 📦 Project Structure

```
├── app.py                        # Main Streamlit dashboard
├── requirements.txt              # Python dependencies
├── data/
│   └── synthetic_data.csv        # 1,000-row synthetic survey dataset
└── README.md
```

---

## 🔧 Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/verifai-dashboard.git
cd verifai-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser and upload `data/synthetic_data.csv`.

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set **Main file path** to `app.py`
4. Click **Deploy**

Once live, upload `data/synthetic_data.csv` via the sidebar file uploader.

---

## 📊 Dashboard Modules

| Tab | Method | Business Question Answered |
|---|---|---|
| 📊 Overview | Descriptive Stats + Charts | What does our survey data tell us? |
| 🔮 Classification | Random Forest + Logistic Regression | Which users will subscribe? |
| 👥 Clustering | K-Means | What are our buyer personas? |
| 🔗 Association Rules | Apriori-style Mining | Which tools get used together → bundles? |
| 📈 Regression | Gradient Boosting Regressor | What MRR can we forecast? |
| 📋 Data Explorer | Filterable Table | Explore the raw dataset |

---

## 🧪 Dataset Columns (25 variables)

| Column | Type | Description |
|---|---|---|
| respondent_id | ID | Unique survey respondent identifier |
| user_type | Categorical | Individual / Freelancer / SME / Enterprise / Student |
| industry | Categorical | Design, Dev, Marketing, Data, Finance, etc. |
| region | Categorical | Geographic region of respondent |
| age_group | Categorical | Age bracket |
| num_ai_tools_currently_used | Numeric | Count of AI tools currently in use |
| monthly_spend_on_ai_usd | Numeric | Current monthly spend on AI tools (USD) |
| primary_discovery_method | Categorical | How they currently find AI tools |
| difficulty_finding_tools | Ordinal | How hard it is to find quality tools |
| trust_concern_frequency | Ordinal | How often safety/security concerns arise |
| subscription_fatigue | Ordinal | Frustration with managing multiple subscriptions |
| uses_llm_writing_tools | Binary | Uses LLM/writing tools (0/1) |
| uses_image_gen_tools | Binary | Uses image generation tools (0/1) |
| uses_video_gen_tools | Binary | Uses video generation tools (0/1) |
| uses_code_assistant_tools | Binary | Uses code assistant tools (0/1) |
| uses_data_analytics_tools | Binary | Uses data analytics AI tools (0/1) |
| uses_voice_audio_tools | Binary | Uses voice/audio tools (0/1) |
| uses_productivity_tools | Binary | Uses productivity AI tools (0/1) |
| preferred_subscription_tier | Ordinal | Free / Silver / Gold / Platinum |
| preferred_billing_cadence | Ordinal | Monthly / Half-Yearly / Yearly |
| bundle_interest | Ordinal | Interest in curated tool bundles |
| wtp_monthly_usd | Numeric | Willingness to pay per month (USD) |
| will_subscribe | Binary | **Classification target** — likely to subscribe (0/1) |
| churn_risk_score | Numeric | Estimated churn probability (0–1) |
| estimated_mrr_usd | Numeric | **Regression target** — projected monthly revenue (USD) |

---

## 👤 Author
MBA Data Analytics Module — VerifAI Marketplace Business Validation  
Built with Python · Streamlit · scikit-learn · pandas · matplotlib
