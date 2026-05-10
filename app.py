# ============================================================
# RocketVision — Space Mission Launch Success Predictor
# Streamlit Web App
# Run: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="RocketVision",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #2e3554;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-label { font-size: 13px; color: #8892b0; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 32px; font-weight: 700; color: #ccd6f6; }
    .metric-delta { font-size: 12px; color: #64ffda; margin-top: 4px; }
    .section-header {
        font-size: 20px; font-weight: 600; color: #ccd6f6;
        border-left: 4px solid #64ffda;
        padding-left: 12px; margin: 24px 0 16px 0;
    }
    .predict-card {
        background: linear-gradient(135deg, #0a192f, #112240);
        border: 1px solid #64ffda44;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
    }
    .prob-value { font-size: 72px; font-weight: 800; margin: 10px 0; }
    .prob-label { font-size: 16px; color: #8892b0; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1e2130; border-radius: 8px;
        color: #8892b0; border: 1px solid #2e3554;
        padding: 8px 20px; font-size: 14px;
    }
    .stTabs [aria-selected="true"] {
        background: #112240 !important; color: #64ffda !important;
        border-color: #64ffda !important;
    }
    .info-box {
        background: #112240; border: 1px solid #1d4ed8;
        border-radius: 8px; padding: 14px 18px;
        font-size: 13px; color: #93c5fd; margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load & cache data ────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('Space_Corrected.csv', index_col=0)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'Company Name':   'company',
        'Location':       'location',
        'Datum':          'date',
        'Detail':         'rocket_detail',
        'Status Rocket':  'rocket_status',
        'Rocket':         'cost_million_usd',
        'Status Mission': 'mission_status',
    })
    df['date']   = pd.to_datetime(df['date'], utc=True, errors='coerce')
    df['year']   = df['date'].dt.year
    df['decade'] = (df['year'] // 10 * 10).astype('Int64')
    df['cost_million_usd'] = pd.to_numeric(
        df['cost_million_usd'].astype(str).str.replace(',','',regex=False).str.strip(),
        errors='coerce'
    )
    df['country'] = df['location'].str.split(',').str[-1].str.strip()
    df['success'] = df['mission_status'].apply(
        lambda x: 1 if str(x).strip() == 'Success' else 0
    )
    df = df.dropna(subset=['year','company','mission_status']).reset_index(drop=True)
    return df

@st.cache_resource
def train_model(df):
    model_df = df[['year','company','rocket_status','success']].copy().dropna()
    le = LabelEncoder()
    model_df['company_enc']   = le.fit_transform(model_df['company'])
    model_df['rocket_active'] = model_df['rocket_status'].apply(
        lambda x: 1 if 'Active' in str(x) else 0
    )
    X = model_df[['year','company_enc','rocket_active']].values
    y = model_df['success'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, le, acc, X_test, y_test

# ── Load ─────────────────────────────────────────────────────
df = load_data()
model, le, acc, X_test, y_test = train_model(df)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚀 RocketVision")
    st.markdown("*Space Mission Launch Success Predictor*")
    st.divider()

    st.markdown("### Filters")
    year_range = st.slider(
        "Year range", int(df.year.min()), int(df.year.max()),
        (1957, 2020)
    )
    selected_countries = st.multiselect(
        "Filter by country",
        options=sorted(df['country'].dropna().unique()),
        default=[]
    )
    st.divider()
    st.markdown(f"**Dataset:** {len(df):,} launches")
    st.markdown(f"**Period:** 1957 – 2020")
    st.markdown(f"**Model accuracy:** {acc*100:.1f}%")
    st.divider()
    st.markdown("**Project:** Prob & Stats Spring 2026")
    st.markdown("**Team:** RocketVision")

# ── Apply filters ────────────────────────────────────────────
filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
if selected_countries:
    filtered = filtered[filtered['country'].isin(selected_countries)]

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠  Dashboard",
    "📊  Charts & Analysis",
    "📈  Probability & Stats",
    "🎯  Mission Predictor"
])

# ============================================================
# TAB 1 — DASHBOARD
# ============================================================
with tab1:
    st.markdown("# 🚀 RocketVision Dashboard")
    st.markdown("*Analyzing 65 years of global space mission data*")

    # KPI cards
    total   = len(filtered)
    success = filtered['success'].sum()
    fail    = total - success
    rate    = success / total * 100 if total > 0 else 0
    countries_count = filtered['country'].nunique()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Launches</div>
            <div class="metric-value">{total:,}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Successes</div>
            <div class="metric-value" style="color:#64ffda">{success:,}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Failures</div>
            <div class="metric-value" style="color:#ff6b6b">{fail:,}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Success Rate</div>
            <div class="metric-value" style="color:#ffd700">{rate:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Countries</div>
            <div class="metric-value">{countries_count}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Overview charts side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Launches per Year</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 3.5), facecolor='#0e1117')
        ax.set_facecolor('#1e2130')
        yearly = filtered.groupby('year').size()
        ax.fill_between(yearly.index, yearly.values, alpha=0.3, color='#64ffda')
        ax.plot(yearly.index, yearly.values, color='#64ffda', linewidth=2)
        ax.set_xlabel('Year', color='#8892b0'); ax.set_ylabel('Launches', color='#8892b0')
        ax.tick_params(colors='#8892b0'); ax.spines[:].set_color('#2e3554')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-header">Mission Status Breakdown</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 3.5), facecolor='#0e1117')
        ax.set_facecolor('#1e2130')
        status_counts = filtered['mission_status'].value_counts()
        colors = ['#64ffda','#ff6b6b','#ffd700','#a78bfa']
        wedges, texts, autos = ax.pie(
            status_counts, labels=status_counts.index,
            autopct='%1.1f%%', colors=colors[:len(status_counts)],
            startangle=140, wedgeprops={'edgecolor':'#0e1117','linewidth':2},
            textprops={'color':'#ccd6f6', 'fontsize':10}
        )
        for a in autos: a.set_color('#0e1117'); a.set_fontweight('bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Top companies table
    st.markdown('<div class="section-header">Top Companies by Launch Count</div>', unsafe_allow_html=True)
    top_co = (
        filtered.groupby('company')
        .agg(launches=('success','count'), successes=('success','sum'))
        .assign(success_rate=lambda x: (x['successes']/x['launches']*100).round(1))
        .sort_values('launches', ascending=False)
        .head(10)
        .reset_index()
    )
    top_co.columns = ['Company','Total Launches','Successes','Success Rate (%)']
    st.dataframe(top_co, use_container_width=True, hide_index=True)

# ============================================================
# TAB 2 — CHARTS & ANALYSIS
# ============================================================
with tab2:
    st.markdown("# 📊 Charts & Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Chart: Top countries
        st.markdown('<div class="section-header">Top 10 Countries by Launches</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='#0e1117')
        ax.set_facecolor('#1e2130')
        top_c = filtered['country'].value_counts().head(10)
        bars = ax.bar(top_c.index, top_c.values, color='#64ffda', edgecolor='#0e1117', linewidth=0.8)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                    str(int(bar.get_height())), ha='center', color='#ccd6f6', fontsize=9)
        ax.set_xlabel('Country', color='#8892b0'); ax.set_ylabel('Launches', color='#8892b0')
        ax.tick_params(colors='#8892b0', axis='both'); ax.spines[:].set_color('#2e3554')
        plt.xticks(rotation=35, ha='right')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        # Chart: Success rate by decade
        st.markdown('<div class="section-header">Success Rate by Decade</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='#0e1117')
        ax.set_facecolor('#1e2130')
        decade_rate = filtered.groupby('decade')['success'].mean() * 100
        ax.plot(decade_rate.index.astype(int), decade_rate.values,
                marker='o', color='#ffd700', linewidth=2.5, markersize=9)
        for x, y in zip(decade_rate.index.astype(int), decade_rate.values):
            ax.annotate(f'{y:.0f}%', (x, y), textcoords='offset points',
                        xytext=(0, 12), ha='center', color='#ccd6f6', fontsize=9)
        ax.set_xlabel('Decade', color='#8892b0'); ax.set_ylabel('Success Rate (%)', color='#8892b0')
        ax.tick_params(colors='#8892b0'); ax.spines[:].set_color('#2e3554')
        ax.set_ylim(0, 110)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)

    with col3:
        # Chart: Company success rate
        st.markdown('<div class="section-header">Company Success Rates (≥20 launches)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='#0e1117')
        ax.set_facecolor('#1e2130')
        comp = (
            filtered.groupby('company')['success']
            .agg(['mean','count'])
            .query('count >= 20')
            .sort_values('mean', ascending=True)
            .tail(12)
        )
        comp['mean'] = comp['mean'] * 100
        bar_colors = ['#64ffda' if v >= 90 else '#ffd700' if v >= 75 else '#ff6b6b'
                      for v in comp['mean']]
        bars = ax.barh(comp.index, comp['mean'], color=bar_colors, edgecolor='#0e1117')
        for bar, val in zip(bars, comp['mean']):
            ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                    f'{val:.1f}%', va='center', color='#ccd6f6', fontsize=9)
        ax.set_xlabel('Success Rate (%)', color='#8892b0')
        ax.tick_params(colors='#8892b0'); ax.spines[:].set_color('#2e3554')
        ax.set_xlim(0, 112)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        # Chart: Launch volume scatter
        st.markdown('<div class="section-header">Launch Volume vs. Success Rate</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='#0e1117')
        ax.set_facecolor('#1e2130')
        scatter_data = (
            filtered.groupby('company')['success']
            .agg(['mean','count'])
            .query('count >= 5')
            .reset_index()
        )
        sc = ax.scatter(
            scatter_data['count'], scatter_data['mean']*100,
            alpha=0.7, s=80,
            c=scatter_data['mean'], cmap='RdYlGn',
            edgecolors='#0e1117', linewidths=0.8
        )
        plt.colorbar(sc, ax=ax, label='Success Rate')
        ax.set_xlabel('Total Launches', color='#8892b0')
        ax.set_ylabel('Success Rate (%)', color='#8892b0')
        ax.tick_params(colors='#8892b0'); ax.spines[:].set_color('#2e3554')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Confusion matrix full width
    st.markdown('<div class="section-header">Logistic Regression — Confusion Matrix</div>', unsafe_allow_html=True)
    col5, col6 = st.columns([1, 2])
    with col5:
        y_pred = model.predict(X_test)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0e1117')
        ax.set_facecolor('#1e2130')
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Failure','Success'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title('Confusion Matrix', color='#ccd6f6')
        ax.tick_params(colors='#ccd6f6')
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col6:
        st.markdown('<div class="info-box">The logistic regression model achieves <b>90.36% accuracy</b> on the test set. It correctly identifies successful missions with high recall (1.00), meaning it rarely misses a success. The model was trained on 3,358 samples and tested on 840 samples using an 80/20 split.</div>', unsafe_allow_html=True)
        feat_df = pd.DataFrame({
            'Feature':     ['Year', 'Company (encoded)', 'Rocket Active'],
            'Coefficient': [0.0410, 0.0090, -0.6445],
            'Effect':      ['Newer year → higher success', 'Company identity matters', 'Active rocket status affects probability']
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ============================================================
# TAB 3 — PROBABILITY & STATS
# ============================================================
with tab3:
    st.markdown("# 📈 Probability & Statistical Analysis")

    st.markdown('<div class="section-header">95% Confidence Intervals — Mission Success by Company</div>', unsafe_allow_html=True)
    st.markdown("Using the **Wilson Score method** (most accurate for proportions near 0 or 1).")

    # Compute CI table
    company_stats = (
        filtered.groupby('company')['success']
        .agg(['sum','count'])
        .rename(columns={'sum':'successes','count':'launches'})
        .query('launches >= 20')
        .sort_values('launches', ascending=False)
        .head(15)
    )
    z = 1.96
    rows = []
    for company, row in company_stats.iterrows():
        n, k = row['launches'], row['successes']
        p = k / n
        denom  = 1 + z**2/n
        centre = (p + z**2/(2*n)) / denom
        margin = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
        ci_low  = max(0, centre - margin) * 100
        ci_high = min(1, centre + margin) * 100
        rows.append({
            'Company': company, 'Launches': int(n), 'Successes': int(k),
            'Success %': f'{p*100:.1f}%',
            '95% CI Low': f'{ci_low:.1f}%',
            '95% CI High': f'{ci_high:.1f}%',
            'P(next launch succeeds)': f'{p*100:.1f}%'
        })
    ci_df = pd.DataFrame(rows)
    st.dataframe(ci_df, use_container_width=True, hide_index=True)

    # CI chart
    st.markdown('<div class="section-header">Confidence Interval Visualisation</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0e1117')
    ax.set_facecolor('#1e2130')
    companies_list = [r['Company'] for r in rows]
    means     = [float(r['Success %'].rstrip('%')) for r in rows]
    lows      = [float(r['95% CI Low'].rstrip('%')) for r in rows]
    highs     = [float(r['95% CI High'].rstrip('%')) for r in rows]
    err_low   = [m - l for m, l in zip(means, lows)]
    err_high  = [h - m for m, h in zip(means, highs)]
    ax.barh(companies_list, means, color='#4a90d9', alpha=0.6, edgecolor='#0e1117')
    ax.errorbar(means, companies_list, xerr=[err_low, err_high],
                fmt='none', color='#ffd700', capsize=6, linewidth=2)
    ax.axvline(x=np.mean(means), color='#64ffda', linestyle='--',
               linewidth=1.5, label=f'Mean: {np.mean(means):.1f}%')
    ax.set_xlabel('Success Rate (%)', color='#8892b0')
    ax.tick_params(colors='#8892b0'); ax.spines[:].set_color('#2e3554')
    ax.legend(facecolor='#1e2130', labelcolor='#ccd6f6')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Probability distribution chart
    st.markdown('<div class="section-header">Predicted Probability Distribution (Logistic Regression)</div>', unsafe_allow_html=True)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='#0e1117')
    ax.set_facecolor('#1e2130')
    ax.hist(y_pred_prob[y_test == 1], bins=30, alpha=0.65,
            color='#64ffda', label='Actual Success', density=True)
    ax.hist(y_pred_prob[y_test == 0], bins=30, alpha=0.65,
            color='#ff6b6b', label='Actual Failure', density=True)
    ax.set_xlabel('Predicted Probability of Success', color='#8892b0')
    ax.set_ylabel('Density', color='#8892b0')
    ax.tick_params(colors='#8892b0'); ax.spines[:].set_color('#2e3554')
    ax.legend(facecolor='#1e2130', labelcolor='#ccd6f6')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Descriptive stats
    st.markdown('<div class="section-header">Descriptive Statistics — Mission Cost (Million USD)</div>', unsafe_allow_html=True)
    cost_stats = filtered['cost_million_usd'].describe().round(2)
    cost_df = pd.DataFrame({'Statistic': cost_stats.index, 'Value (Million USD)': cost_stats.values})
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(cost_df, use_container_width=True, hide_index=True)
    with col2:
        fig, ax = plt.subplots(figsize=(7, 3.5), facecolor='#0e1117')
        ax.set_facecolor('#1e2130')
        cost_data = filtered['cost_million_usd'].dropna()
        cost_data = cost_data[cost_data < cost_data.quantile(0.95)]
        ax.hist(cost_data, bins=40, color='#a78bfa', edgecolor='#0e1117', alpha=0.8)
        ax.set_xlabel('Cost (Million USD)', color='#8892b0')
        ax.set_ylabel('Frequency', color='#8892b0')
        ax.set_title('Mission Cost Distribution', color='#ccd6f6')
        ax.tick_params(colors='#8892b0'); ax.spines[:].set_color('#2e3554')
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ============================================================
# TAB 4 — MISSION PREDICTOR
# ============================================================
with tab4:
    st.markdown("# 🎯 Mission Success Predictor")
    st.markdown("Enter the details of a future or hypothetical rocket launch to predict its success probability.")

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Enter Launch Details")

        launch_year = st.slider(
            "Launch Year",
            min_value=1957, max_value=2050, value=2025,
            help="Select the year of the mission"
        )

        all_companies = sorted(df['company'].unique().tolist())
        extra = ["Future Company (unknown)", "New Private Company", "Government Agency (generic)"]
        company_options = all_companies + extra

        selected_company = st.selectbox(
            "Space Agency / Company",
            options=company_options,
            index=company_options.index('SpaceX') if 'SpaceX' in company_options else 0,
            help="Select the organization conducting the launch"
        )

        rocket_status = st.radio(
            "Rocket Status",
            options=["Active (new/current rocket)", "Retired (old rocket model)"],
            index=0,
            help="Is the rocket currently in service or a retired model?"
        )

        orbit_type = st.selectbox(
            "Target Orbit (for reference)",
            ["Low Earth Orbit (LEO)", "Geostationary (GEO)", "Geostationary Transfer (GTO)",
             "Moon", "Mars", "Sun-Synchronous (SSO)", "Polar Orbit", "Deep Space"],
            help="This is shown for context. The model uses year, company and rocket status."
        )

        payload_note = st.text_input(
            "Mission / Payload Description (optional)",
            placeholder="e.g. Communication satellite, Crewed mission to ISS..."
        )

        predict_btn = st.button("🚀 Predict Mission Success", use_container_width=True)

    with col2:
        st.markdown("### Prediction Result")

        if predict_btn:
            # Encode inputs
            rocket_active = 1 if "Active" in rocket_status else 0

            if selected_company in le.classes_:
                company_enc = le.transform([selected_company])[0]
            else:
                company_enc = int(np.median(le.transform(le.classes_)))

            features = np.array([[launch_year, company_enc, rocket_active]])
            prob = model.predict_proba(features)[0][1] * 100

            # Color based on probability
            if prob >= 90:
                color = "#64ffda"; verdict = "HIGH SUCCESS PROBABILITY"; emoji = "✅"
            elif prob >= 75:
                color = "#ffd700"; verdict = "MODERATE SUCCESS PROBABILITY"; emoji = "⚠️"
            else:
                color = "#ff6b6b"; verdict = "LOW SUCCESS PROBABILITY"; emoji = "❌"

            st.markdown(f"""
            <div class="predict-card">
                <div class="prob-label">Predicted Probability of Success</div>
                <div class="prob-value" style="color:{color}">{prob:.1f}%</div>
                <div style="font-size:18px;color:{color};font-weight:600">{emoji} {verdict}</div>
                <div style="margin-top:20px;color:#8892b0;font-size:13px">
                    Company: <b style="color:#ccd6f6">{selected_company}</b> &nbsp;|&nbsp;
                    Year: <b style="color:#ccd6f6">{launch_year}</b> &nbsp;|&nbsp;
                    Rocket: <b style="color:#ccd6f6">{"Active" if rocket_active else "Retired"}</b>
                </div>
                {"<div style='margin-top:10px;color:#8892b0;font-size:12px'>Orbit: " + orbit_type + "</div>" if orbit_type else ""}
                {"<div style='margin-top:6px;color:#8892b0;font-size:12px'>Mission: " + payload_note + "</div>" if payload_note else ""}
            </div>
            """, unsafe_allow_html=True)

            # Probability bar
            st.markdown("#### Confidence Breakdown")
            st.progress(int(prob))
            st.markdown(f"The model gives this launch a **{prob:.1f}%** chance of success based on historical patterns from 4,198 launches (1957–2020).")

            # Historical comparison
            st.markdown("#### How does this compare historically?")
            if selected_company in df['company'].values:
                hist = df[df['company'] == selected_company]
                hist_rate = hist['success'].mean() * 100
                hist_count = len(hist)
                st.markdown(f"""
                - **{selected_company}** has historically launched **{hist_count}** missions
                - Historical success rate: **{hist_rate:.1f}%**
                - Model predicted: **{prob:.1f}%**
                - Difference: **{abs(prob - hist_rate):.1f}%** (model adjusts for year and rocket status)
                """)
            else:
                st.markdown("No historical data available for this company — prediction based on overall dataset patterns.")

        else:
            st.markdown("""
            <div class="predict-card" style="padding:60px 30px">
                <div style="font-size:60px">🛸</div>
                <div style="color:#8892b0;font-size:16px;margin-top:16px">
                    Fill in the launch details on the left<br>and click <b style="color:#64ffda">Predict Mission Success</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Historical comparison chart at the bottom
    st.markdown("---")
    st.markdown('<div class="section-header">Future Launch Trend Forecast (2020–2050)</div>', unsafe_allow_html=True)
    st.markdown("Projected success rates based on the historical improvement trend.")

    fig, ax = plt.subplots(figsize=(12, 4), facecolor='#0e1117')
    ax.set_facecolor('#1e2130')

    # Historical
    decade_rate = df.groupby('decade')['success'].mean() * 100
    hist_years = decade_rate.index.astype(int).tolist()
    hist_vals  = decade_rate.values.tolist()
    ax.plot(hist_years, hist_vals, marker='o', color='#64ffda',
            linewidth=2.5, markersize=8, label='Historical success rate')

    # Forecast (linear extrapolation)
    from numpy.polynomial import polynomial as P
    coeffs = np.polyfit(hist_years, hist_vals, 1)
    future_years = list(range(2030, 2055, 5))
    future_vals  = [min(99.5, np.polyval(coeffs, y)) for y in future_years]
    ax.plot(future_years, future_vals, marker='s', color='#ffd700',
            linewidth=2, markersize=8, linestyle='--', label='Projected success rate')
    ax.fill_between(future_years,
                    [max(70, v-3) for v in future_vals],
                    [min(100, v+3) for v in future_vals],
                    alpha=0.15, color='#ffd700')
    ax.axvline(x=2020, color='#ff6b6b', linestyle=':', linewidth=1.5, label='Data cutoff (2020)')

    ax.set_xlabel('Year', color='#8892b0'); ax.set_ylabel('Success Rate (%)', color='#8892b0')
    ax.tick_params(colors='#8892b0'); ax.spines[:].set_color('#2e3554')
    ax.set_ylim(40, 105)
    ax.legend(facecolor='#1e2130', labelcolor='#ccd6f6')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("""
    <div class="info-box">
    <b>Note:</b> The future trend is extrapolated using linear regression on historical decade-wise success rates.
    Real future rates will depend on new technologies, number of new entrants, and mission complexity.
    The shaded region represents a ±3% uncertainty band around the projection.
    </div>
    """, unsafe_allow_html=True)
