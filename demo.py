import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io

# Interactive charts
import plotly.express as px
import plotly.graph_objects as go

# --------------------------
# Page & Style
# --------------------------
st.set_page_config(
    page_title="Smart Investment Dashboard",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle CSS polish
# --------------------------
# Subtle CSS polish with light & dark theme
# --------------------------
st.markdown("""            
<style>
/* Hide Streamlit default toolbar */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Shared card-like container */
.block-container {padding-top: 1.2rem;}
.metric-card {
    border-radius: 16px;
    padding: 16px; 
    box-shadow: 0 1px 8px rgba(0,0,0,0.06);
}

/* Light theme */
[data-theme="light"] .metric-card {
    background: white;
    border: 1px solid #eee;
}
[data-theme="light"] div[data-testid="stMetric"] {
    background: linear-gradient(180deg, #ffffff 0%, #f9fafb 100%);
    border-radius: 14px; 
    padding: 10px 12px; 
    border: 1px solid #eef2f7;
    color: #111;
}
[data-theme="light"] .badge {
    background: #eef2ff; color: #4338ca;
    border: 1px solid #c7d2fe;
}

/* Dark theme */
[data-theme="dark"] .metric-card {
    background: #1e1e1e;
    border: 1px solid #333;
}
[data-theme="dark"] div[data-testid="stMetric"] {
    background: linear-gradient(180deg, #2b2b2b 0%, #1a1a1a 100%);
    border-radius: 14px; 
    padding: 10px 12px; 
    border: 1px solid #333;
    color: #eee;
}
[data-theme="dark"] .badge {
    background: #2d2a4a; color: #c7d2fe;
    border: 1px solid #4c4a6a;
}
</style>
""", unsafe_allow_html=True)


theme_mode = st.sidebar.radio("Theme Mode", ["Auto", "Light", "Dark"])

st.markdown(f"""
<script>
var root = window.parent.document.documentElement;
if ("{theme_mode}" === "Light") {{
    root.setAttribute("data-theme", "light");
}} else if ("{theme_mode}" === "Dark") {{
    root.setAttribute("data-theme", "dark");
}} else {{
    root.removeAttribute("data-theme");
}}
</script>
""", unsafe_allow_html=True)


# --------------------------
# Base Investment Data
# --------------------------
BASE_INVESTMENT_DATA = [
    {'type': 'Equity Stocks',     'risk': 'High',   'expected_return': 12.0, 'min_period': 3,  'tax_benefit': False, 'liquidity': 'High'},
    {'type': 'SIP (Mutual Funds)','risk': 'Medium', 'expected_return': 10.0, 'min_period': 1,  'tax_benefit': True,  'liquidity': 'High'},
    {'type': 'PPF',               'risk': 'Low',    'expected_return': 7.5,  'min_period': 15, 'tax_benefit': True,  'liquidity': 'Low'},
    {'type': 'Fixed Deposits',    'risk': 'Low',    'expected_return': 5.5,  'min_period': 1,  'tax_benefit': False, 'liquidity': 'Medium'},
    {'type': 'ELSS',              'risk': 'Medium', 'expected_return': 11.0, 'min_period': 3,  'tax_benefit': True,  'liquidity': 'Low'},
    {'type': 'Gold ETF',          'risk': 'Medium', 'expected_return': 8.0,  'min_period': 2,  'tax_benefit': False, 'liquidity': 'High'},
    {'type': 'Real Estate',       'risk': 'High',   'expected_return': 9.0,  'min_period': 5,  'tax_benefit': False, 'liquidity': 'Low'},
    {'type': 'Corporate Bonds',   'risk': 'Low',    'expected_return': 6.5,  'min_period': 2,  'tax_benefit': False, 'liquidity': 'Medium'},
]
DF_INV = pd.DataFrame(BASE_INVESTMENT_DATA)

RISK_MAP = {'Low': 1, 'Medium': 2, 'High': 3}

# --------------------------
# Helpers
# --------------------------
def format_currency(amount: float) -> str:
    if amount >= 1e7:   # 1 Cr
        return f"₹ {amount/1e7:.2f} Cr"
    if amount >= 1e5:   # 1 Lakh
        return f"₹ {amount/1e5:.2f} L"
    if amount >= 1e3:
        return f"₹ {amount/1e3:.0f} K"
    return f"₹ {amount:,.0f}"

@st.cache_data(show_spinner=False)
def calculate_compound(principal: float, rate_pct: float, years: int) -> float:
    return principal * (1 + rate_pct/100.0) ** years

@st.cache_data(show_spinner=False)
def calculate_recommendations(df_investments: pd.DataFrame, user_params: dict) -> list:
    amount = user_params['amount']
    period = user_params['period']
    risk_profile = user_params['risk_profile'].lower()  # conservative/moderate/aggressive
    age_group = user_params['age_group']                # '20-30', etc
    goal = user_params['investment_goal'].lower()       # wealth-creation, retirement, etc.

    recs = []
    for _, inv in df_investments.iterrows():
        score = 50
        suit = []

        # Risk match
        if risk_profile == 'conservative' and inv['risk'] == 'Low':
            score += 30; suit.append('Matches conservative risk profile')
        elif risk_profile == 'moderate' and inv['risk'] == 'Medium':
            score += 25; suit.append('Aligns with moderate risk appetite')
        elif risk_profile == 'aggressive' and inv['risk'] == 'High':
            score += 20; suit.append('Suitable for aggressive growth')

        # Period match
        if period >= inv['min_period']:
            score += 15; suit.append(f'Min period {inv["min_period"]}y satisfied')
        else:
            score -= 20; suit.append(f'Needs at least {inv["min_period"]}y')

        # Age hint
        age_num = int(age_group.split('-')[0]) if '-' in age_group else 60
        if age_num < 35 and inv['risk'] == 'High':
            score += 10; suit.append('Young investor can absorb risk')
        elif age_num > 50 and inv['risk'] == 'Low':
            score += 15; suit.append('Suitable closer to retirement')

        # Goal
        if goal == 'tax-saving' and inv['tax_benefit']:
            score += 25; suit.append('Eligible for 80C tax benefit')
        elif goal == 'wealth-creation' and inv['expected_return'] > 10:
            score += 20; suit.append('Higher growth potential')
        elif goal == 'retirement' and 'PPF' in inv['type']:
            score += 30; suit.append('Ideal for long-term retirement')
        elif goal == 'emergency-fund' and inv['liquidity'] == 'High':
            score += 15; suit.append('High liquidity for emergencies')

        maturity = calculate_compound(amount, inv['expected_return'], period)
        gains = maturity - amount
        annual_gains = gains / max(period, 1)

        recs.append({
            'type': inv['type'],
            'risk': inv['risk'],
            'expected_return': float(inv['expected_return']),
            'min_period': int(inv['min_period']),
            'tax_benefit': bool(inv['tax_benefit']),
            'liquidity': inv['liquidity'],
            'score': float(min(score, 100)),
            'maturity_amount': float(maturity),
            'total_gains': float(gains),
            'annual_gains': float(annual_gains),
            'suitability_factors': suit
        })
    recs.sort(key=lambda x: x['score'], reverse=True)
    return recs

def portfolio_risk_label(top):
    vals = [RISK_MAP[r['risk']] for r in top]
    avg = sum(vals)/len(vals)
    if avg <= 1.5: return "Conservative"
    if avg <= 2.5: return "Moderate"
    return "Aggressive"

def sip_maturity(monthly_amount, annual_return_pct, years):
    n = years * 12
    r = (annual_return_pct/100.0)/12.0
    if r == 0: return monthly_amount * n
    return monthly_amount * (((1 + r) ** n - 1) / r) * (1 + r)

def monte_carlo(principal, annual_return, years, simulations=1000, vol=3.0, seed=42):
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(simulations):
        amount = principal
        for _y in range(years):
            drift = rng.uniform(-vol, vol)
            amount *= (1 + (annual_return + drift)/100.0)
        results.append(amount)
    arr = np.array(results)
    return {
        'min': float(arr.min()),
        'max': float(arr.max()),
        'average': float(arr.mean()),
        'median': float(np.median(arr)),
        'p10': float(np.percentile(arr, 10)),
        'p90': float(np.percentile(arr, 90))
    }

def inflation_adjusted(amount, annual_return, years, inflation_rate=6.0):
    nominal = calculate_compound(amount, annual_return, years)
    real = nominal / ((1 + inflation_rate/100.0) ** years)
    real_rate = ((real/amount) ** (1/years) - 1) * 100.0 if years > 0 else 0
    return {
        'nominal_amount': float(nominal),
        'real_amount': float(real),
        'inflation_impact': float(nominal - real),
        'real_return_rate': float(real_rate)
    }

def to_df(recs: list) -> pd.DataFrame:
    df = pd.DataFrame(recs)
    df['risk_num'] = df['risk'].map(RISK_MAP)
    return df

def download_excel(df: pd.DataFrame, filename="recommendations.xlsx"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Recommendations', index=False)
    return output.getvalue()

# --------------------------
# Sidebar Controls
# --------------------------
st.sidebar.title("🔧 Controls")

uploaded = st.sidebar.file_uploader("Optional: Upload custom investment dataset (Excel/CSV)", type=['xlsx', 'xls', 'csv'])
if uploaded is not None:
    if uploaded.name.lower().endswith('.csv'):
        DF_CUSTOM = pd.read_csv(uploaded)
    else:
        DF_CUSTOM = pd.read_excel(uploaded)
    # Validate minimal fields, else fallback to base
    required = {'type','risk','expected_return','min_period','tax_benefit','liquidity'}
    if required.issubset(set(map(str.lower, DF_CUSTOM.columns.str.lower()))):
        # Normalize column names
        DF_CUSTOM.columns = DF_CUSTOM.columns.str.lower()
        DF_CUSTOM = DF_CUSTOM.rename(columns={
            'expected_return':'expected_return',
            'min_period':'min_period',
            'tax_benefit':'tax_benefit'
        })
        # Reorder/rename to consistent schema
        def pick(col): # pick with case-insensitive
            cols = [c for c in DF_CUSTOM.columns if c.lower()==col]
            return cols[0] if cols else None
        DF_INV_USED = DF_CUSTOM.rename(columns={
            pick('type'): 'type',
            pick('risk'): 'risk',
            pick('liquidity'): 'liquidity',
        })[['type','risk','expected_return','min_period','tax_benefit','liquidity']].copy()
    else:
        st.sidebar.warning("Uploaded file missing required columns. Using default dataset.")
        DF_INV_USED = DF_INV.copy()
else:
    DF_INV_USED = DF_INV.copy()

amount = st.sidebar.number_input("💰 Investment Amount (₹)", min_value=1000, value=100000, step=5000)
period = st.sidebar.slider("⏳ Investment Period (Years)", 1, 40, 10)
risk_profile = st.sidebar.selectbox("⚖️ Risk Profile", ['Conservative','Moderate','Aggressive'])
age_group = st.sidebar.selectbox("👤 Age Group", ['20-30','30-40','40-50','50-60','60+'])
goal_map = {
    'Wealth Creation':'wealth-creation',
    'Retirement':'retirement',
    'Tax Saving':'tax-saving',
    'Emergency Fund':'emergency-fund',
    'Child Education':'child-education'
}
goal_display = st.sidebar.selectbox("🎯 Investment Goal", list(goal_map.keys()))
goal_key = goal_map[goal_display]

user_params = {
    'amount': amount,
    'period': period,
    'risk_profile': risk_profile.lower(),
    'age_group': age_group,
    'investment_goal': goal_key
}

# --------------------------
# Header
# --------------------------
st.title("💹 Smart Investment Recommendation Dashboard")
st.caption("Personalized suggestions, interactive analysis, and exportable reports.")

# Compute recommendations
recs = calculate_recommendations(DF_INV_USED, user_params)
df_recs = to_df(recs)

# --------------------------
# KPI Row
# --------------------------
top3 = recs[:3]
avg_return_top3 = np.mean([r['expected_return'] for r in top3]) if top3 else 0
portfolio_risk = portfolio_risk_label(top3) if top3 else "—"
best = recs[0] if recs else None

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Amount", format_currency(amount))
with c2:
    st.metric("Period", f"{period} years")
with c3:
    st.metric("Avg Exp. Return (Top 3)", f"{avg_return_top3:.1f}%")
with c4:
    st.metric("Portfolio Risk (Top 3)", portfolio_risk)

# --------------------------
# Tabs
# --------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Recommendations",
    "📊 Visual Analysis",
    "📈 SIP & Growth",
    "🎲 Monte-Carlo & Inflation",
    "📥 Export & Table"
])

# --------------------------
# Tab 1: Recommendations
# --------------------------
with tab1:
    st.subheader("Top Matches")

    if not recs:
        st.info("No recommendations available.")
    else:
        # Show top 6 as cards
        grid = st.columns(3)
        for i, rec in enumerate(recs[:6]):
            col = grid[i % 3]
            with col:
                st.markdown(f"<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"**{i+1}. {rec['type']}**  &nbsp; <span class='badge'>{rec['risk']} Risk</span>", unsafe_allow_html=True)
                st.write(f"**Score:** {rec['score']:.0f}%")
                st.write(f"**Expected Return:** {rec['expected_return']}%")
                st.write(f"**Min Period:** {rec['min_period']} years | **Liquidity:** {rec['liquidity']}")
                st.write(f"**Tax Benefit:** {'Yes (80C)' if rec['tax_benefit'] else 'No'}")
                st.write(f"**Maturity:** {format_currency(rec['maturity_amount'])}")
                st.write(f"**Total Gains:** {format_currency(rec['total_gains'])}")
                if rec['suitability_factors']:
                    st.caption("Why it fits: " + " • ".join(rec['suitability_factors'][:3]))
                st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("Suggested Allocation (Top 3)")
        if len(top3) >= 3:
            allocations = [50, 30, 20]
        else:
            # Spread evenly if fewer than 3 options
            allocations = [int(100/len(top3))]*len(top3) if top3 else []

        if top3:
            alloc_df = pd.DataFrame({
                'Investment': [r['type'] for r in top3],
                'Allocation %': allocations[:len(top3)],
                'Amount (₹)': [amount * a/100 for a in allocations[:len(top3)]]
            })
            fig_alloc = px.pie(
                alloc_df, names='Investment', values='Allocation %',
                hole=0.45, title="Suggested Portfolio Split"
            )
            fig_alloc.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_alloc, use_container_width=True)
            st.dataframe(alloc_df, use_container_width=True)
        else:
            st.info("Insufficient options for allocation.")

# --------------------------
# Tab 2: Visual Analysis
# --------------------------
with tab2:
    st.subheader("Comparative Charts")

    if not recs:
        st.info("No data to visualize.")
    else:
        top_n = st.slider("Show Top N", 3, min(10, len(recs)), 5)
        df_top = df_recs.head(top_n).copy()

        colA, colB = st.columns(2)
        with colA:
            fig1 = px.bar(
                df_top, x='type', y='expected_return',
                title="Expected Annual Returns (%)",
            )
            fig1.update_layout(xaxis_title="", yaxis_title="Return (%)")
            st.plotly_chart(fig1, use_container_width=True)

        with colB:
            fig2 = px.bar(
                df_top, x='type', y=(df_top['maturity_amount']/1e5),
                title="Projected Maturity (₹ Lakhs)"
            )
            fig2.update_layout(xaxis_title="", yaxis_title="Lakhs")
            st.plotly_chart(fig2, use_container_width=True)

        colC, colD = st.columns(2)
        with colC:
            # Risk vs Return bubble, bubble size = score
            fig3 = px.scatter(
                df_top,
                x='risk_num', y='expected_return', size='score',
                color='type', hover_data=['liquidity','tax_benefit','min_period'],
                title="Risk vs Return (Bubble = Score)"
            )
            fig3.update_layout(
                xaxis=dict(tickmode='array', tickvals=[1,2,3], ticktext=['Low','Medium','High']),
                xaxis_title="Risk Level", yaxis_title="Expected Return (%)"
            )
            st.plotly_chart(fig3, use_container_width=True)

        with colD:
            # Radar/polar: score vs return vs liquidity (quantify liquidity)
            liquidity_map = {'Low': 1, 'Medium': 2, 'High': 3}
            df_top['liquidity_num'] = df_top['liquidity'].map(liquidity_map).fillna(2)
            # Normalize for radar
            def norm(series): 
                if series.max()==series.min(): 
                    return series*0 + 50
                return 100*(series - series.min())/(series.max()-series.min())
            radar_df = pd.DataFrame({
                'type': df_top['type'],
                'Score (norm)': norm(df_top['score']),
                'Return (norm)': norm(df_top['expected_return']),
                'Liquidity (norm)': norm(df_top['liquidity_num']),
            })
            categories = ['Score (norm)','Return (norm)','Liquidity (norm)']
            fig4 = go.Figure()
            for _, r in radar_df.iterrows():
                fig4.add_trace(go.Scatterpolar(
                    r=[r[c] for c in categories],
                    theta=categories,
                    fill='toself',
                    name=r['type']
                ))
            fig4.update_layout(title="Multi-Factor Radar (Normalized)", polar=dict(radialaxis=dict(visible=True, range=[0,100])))
            st.plotly_chart(fig4, use_container_width=True)

# --------------------------
# Tab 3: SIP & Growth
# --------------------------
with tab3:
    st.subheader("SIP Calculator & Growth Projection")

    if not recs:
        st.info("No recommendations available.")
    else:
        monthly = st.number_input("Monthly SIP (₹)", min_value=500, value=5000, step=500)
        pick_for_sip = st.selectbox("Choose investment for SIP projection", [r['type'] for r in recs[:5]])
        sel = next((r for r in recs if r['type']==pick_for_sip), recs[0])

        n_months = period * 12
        maturity_sip = sip_maturity(monthly, sel['expected_return'], period)
        invested = monthly * n_months
        gains = maturity_sip - invested

        k1,k2,k3 = st.columns(3)
        k1.metric("Total Invested", format_currency(invested))
        k2.metric("SIP Maturity", format_currency(maturity_sip))
        k3.metric("Total Gains", format_currency(gains))

        # Growth curve for top 3
        st.markdown("#### Growth over Time (Lumpsum)")
        years = list(range(period+1))
        fig_growth = go.Figure()
        for r in recs[:3]:
            vals = [calculate_compound(amount, r['expected_return'], y)/1e5 for y in years]  # Lakhs
            fig_growth.add_trace(go.Scatter(x=years, y=vals, mode='lines+markers', name=r['type']))
        fig_growth.update_layout(xaxis_title="Years", yaxis_title="Value (₹ Lakhs)")
        st.plotly_chart(fig_growth, use_container_width=True)

# --------------------------
# Tab 4: Monte-Carlo & Inflation
# --------------------------
with tab4:
    st.subheader("Risk Simulation & Real Returns")

    if not recs:
        st.info("No recommendations available.")
    else:
        pick = st.selectbox("Select investment for simulation", [r['type'] for r in recs[:5]], key="sim_pick")
        sel = next((r for r in recs if r['type']==pick), recs[0])

        col_sim = st.columns(3)
        vol = col_sim[0].slider("Volatility (±%)", 1.0, 8.0, 3.0, step=0.5)
        sims = col_sim[1].slider("Simulations", 200, 5000, 1000, step=200)
        infl = col_sim[2].slider("Inflation (%)", 2.0, 10.0, 6.0, step=0.5)

        stats = monte_carlo(amount, sel['expected_return'], period, simulations=sims, vol=vol)
        colA, colB, colC, colD, colE = st.columns(5)
        colA.metric("Min", format_currency(stats['min']))
        colB.metric("P10", format_currency(stats['p10']))
        colC.metric("Median", format_currency(stats['median']))
        colD.metric("P90", format_currency(stats['p90']))
        colE.metric("Max", format_currency(stats['max']))

        real = inflation_adjusted(amount, sel['expected_return'], period, inflation_rate=infl)
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Nominal Amount", format_currency(real['nominal_amount']))
        r2.metric("Real Amount", format_currency(real['real_amount']))
        r3.metric("Inflation Impact", format_currency(real['inflation_impact']))
        r4.metric("Real CAGR", f"{real['real_return_rate']:.2f}%")

# --------------------------
# Tab 5: Export & Table
# --------------------------
with tab5:
    st.subheader("Detailed Table & Export")

    if not recs:
        st.info("No data to export.")
    else:
        show_cols = [
            'type','risk','expected_return','min_period','tax_benefit','liquidity',
            'score','maturity_amount','total_gains','annual_gains'
        ]
        df_show = df_recs[show_cols].copy()
        df_show.rename(columns={
            'type':'Investment Type', 'risk':'Risk', 'expected_return':'Expected Return (%)',
            'min_period':'Min Period (y)', 'tax_benefit':'Tax Benefit',
            'liquidity':'Liquidity', 'score':'Match Score (%)',
            'maturity_amount':'Maturity (₹)', 'total_gains':'Total Gains (₹)',
            'annual_gains':'Annual Gains (₹)'
        }, inplace=True)
        st.dataframe(df_show, use_container_width=True)

        cdl, cdr = st.columns(2)
        with cdl:
            csv = df_show.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download CSV", csv, file_name="investment_recommendations.csv", mime="text/csv")
        with cdr:
            xlsx = download_excel(df_show, filename="investment_recommendations.xlsx")
            st.download_button("⬇️ Download Excel", xlsx, file_name="investment_recommendations.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --------------------------
# Footer note
# --------------------------
st.caption("Disclaimer: This dashboard is for educational purposes. Past performance does not guarantee future returns. Consult a financial advisor for personalized advice.")
