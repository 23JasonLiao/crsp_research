import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import skew, kurtosis

# --- 1. 網頁配置與標題 ---
st.set_page_config(layout="wide", page_title="平衡型基金：基金家族與經理人績效分析儀表板")

st.title("👨‍💼 專業資產管理：基金家族規模、Flow、經理人績效與研究邊界分析")
st.markdown("""
本系統依據研究計畫中「專家資料強化 AI 投資決策」的精神，將基金經理人的歷史投資行為、基金家族規模、費用率、資金流、風險控管與投組特徵轉換為可視化分析指標。
新增的 **Tab5** 專門回應老師要求：
1. **第一層**：先將所有基金公司 / 基金家族的資產規模加總，形成基金家族規模排序。
2. **第二層**：再檢查基金家族過去 1 年、3 年、5 年的平均報酬率、平均費用率、平均 Flow、管理年資、風險控管與投組情況，並將報酬率與 S&P 500 benchmark 比較。
3. 通過兩層條件後，用互動式研究邊界找出經理人績效較好，究竟較可能來自大公司、低費用、高 Flow、長年資，或較佳風險控管。
""")

# --- 2. 資料讀取與處理函數 ---
@st.cache_data
def load_and_combine_data(uploaded_files):
    df_list = []
    for file in uploaded_files:
        try:
            temp_df = pd.read_csv(file, low_memory=False)
            df_list.append(temp_df)
        except Exception as e:
            st.error(f"讀取檔案 {file.name} 時發生錯誤: {e}")
    if not df_list:
        return None

    df = pd.concat(df_list, ignore_index=True)

    # 基礎欄位轉型
    for c in ['caldt', 'mgr_dt', 'first_offer_dt', 'flow_report_dt', 'trans_dt']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    for c in ['mret', 'mtna', 'exp_ratio', 'mgmt_fee', 'turn_ratio', 'new_sls', 'rein_sls', 'oth_sls', 'redemp', 'age']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    if 'mgr_dt' in df.columns:
        df['mgr_dt'] = df['mgr_dt'].fillna(df['caldt'])
    else:
        df['mgr_dt'] = df['caldt']

    if 'mgmt_name' not in df.columns and 'mgr_name' in df.columns:
        df['mgmt_name'] = df['mgr_name']
    if 'mgr_name' not in df.columns:
        df['mgr_name'] = 'Unknown Manager'

    # Flow：老師要求分析過去 flow，因此直接建立 net_flow
    if {'new_sls', 'redemp'}.issubset(df.columns):
        df['net_flow'] = df['new_sls'].fillna(0) + df.get('rein_sls', 0).fillna(0) + df.get('oth_sls', 0).fillna(0) - df['redemp'].fillna(0)
    elif 'net_flow' not in df.columns:
        df['net_flow'] = np.nan

    # 管理年資
    df['tenure'] = (df['caldt'] - df['mgr_dt']).dt.days / 365.25
    df['tenure'] = df['tenure'].clip(lower=0)
    df['seniority_label'] = (df['tenure'] >= 10).map({True: '資深 (10年以上)', False: '一般資歷'})

    return df.dropna(subset=['mret', 'caldt', 'mgmt_name'])

@st.cache_data
def load_sp500_benchmark(uploaded_file):
    """讀取 S&P500 benchmark CSV。接受欄位：caldt/date/month 與 sp500_ret/ret/mret/return。"""
    if uploaded_file is None:
        return None
    b = pd.read_csv(uploaded_file, low_memory=False)
    date_col = next((c for c in ['caldt', 'date', 'month', 'Date', 'DATE'] if c in b.columns), None)
    ret_col = next((c for c in ['sp500_ret', 'sp500_mret', 'mret', 'ret', 'return', 'Return'] if c in b.columns), None)
    if date_col is None or ret_col is None:
        st.warning("S&P500 CSV 需要日期欄位 caldt/date/month，以及報酬欄位 sp500_ret/ret/mret/return。已改用手動 benchmark。")
        return None
    b = b[[date_col, ret_col]].copy()
    b.columns = ['caldt', 'sp500_ret']
    b['caldt'] = pd.to_datetime(b['caldt'], errors='coerce')
    b['sp500_ret'] = pd.to_numeric(b['sp500_ret'], errors='coerce')
    return b.dropna(subset=['caldt', 'sp500_ret'])

# --- 3. 市場環境與既有指標 ---
def detect_market_regime(df):
    market_monthly = df.groupby('caldt')['mret'].agg(['mean', 'std']).reset_index()
    vol_threshold = market_monthly['std'].median()
    def label_regime(row):
        if row['mean'] < 0:
            return "市場低迷 (股債雙殺)"
        if row['std'] > vol_threshold:
            return "股票動能強 (高波動擴張)"
        return "債券/穩健強 (低波動避險)"
    market_monthly['市場環境'] = market_monthly.apply(label_regime, axis=1)
    return market_monthly[['caldt', '市場環境', 'mean']]

def calculate_drawdown_series(df):
    drawdown_list = []
    for name, group in df.sort_values('caldt').groupby('mgmt_name'):
        group = group.copy()
        group['wealth_index'] = (1 + group['mret']).cumprod()
        group['previous_peaks'] = group['wealth_index'].cummax()
        group['drawdown'] = (group['wealth_index'] - group['previous_peaks']) / group['previous_peaks']
        drawdown_list.append(group)
    return pd.concat(drawdown_list) if drawdown_list else pd.DataFrame()

def calculate_asset_management_factors(df):
    results = []
    for name, group in df.groupby('mgmt_name'):
        mrets = group['mret'].dropna()
        if len(mrets) < 6:
            continue
        ann_ret = (1 + mrets.mean())**12 - 1
        ann_vol = mrets.std() * np.sqrt(12)
        sharpe = (ann_ret - 0.01) / ann_vol if ann_vol > 0 else 0
        downside_rets = mrets[mrets < 0]
        downside_vol = downside_rets.std() * np.sqrt(12) if len(downside_rets) > 0 else 0
        sortino = (ann_ret - 0.01) / downside_vol if downside_vol > 0 else 0
        cum_ret = (1 + mrets).cumprod()
        running_max = cum_ret.cummax()
        mdd = ((cum_ret - running_max) / running_max).min()
        results.append({
            '管理公司': name,
            '年化報酬率': ann_ret,
            '年化波動度': ann_vol,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': mdd,
            '偏度 (Skewness)': round(skew(mrets), 2),
            '峰度 (Kurtosis)': round(kurtosis(mrets), 2),
            'VaR 95% (月度)': f"{np.percentile(mrets, 5):.2%}"
        })
    return pd.DataFrame(results)

def estimate_allocation(row):
    lipper = str(row.get('lipper_class_name', '')).upper()
    policy = str(row.get('policy', '')).upper()
    text = lipper + ' ' + policy
    if 'GROWTH' in text or '70% TO 90%' in text:
        return "80:20 (激進型)"
    if 'MODERATE' in text or '50% TO 70%' in text:
        return "60:40 (平衡型)"
    if 'CONSERVATIVE' in text or '30% TO 50%' in text:
        return "40:60 (保守型)"
    return "60:40 (標準平衡)"

def render_mgmt_treemap(df):
    latest_df = df.sort_values('caldt').groupby('crsp_fundno').last().reset_index()
    latest_df['股債配置比'] = latest_df.apply(estimate_allocation, axis=1)
    mgmt_tree = latest_df.groupby(['mgmt_name', '股債配置比']).agg(
        mtna=('mtna', 'sum'), mret=('mret', 'mean'), fund_name=('fund_name', 'count')
    ).reset_index()
    fig = px.treemap(
        mgmt_tree,
        path=[px.Constant("全體平衡型基金市場"), '股債配置比', 'mgmt_name'],
        values='mtna', color='mret', color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
        hover_data=['fund_name'],
        title="資產管理版圖：規模(大小) vs. 表現(顏色) vs. 配置(層級)",
        labels={'mtna': '資產規模 (M)', 'mret': '平均月回報', 'mgmt_name': '管理公司'}
    )
    fig.update_traces(textinfo="label+value+percent parent")
    return fig

def render_dynamic_factor_matrix(df, selected_year):
    st.subheader(f"🔄 {selected_year} 年：資金流、規模與報酬之動態矩陣")
    cols = [c for c in ['mret', 'mtna', 'net_flow'] if c in df.columns]
    df_year = df[df['caldt'].dt.year == selected_year].dropna(subset=cols).copy()
    if df_year.empty:
        st.warning(f"{selected_year} 年份無足夠數據進行矩陣分析。")
        return None
    fig = px.scatter_matrix(
        df_year, dimensions=cols, color="seniority_label", opacity=0.6,
        labels={'mret': '報酬 (Mret)', 'mtna': '規模 (MTNA)', 'net_flow': '資金流 (Flow)'},
        title=f"股債平衡基金因子交互作用 ({selected_year})", height=700,
    )
    fig.update_traces(diagonal_visible=True, marker=dict(size=5))
    return fig

# --- 4. 新增：老師要求的基金家族兩層 Data Mining ---
def annualized_return_from_monthly(s):
    s = pd.to_numeric(s, errors='coerce').dropna()
    if len(s) == 0:
        return np.nan
    return (1 + s.mean())**12 - 1

def max_drawdown_from_monthly(s):
    s = pd.to_numeric(s, errors='coerce').dropna()
    if len(s) == 0:
        return np.nan
    wealth = (1 + s).cumprod()
    peak = wealth.cummax()
    return ((wealth - peak) / peak).min()

def classify_excess_return(excess_ret):
    if pd.isna(excess_ret):
        return "缺 benchmark"
    if excess_ret >= 0.20:
        return "好：超越 S&P500 20% 以上"
    if excess_ret < 0:
        return "不好：落後 S&P500"
    return "普通：有超額但未達 20%"

def add_sp500_comparison(period_df, sp500_df, manual_sp500_ann):
    """回傳該期間 S&P500 年化報酬。若有 benchmark CSV 則用實際期間資料，否則用手動值。"""
    if sp500_df is not None and not sp500_df.empty and not period_df.empty:
        min_d, max_d = period_df['caldt'].min(), period_df['caldt'].max()
        b = sp500_df[(sp500_df['caldt'] >= min_d) & (sp500_df['caldt'] <= max_d)]
        if not b.empty:
            return annualized_return_from_monthly(b['sp500_ret'])
    return manual_sp500_ann

def build_family_two_layer_table(df, years, sp500_df=None, manual_sp500_ann=0.10):
    """第一層：基金家族總規模；第二層：1/3/5 年績效、費用、flow、年資、風控、投組狀況。"""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    base = df.copy()
    base['股債配置比'] = base.apply(estimate_allocation, axis=1)
    end_date = base['caldt'].max()

    # 第一層：每檔基金取最新 mtna，再依基金家族加總，避免同一基金所有月份重複加總
    latest_fund = base.sort_values('caldt').groupby('crsp_fundno').tail(1)
    layer1 = latest_fund.groupby('mgmt_name').agg(
        基金家族總資產規模_MTNA=('mtna', 'sum'),
        旗下基金數=('crsp_fundno', 'nunique'),
        最新平均月報酬=('mret', 'mean')
    ).reset_index().rename(columns={'mgmt_name': '基金家族'})
    layer1['規模分位數'] = layer1['基金家族總資產規模_MTNA'].rank(pct=True)
    layer1['基金家族規模分類'] = pd.qcut(
        layer1['基金家族總資產規模_MTNA'].rank(method='first'),
        q=3, labels=['小型基金家族', '中型基金家族', '大型基金家族']
    )

    rows = []
    for family, g_all in base.groupby('mgmt_name'):
        for y in years:
            start_date = end_date - pd.DateOffset(years=y)
            g = g_all[g_all['caldt'] >= start_date].copy()
            if len(g) < 6:
                continue
            sp500_ann = add_sp500_comparison(g, sp500_df, manual_sp500_ann)
            ann_ret = annualized_return_from_monthly(g['mret'])
            ann_vol = g['mret'].std() * np.sqrt(12)
            mdd = max_drawdown_from_monthly(g.sort_values('caldt')['mret'])
            sharpe = (ann_ret - 0.01) / ann_vol if ann_vol and ann_vol > 0 else np.nan
            excess = ann_ret - sp500_ann if pd.notna(sp500_ann) else np.nan
            # 投組情況：以 CRSP/Lipper 分類近似股債配置，並納入換手率
            alloc_mode = g['股債配置比'].mode().iloc[0] if not g['股債配置比'].mode().empty else '未知'
            rows.append({
                '基金家族': family,
                '期間': f'過去{y}年',
                '期間年數': y,
                '平均年化報酬率': ann_ret,
                'S&P500年化報酬率': sp500_ann,
                '相對S&P500超額報酬': excess,
                '績效判定': classify_excess_return(excess),
                '平均費用率': g['exp_ratio'].mean(),
                '平均Flow': g['net_flow'].mean(),
                '累積Flow': g['net_flow'].sum(),
                '平均管理年資': g['tenure'].mean(),
                '年化波動度': ann_vol,
                'Max Drawdown': mdd,
                'Sharpe Ratio': sharpe,
                '平均換手率': g['turn_ratio'].mean() if 'turn_ratio' in g.columns else np.nan,
                '主要投組型態': alloc_mode,
                '觀測月數': g['caldt'].nunique(),
                '基金檔數': g['crsp_fundno'].nunique(),
                '經理人數': g['mgr_name'].nunique()
            })
    layer2 = pd.DataFrame(rows)
    if not layer2.empty:
        layer2 = layer2.merge(layer1[['基金家族', '基金家族總資產規模_MTNA', '基金家族規模分類']], on='基金家族', how='left')
    return layer1, layer2

def build_manager_performance_table(df, selected_years=3):
    """用於回答：基金經理人表現好是否與公司大、低費用、高 Flow、年資有關。"""
    if df.empty:
        return pd.DataFrame()
    end_date = df['caldt'].max()
    gdf = df[df['caldt'] >= end_date - pd.DateOffset(years=selected_years)].copy()
    if gdf.empty:
        return pd.DataFrame()
    latest_fund = gdf.sort_values('caldt').groupby('crsp_fundno').tail(1)
    family_size = latest_fund.groupby('mgmt_name')['mtna'].sum().rename('基金家族總資產規模_MTNA')
    out = gdf.groupby(['mgmt_name', 'mgr_name']).agg(
        平均年化報酬率=('mret', annualized_return_from_monthly),
        平均費用率=('exp_ratio', 'mean'),
        平均Flow=('net_flow', 'mean'),
        累積Flow=('net_flow', 'sum'),
        平均管理年資=('tenure', 'mean'),
        年化波動度=('mret', lambda s: s.std() * np.sqrt(12)),
        MaxDrawdown=('mret', max_drawdown_from_monthly),
        平均換手率=('turn_ratio', 'mean'),
        基金檔數=('crsp_fundno', 'nunique')
    ).reset_index().rename(columns={'mgmt_name': '基金家族', 'mgr_name': '基金經理人'})
    out = out.merge(family_size.reset_index().rename(columns={'mgmt_name': '基金家族'}), on='基金家族', how='left')
    out['Sharpe近似'] = (out['平均年化報酬率'] - 0.01) / out['年化波動度'].replace(0, np.nan)
    return out

def format_pct_columns(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: '' if pd.isna(x) else f'{x:.2%}')
    return out

# --- 5. 主程式 ---
uploaded_files = st.sidebar.file_uploader("上傳平衡型基金 CSV (可選多個)", type="csv", accept_multiple_files=True)
sp500_file = st.sidebar.file_uploader("可選：上傳 S&P500 月報酬 CSV", type="csv", accept_multiple_files=False)
manual_sp500_ann = st.sidebar.number_input("若未上傳 S&P500，手動設定 benchmark 年化報酬率", value=0.10, min_value=-1.0, max_value=2.0, step=0.01, format="%.2f")

if uploaded_files:
    df = load_and_combine_data(uploaded_files)
    sp500_df = load_sp500_benchmark(sp500_file)

    if df is not None:
        regime_df = detect_market_regime(df)
        df = df.merge(regime_df, on='caldt', how='left')

        # 側邊欄過濾
        all_mgmt = sorted(df['mgmt_name'].dropna().unique())
        selected_mgmt = st.sidebar.multiselect("選擇管理公司", options=all_mgmt)
        df_f = df.copy()
        if selected_mgmt:
            df_f = df_f[df_f['mgmt_name'].isin(selected_mgmt)]

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "💼 專業資產管理因子",
            "📈 年資與股債環境適應性",
            "📊 資金流行為與資產動態生態分析",
            "🔍 自定義交叉探索",
            "🧭 基金家族兩層篩選與研究邊界",
            "📋 原始數據"
        ])

        with tab1:
            st.header("資產管理核心指標 (MPT & Risk Factors)")
            st.info("💡 透過 Sharpe、Sortino、Max Drawdown、VaR 檢查基金家族或管理公司的風險調整後績效。")
            factor_df = calculate_asset_management_factors(df_f)
            if factor_df.empty:
                st.warning("資料不足，無法計算因子。")
            else:
                display_df = format_pct_columns(factor_df, ['年化報酬率', '年化波動度', 'Max Drawdown'])
                for col in ['Sharpe Ratio', 'Sortino Ratio', '偏度 (Skewness)', '峰度 (Kurtosis)']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].map(lambda x: '' if pd.isna(x) else f'{x:.4f}')
                st.dataframe(display_df, use_container_width=True)

                factor_df['BubbleSize'] = factor_df['Sharpe Ratio'].apply(lambda x: max(x, 0) + 0.05)
                fig_risk = px.scatter(
                    factor_df, x="年化波動度", y="年化報酬率", size="BubbleSize", color="管理公司",
                    hover_data=["Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "偏度 (Skewness)", "VaR 95% (月度)"],
                    title="風險-報酬效率前緣圖：氣泡大小代表 Sharpe Ratio"
                )
                fig_risk.update_layout(xaxis_tickformat='.1%', yaxis_tickformat='.1%')
                st.plotly_chart(fig_risk, use_container_width=True)

                st.divider()
                st.subheader("📉 歷史回撤趨勢圖 (Underwater Chart)")
                df_dd = calculate_drawdown_series(df_f)
                if not df_dd.empty:
                    fig_dd = px.line(df_dd, x="caldt", y="drawdown", color="mgmt_name", labels={'drawdown': '回撤幅度', 'caldt': '日期'}, title="基金歷史回撤路徑")
                    fig_dd.update_layout(yaxis_tickformat='.1%', yaxis_range=[df_dd['drawdown'].min()*1.1, 0])
                    fig_dd.add_hline(y=0, line_dash="dash", line_color="black")
                    st.plotly_chart(fig_dd, use_container_width=True)

        with tab2:
            st.header("經理人任期與股債市場環境分析")
            col_a, col_b = st.columns([2, 1])
            with col_a:
                fig_regime = px.scatter(df_f, x="caldt", y="mret", color="市場環境", symbol="seniority_label", title="不同市場環境下的經理人表現點位", labels={'mret': '月回報率', 'caldt': '日期'})
                st.plotly_chart(fig_regime, use_container_width=True)
            with col_b:
                regime_compare = df_f.groupby(['市場環境', 'seniority_label'])['mret'].mean().reset_index()
                fig_bar = px.bar(regime_compare, x="市場環境", y="mret", color="seniority_label", barmode="group", title="不同資歷在股/債環境的平均回報", labels={'mret': '平均月報酬'})
                st.plotly_chart(fig_bar, use_container_width=True)
            fig_trend = px.scatter(df_f, x="tenure", y="mret", color="市場環境", trendline="ols", title="經理人年資 vs 月回報", labels={'tenure': '在職年資 (年)', 'mret': '月回報率'})
            st.plotly_chart(fig_trend, use_container_width=True)

        with tab3:
            st.header("📊 資產與因子動態交互分析")
            if 'mtna' in df_f.columns:
                st.subheader("🏢 資產管理版圖 (Treemap)")
                st.plotly_chart(render_mgmt_treemap(df_f), use_container_width=True)
            st.divider()
            st.subheader("🗓️ 時間維度因子分析")
            available_years = sorted(df_f['caldt'].dt.year.dropna().unique())
            selected_year = st.slider("選擇分析年份", min_value=int(min(available_years)), max_value=int(max(available_years)), value=int(max(available_years)))
            fig_matrix = render_dynamic_factor_matrix(df_f, selected_year)
            if fig_matrix:
                st.plotly_chart(fig_matrix, use_container_width=True)
            st.divider()
            st.subheader("🌊 資金流敏感度：資深經理人更能留住資金嗎？")
            if 'net_flow' in df_f.columns and df_f['net_flow'].notna().any():
                col_chart, col_stat = st.columns([3, 2])
                with col_chart:
                    fig_flow = px.scatter(df_f, x="mret", y="net_flow", color="seniority_label", trendline="ols", title="資金流對報酬之敏感度", labels={"mret": "月報酬率", "net_flow": "淨資金流"}, opacity=0.4)
                    fig_flow.update_layout(xaxis_tickformat='.1%', hovermode=False)
                    st.plotly_chart(fig_flow, use_container_width=True)
                with col_stat:
                    flow_stats = df_f.groupby('seniority_label').agg(平均月流向=('net_flow', 'mean'), 流向波動=('net_flow', 'std'), 總淨流向=('net_flow', 'sum'), 平均回報=('mret', 'mean'))
                    st.table(flow_stats.style.format({'平均回報': '{:.2%}', '平均月流向': '{:.2f}', '總淨流向': '{:.2f}'}))
            else:
                st.warning("數據中缺少 Flow 欄位，無法分析資金流敏感度。")

        with tab4:
            st.header("🔍 自定義交叉探索沙盒：動態邊界分析")
            exp_col1, exp_col2, exp_col3 = st.columns(3)
            with exp_col1:
                senior_threshold = st.slider("資深經理人年資門檻 (年)", 1.0, 20.0, 10.0, 0.5, key='tab4_senior')
            with exp_col2:
                ret_cutoff = st.slider("市場低迷判定門檻 (月報酬 %)", -5.0, 2.0, 0.0, 0.1, key='tab4_ret') / 100
            with exp_col3:
                vol_adjustment = st.slider("市場波動判定偏移 (±%)", -50, 50, 0, 5, key='tab4_vol') / 100
            df_sandbox = df_f.copy()
            df_sandbox['seniority_label'] = df_sandbox['tenure'].apply(lambda x: f'資深 ({senior_threshold}Y+)' if x >= senior_threshold else '一般資歷')
            market_stats = df_sandbox.groupby('caldt')['mret'].agg(['mean', 'std']).reset_index()
            dynamic_vol_limit = market_stats['std'].median() * (1 + vol_adjustment)
            market_stats['市場環境'] = market_stats.apply(lambda row: "市場低迷 (雙殺)" if row['mean'] < ret_cutoff else ("股票動能強 (高波)" if row['std'] > dynamic_vol_limit else "債券/穩健強 (低波)"), axis=1)
            df_sandbox = df_sandbox.drop(columns=['市場環境'], errors='ignore').merge(market_stats[['caldt', '市場環境']], on='caldt', how='left')
            df_sandbox['股債配置比'] = df_sandbox.apply(estimate_allocation, axis=1)
            c1, c2, c3 = st.columns(3)
            with c1:
                x_axis = st.selectbox("選擇橫軸", ['caldt', '年份', '市場環境', 'seniority_label', 'mgmt_name', '股債配置比', 'index_fund_flag', 'dead_flag'])
            with c2:
                y_metrics_raw = {'月報酬率': 'mret', '資產規模 (MTNA)': 'mtna', '淨資金流 (Net Flow)': 'net_flow', '費用率 (Exp Ratio)': 'exp_ratio', '管理費 (Mgmt Fee)': 'mgmt_fee', '換手率 (Turnover)': 'turn_ratio', '基金年齡 (Age)': 'age'}
                y_sel = st.selectbox("選擇縱軸", list(y_metrics_raw.keys()))
                y_col = y_metrics_raw[y_sel]
            with c3:
                color_col = st.selectbox("選擇分組顏色", ['seniority_label', '市場環境', '股債配置比', 'index_fund_flag'])
            final_x = 'yr_tmp' if x_axis == '年份' else x_axis
            if x_axis == '年份':
                df_sandbox['yr_tmp'] = df_sandbox['caldt'].dt.year
            fig_raw = px.scatter(df_sandbox, x=final_x, y=y_col, color=color_col, opacity=0.5, marginal_y="violin", title=f"原始數據分佈：{x_axis} vs {y_sel}", labels={y_col: y_sel, final_x: x_axis}, hover_data=['fund_name', 'mgmt_name'])
            st.plotly_chart(fig_raw, use_container_width=True)
            st.divider()
            st.subheader("🧪 邊界敏感度與相關性分析")
            m1, m2 = st.columns(2)
            senior_val = df_sandbox[df_sandbox['tenure'] >= senior_threshold]['mret'].mean()
            fringe_mask = (df_sandbox['tenure'] >= senior_threshold - 0.5) & (df_sandbox['tenure'] <= senior_threshold + 0.5)
            fringe_val = df_sandbox[fringe_mask]['mret'].mean()
            m1.metric("資深組平均月回報", f"{senior_val:.2%}")
            m2.metric("臨界區間 (±0.5Y) 回報", f"{fringe_val:.2%}")
            num_df = df_sandbox.select_dtypes(include=[np.number])
            valid_cols = [c for c in num_df.columns if c not in ['crsp_fundno', 'tenure', 'yr']]
            if len(valid_cols) > 1:
                fig_corr = px.imshow(num_df[valid_cols].corr(), text_auto=".2f", color_continuous_scale='RdBu_r', aspect="auto")
                st.plotly_chart(fig_corr, use_container_width=True)

        with tab5:
            st.header("🧭 基金家族兩層篩選與研究邊界分析（新增）")
            st.info("此 Tab 對應老師要求：第一層先看基金家族總規模，第二層再看 1/3/5 年平均報酬、費用、Flow、管理年資、風險控管與投組情況，並與 S&P500 比較。")

            layer1, layer2 = build_family_two_layer_table(df_f, years=[1, 3, 5], sp500_df=sp500_df, manual_sp500_ann=manual_sp500_ann)
            if layer1.empty or layer2.empty:
                st.warning("資料不足，無法建立兩層分析表。")
            else:
                st.subheader("第一層：基金家族規模加總（以每檔基金最新 MTNA 加總）")
                top_n = st.slider("顯示前 N 大基金家族", 5, 50, 20, 1)
                layer1_sorted = layer1.sort_values('基金家族總資產規模_MTNA', ascending=False).head(top_n)
                st.dataframe(layer1_sorted, use_container_width=True)
                fig_size = px.bar(layer1_sorted, x='基金家族', y='基金家族總資產規模_MTNA', color='基金家族規模分類', title='第一層：基金家族總資產規模排名')
                fig_size.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_size, use_container_width=True)

                st.divider()
                st.subheader("第二層：1/3/5 年 Data Mining 指標與 S&P500 比較")
                period_sel = st.selectbox("選擇分析期間", ['過去1年', '過去3年', '過去5年'], index=1)
                layer2_period = layer2[layer2['期間'] == period_sel].copy()

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    min_size_pct = st.slider("基金家族規模分位數門檻", 0, 100, 50, 5) / 100
                with c2:
                    min_flow = st.number_input("平均 Flow 最低門檻", value=float(np.nanpercentile(layer2_period['平均Flow'].dropna(), 50)) if layer2_period['平均Flow'].notna().any() else 0.0)
                with c3:
                    max_exp = st.slider("平均費用率上限", 0.0, 0.05, 0.02, 0.001, format="%.3f")
                with c4:
                    min_tenure = st.slider("平均管理年資最低門檻", 0.0, 20.0, 5.0, 0.5)

                size_rank = layer1[['基金家族', '規模分位數']]
                boundary = layer2_period.merge(size_rank, on='基金家族', how='left')
                boundary['通過第一層_規模'] = boundary['規模分位數'] >= min_size_pct
                boundary['通過第二層_基本條件'] = (
                    (boundary['平均Flow'].fillna(-np.inf) >= min_flow) &
                    (boundary['平均費用率'].fillna(np.inf) <= max_exp) &
                    (boundary['平均管理年資'].fillna(-np.inf) >= min_tenure)
                )
                boundary['兩層皆通過'] = boundary['通過第一層_規模'] & boundary['通過第二層_基本條件']

                st.dataframe(format_pct_columns(boundary.sort_values('相對S&P500超額報酬', ascending=False), ['平均年化報酬率', 'S&P500年化報酬率', '相對S&P500超額報酬', '平均費用率', '年化波動度', 'Max Drawdown', '平均換手率']), use_container_width=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("通過兩層家族數", int(boundary['兩層皆通過'].sum()))
                m2.metric("平均超額報酬", f"{boundary['相對S&P500超額報酬'].mean():.2%}")
                m3.metric("超越 S&P500 20% 以上", int((boundary['相對S&P500超額報酬'] >= 0.20).sum()))
                m4.metric("落後 S&P500", int((boundary['相對S&P500超額報酬'] < 0).sum()))

                # Plotly 的 marker size 不接受負數；Flow 可能為負，
                # 因此用絕對值作為氣泡大小，方向/好壞仍保留在 hover 與表格中。
                boundary['Flow氣泡大小'] = boundary['平均Flow'].abs().fillna(0)
                if boundary['Flow氣泡大小'].max() == 0:
                    boundary['Flow氣泡大小'] = 1.0

                fig_scatter = px.scatter(
                    boundary,
                    x='基金家族總資產規模_MTNA', y='相對S&P500超額報酬',
                    size='Flow氣泡大小', color='績效判定', hover_name='基金家族',
                    hover_data={
                        '平均Flow': ':,.2f',
                        'Flow氣泡大小': False,
                        '平均費用率': ':.2%',
                        '平均管理年資': ':.2f',
                        'Sharpe Ratio': ':.3f',
                        'Max Drawdown': ':.2%',
                        '主要投組型態': True,
                        '兩層皆通過': True,
                    },
                    title=f'{period_sel}：基金家族規模、Flow 絕對強度與相對 S&P500 績效',
                    labels={'基金家族總資產規模_MTNA': '基金家族總規模', '相對S&P500超額報酬': '相對 S&P500 超額年化報酬'}
                )
                fig_scatter.add_hline(y=0, line_dash='dash')
                fig_scatter.add_hline(y=0.20, line_dash='dot')
                fig_scatter.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig_scatter, use_container_width=True)

                fig_heat = px.imshow(
                    boundary[['基金家族總資產規模_MTNA', '平均年化報酬率', '相對S&P500超額報酬', '平均費用率', '平均Flow', '平均管理年資', '年化波動度', 'Max Drawdown', 'Sharpe Ratio', '平均換手率']].corr(),
                    text_auto='.2f', color_continuous_scale='RdBu_r', aspect='auto', title='Data Mining 相關性熱圖：誰最可能解釋績效？'
                )
                st.plotly_chart(fig_heat, use_container_width=True)

                st.divider()
                st.subheader("基金經理人層級：找出高績效經理人屬於哪種條件")
                mgr_years = st.selectbox("經理人績效分析期間", [1, 3, 5], index=1)
                mgr_df = build_manager_performance_table(df_f, selected_years=mgr_years)
                if not mgr_df.empty:
                    mgr_df['績效分組'] = pd.qcut(mgr_df['平均年化報酬率'].rank(method='first'), q=4, labels=['低績效', '中低績效', '中高績效', '高績效'])
                    fig_mgr = px.scatter(
                        mgr_df,
                        x='平均費用率', y='平均年化報酬率', size='基金家族總資產規模_MTNA', color='績效分組',
                        hover_name='基金經理人', hover_data=['基金家族', '平均Flow', '平均管理年資', 'Sharpe近似', 'MaxDrawdown'],
                        title='經理人表現來源判讀：低費用？大公司？高 Flow？長年資？',
                        labels={'平均費用率': '平均費用率', '平均年化報酬率': '平均年化報酬率'}
                    )
                    fig_mgr.update_layout(xaxis_tickformat='.2%', yaxis_tickformat='.1%')
                    st.plotly_chart(fig_mgr, use_container_width=True)

                    explain = mgr_df.groupby('績效分組').agg(
                        平均基金家族規模=('基金家族總資產規模_MTNA', 'mean'),
                        平均費用率=('平均費用率', 'mean'),
                        平均Flow=('平均Flow', 'mean'),
                        平均管理年資=('平均管理年資', 'mean'),
                        平均Sharpe=('Sharpe近似', 'mean'),
                        平均MDD=('MaxDrawdown', 'mean'),
                        經理人數=('基金經理人', 'count')
                    ).reset_index()
                    st.write("**績效分組比較表：可用來寫研究邊界與初步結論**")
                    st.dataframe(format_pct_columns(explain, ['平均費用率', '平均MDD']), use_container_width=True)

                    st.markdown("""
                    **視覺化判讀指示：**
                    - 若高績效點集中在大氣泡，代表績效可能與 **大型基金家族資源 / 品牌優勢** 有關。
                    - 若高績效點集中在左側低費用區，代表可能與 **低費用率** 有關。
                    - 若高績效點 hover 顯示平均 Flow 較高，代表可能與 **資金流動能 / 投資人信任** 有關。
                    - 若高績效經理人的平均管理年資較高，代表可能與 **經理人經驗與任期穩定性** 有關。
                    - 若高績效同時具有較高 Sharpe、較低 Max Drawdown，代表不只是報酬高，也具有 **風險控管能力**。
                    """)

                st.download_button("📥 下載 Tab5 兩層篩選結果", boundary.to_csv(index=False), "family_two_layer_boundary.csv")

        with tab6:
            st.header("原始數據明細")
            st.dataframe(df_f.head(1000), use_container_width=True)
            st.download_button("📥 下載完整過濾後數據", df_f.to_csv(index=False), "balanced_full_analysis.csv")
else:
    st.info("👋 歡迎！請從側邊欄上傳您的 CSV 數據檔案（例如 balanced_before2010.csv、balanced_after2010.csv）以啟動分析。")
