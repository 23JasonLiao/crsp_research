import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.stats import skew, kurtosis

# --- 1. 網頁配置與標題 ---
st.set_page_config(layout="wide", page_title="平衡型基金：多因子與股債輪動分析儀表板")

st.title("👨‍💼 專業資產管理：多因子指標與股債輪動分析儀表板")
st.markdown("""
本系統結合了 **Modern Portfolio Theory (MPT)** 與 **市場環境偵測 (Market Regime Detection)**。
您可以觀測經理人在「股票動能期」與「債券避險期」的表現差異，並透過專業金融指標評估其長期穩定性。
""")

# --- 2. 資料讀取與處理函數 ---
@st.cache_data
def load_and_combine_data(uploaded_files):
    df_list = []
    for file in uploaded_files:
        try:
            # 增加 low_memory=False 解決大型 CSV 的 DtypeWarning
            temp_df = pd.read_csv(file, low_memory=False)
            df_list.append(temp_df)
        except Exception as e:
            st.error(f"讀取檔案 {file.name} 時發生錯誤: {e}")
    
    if not df_list:
        return None
        
    df = pd.concat(df_list, ignore_index=True)
    df['caldt'] = pd.to_datetime(df['caldt'], errors='coerce')
    df['mret'] = pd.to_numeric(df['mret'], errors='coerce')
    df['exp_ratio'] = pd.to_numeric(df['exp_ratio'], errors='coerce')
    
    # 處理經理人就任日期
    if 'mgr_dt' in df.columns:
        df['mgr_dt'] = pd.to_datetime(df['mgr_dt'], errors='coerce')
        df['mgr_dt'] = df['mgr_dt'].fillna(df['caldt'])
    else:
        df['mgr_dt'] = df['caldt']

    # 處理管理公司名稱
    if 'mgmt_name' not in df.columns and 'mgr_name' in df.columns:
        df['mgmt_name'] = df['mgr_name']

    return df.dropna(subset=['mret', 'caldt', 'mgmt_name'])

# --- 3. 核心功能：市場環境判斷 (股 vs 債) ---
def detect_market_regime(df):
    """
    透過全市場平均報酬與波動判斷市場環境
    """
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
    """
    計算每家管理公司的時間序列回撤數據
    """
    df_sorted = df.sort_values('caldt')
    drawdown_list = []
    
    for name, group in df_sorted.groupby('mgmt_name'):
        group = group.copy()
        # 計算累計回報 (Wealth Index)
        group['wealth_index'] = (1 + group['mret']).cumprod()
        # 計算歷史滾動最高點
        group['previous_peaks'] = group['wealth_index'].cummax()
        # 計算當前對比最高點的跌幅 (回撤)
        group['drawdown'] = (group['wealth_index'] - group['previous_peaks']) / group['previous_peaks']
        drawdown_list.append(group)
        
    return pd.concat(drawdown_list)

# --- 4. 專業資產管理因子計算 (Expert Metrics) ---
def calculate_asset_management_factors(df):
    results = []
    for name, group in df.groupby('mgmt_name'):
        mrets = group['mret'].dropna()
        if len(mrets) < 6: continue
        
        # 1. 報酬與風險 (年化)
        ann_ret = (1 + mrets.mean())**12 - 1
        ann_vol = mrets.std() * np.sqrt(12)
        
        # 2. Sharpe & Sortino (無風險利率設為 1%)
        sharpe = (ann_ret - 0.01) / ann_vol if ann_vol > 0 else 0
        downside_rets = mrets[mrets < 0]
        downside_vol = downside_rets.std() * np.sqrt(12) if len(downside_rets) > 0 else 0
        sortino = (ann_ret - 0.01) / downside_vol if downside_vol > 0 else 0
        
        # 3. 最大回撤 (MDD)
        cum_ret = (1 + mrets).cumprod()
        running_max = cum_ret.cummax()
        mdd = ((cum_ret - running_max) / running_max).min()
        
        # 4. 統計因子
        sk = skew(mrets)
        kt = kurtosis(mrets)
        var_95 = np.percentile(mrets, 5)
        
        results.append({
            '管理公司': name,
            '年化報酬率': ann_ret,
            '年化波動度': ann_vol,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': mdd,
            '偏度 (Skewness)': round(sk, 2),
            '峰度 (Kurtosis)': round(kt, 2),
            'VaR 95% (月度)': f"{var_95:.2%}"
        })
    return pd.DataFrame(results)

def render_mgmt_treemap(df):
    """
    製作資產管理 Treemap
    大小：資產規模 (mtna)
    顏色：月報酬率 (mret)
    標籤：管理公司、推估股債配置
    """
    # 1. 取得每家基金最新的資料點 (最新資產規模)
    latest_df = df.sort_values('caldt').groupby('crsp_fundno').last().reset_index()
    
    # 2. 處理股債比標籤 (從 lipper_class_name 或 policy 提取)
    # 這裡建立一個簡單的映射邏輯，你可以根據實際數據調整
    def estimate_allocation(row):
        policy = str(row.get('policy', '')).upper()
        lipper = str(row.get('lipper_class_name', '')).upper()
        if 'GROWTH' in lipper or '70% TO 90% EQUITY' in lipper: return "80:20 (激進型)"
        if 'MODERATE' in lipper or '50% TO 70% EQUITY' in lipper: return "60:40 (平衡型)"
        if 'CONSERVATIVE' in lipper or '30% TO 50% EQUITY' in lipper: return "40:60 (保守型)"
        return "60:40 (標準平衡)"

    latest_df['股債配置比'] = latest_df.apply(estimate_allocation, axis=1)
    
    # 3. 彙總到管理公司層級
    mgmt_tree = latest_df.groupby(['mgmt_name', '股債配置比']).agg({
        'mtna': 'sum',      # 總資產規模
        'mret': 'mean',      # 平均報酬
        'fund_name': 'count' # 旗下基金數量
    }).reset_index()
    
    # 4. 繪製 Treemap
    fig = px.treemap(
        mgmt_tree,
        path=[px.Constant("全體平衡型基金市場"), '股債配置比', 'mgmt_name'],
        values='mtna',
        color='mret',
        color_continuous_scale='RdYlGn', # 紅黃綠顏色分支
        color_continuous_midpoint=0,     # 以 0% 報酬為顏色中點
        hover_data=['fund_name'],
        title="資產管理版圖：規模(大小) vs. 表現(顏色) vs. 配置(層級)",
        labels={'mtna': '資產規模 (M)', 'mret': '平均月回報', 'mgmt_name': '管理公司'}
    )
    
    fig.update_traces(textinfo="label+value+percent parent")
    return fig


# --- 修改後的矩陣函數 (移除不支援的 animation_frame) ---
def render_dynamic_factor_matrix(df, selected_year):
    """
    根據選定年份分析 Flow, MTNA, Mret 的動態矩陣
    """
    st.subheader(f"🔄 {selected_year} 年：資金流、規模與報酬之動態矩陣")
    
    # 1. 準備必要欄位
    cols = ['mret', 'mtna']
    if 'new_sls' in df.columns and 'redemp' in df.columns:
        # 如果沒有 net_flow，計算一個
        if 'net_flow' not in df.columns:
            df['net_flow'] = df['new_sls'] - df['redemp']
        cols.append('net_flow')
    
    # 2. 過濾特定年份數據
    df_year = df[df['caldt'].dt.year == selected_year].copy()
    df_year = df_year.dropna(subset=cols)

    if df_year.empty:
        st.warning(f"{selected_year} 年份無足夠數據進行矩陣分析。")
        return None

    # 3. 繪製矩陣圖
    fig = px.scatter_matrix(
        df_year,
        dimensions=cols,
        color="seniority_label",
        opacity=0.6,
        labels={'mret': '報酬 (Mret)', 'mtna': '規模 (MTNA)', 'net_flow': '資金流 (Flow)'},
        title=f"股債平衡基金因子交互作用 ({selected_year})",
        height=700,
    )

    # 4. 優化矩陣外觀
    fig.update_traces(diagonal_visible=True, marker=dict(size=5))
    return fig

# --- 5. 主程式 ---
uploaded_files = st.sidebar.file_uploader("上傳平衡型基金 CSV (可選多個)", type="csv", accept_multiple_files=True)

if uploaded_files:
    df = load_and_combine_data(uploaded_files)
    
    if df is not None:
        # --- 5.1 合併市場環境與年資計算 ---
        regime_df = detect_market_regime(df)
        df = df.merge(regime_df, on='caldt')
        
        df['tenure'] = (df['caldt'] - df['mgr_dt']).dt.days / 365.25
        df['tenure'] = df['tenure'].clip(lower=0)
        df['seniority_label'] = (df['tenure'] > 10).map({True: '資深 (10年以上)', False: '一般資歷'})

        # --- 5.2 側邊欄過濾 ---
        all_mgmt = sorted(df['mgmt_name'].dropna().unique())
        selected_mgmt = st.sidebar.multiselect("選擇管理公司", options=all_mgmt)
        
        df_f = df.copy()
        if selected_mgmt:
            df_f = df_f[df_f['mgmt_name'].isin(selected_mgmt)]

        # --- 6. 分頁介面 ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["💼 專業資產管理因子", "📈 年資與股債環境適應性", "📊 資金流行為與資產動態生態分析", "🔍 自定義交叉探索", "📋 原始數據"])

        with tab1:
            st.header("資產管理核心指標 (MPT & Risk Factors)")
            st.info("💡 專業分析：透過夏普值與索提諾值判斷風險性價比，並觀測最大回撤以評估抗跌性。")
            
            factor_df = calculate_asset_management_factors(df_f)
            
            # --- 修改後的表格格式化顯示 ---
            display_df = factor_df.copy()
            
            # 1. 處理百分比欄位 (報酬、波動、MDD)
            pct_cols = ['年化報酬率', '年化波動度', 'Max Drawdown']
            for col in pct_cols:
                display_df[col] = display_df[col].map('{:.2%}'.format)
            
            # 2. 處理數值欄位 (Sharpe, Sortino, Skewness, Kurtosis) 
            # 確保保留兩到四位小數，方便對比
            num_cols = ['Sharpe Ratio', 'Sortino Ratio', '偏度 (Skewness)', '峰度 (Kurtosis)']
            for col in num_cols:
                display_df[col] = display_df[col].map('{:.4f}'.format)
                
            # 3. 顯示表格 (VaR 95% 在函數中已經是字串格式，故不需額外處理)
            st.dataframe(display_df, use_container_width=True)
            
            # --- 風險收益效率前緣圖 (維持不變) ---
            st.subheader("風險-報酬效率前緣圖")
            factor_df['BubbleSize'] = factor_df['Sharpe Ratio'].apply(lambda x: max(x, 0) + 0.05)
            
            fig_risk = px.scatter(
                factor_df, x="年化波動度", y="年化報酬率", size="BubbleSize", color="管理公司",
                # 這裡也可以多增加 hover 資訊
                hover_data=["Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "偏度 (Skewness)", "VaR 95% (月度)"],
                title="氣泡大小代表 Sharpe Ratio (越高性能越好)"
            )
            fig_risk.update_layout(xaxis_tickformat='.1%', yaxis_tickformat='.1%')
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # --- 下方歷史回撤圖維持不變 ---
            st.divider()
            st.subheader("📉 歷史回撤趨勢圖 (Underwater Chart)")
            st.info("💡 此圖顯示基金從歷史高點跌落的幅度，跌得越深代表抗風險能力越弱。")
            
            # 計算時間序列回撤
            df_dd = calculate_drawdown_series(df_f)
            
            # 繪製 Underwater Chart
            fig_dd = px.line(
                df_dd, x="caldt", y="drawdown", color="mgmt_name",
                labels={'drawdown': '回撤幅度', 'caldt': '日期'},
                title="基金歷史回撤路徑 (Max Drawdown Path)"
            )
            # 設定 Y 軸為百分比，並將範圍限制在 [最小值, 0]
            fig_dd.update_layout(yaxis_tickformat='.1%', yaxis_range=[df_dd['drawdown'].min()*1.1, 0])
            # 增加 0 位水平線
            fig_dd.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig_dd, use_container_width=True)

        with tab2:
            st.header("經理人任期與股債市場環境分析")
            st.info("""
            💡 **股債環境判斷邏輯：**
            - **股票動能強**：市場報酬為正且波動較高（高 Beta 策略佔優）。
            - **債券/穩健強**：市場報酬為正但波動極低（穩健防禦策略佔優）。
            """)

            col_a, col_b = st.columns([2, 1])
            with col_a:
                # 股債環境分佈圖
                fig_regime = px.scatter(
                    df_f, x="caldt", y="mret", color="市場環境",
                    symbol="seniority_label",
                    title="不同市場環境下的經理人表現點位",
                    labels={'mret': '月回報率', 'caldt': '日期'}
                )
                st.plotly_chart(fig_regime, use_container_width=True)
            
            with col_b:
                # 環境適應性對比 (平均報酬)
                regime_compare = df_f.groupby(['市場環境', 'seniority_label'])['mret'].mean().reset_index()
                fig_bar = px.bar(
                    regime_compare, x="市場環境", y="mret", color="seniority_label", barmode="group",
                    title="不同資歷在股/債環境的平均回報",
                    labels={'mret': '平均月報酬'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("資歷深淺對環境的敏感度趨勢")
            fig_trend = px.scatter(
                df_f, x="tenure", y="mret", color="市場環境", 
                trendline="ols",
                title="經理人年資 vs 月回報 (按市場環境分類)",
                labels={'tenure': '在職年資 (年)', 'mret': '月回報率'}
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with tab3:
          st.header("📊 資產與因子動態交互分析")
    
          # 第一部分：Treemap (規模與配置)
          if 'mtna' in df_f.columns:
             st.subheader("🏢 資產管理版圖 (Treemap)")
             fig_tree = render_mgmt_treemap(df_f) # 之前給你的 Treemap FUNC
             st.plotly_chart(fig_tree, use_container_width=True)
    
             st.divider()
    
        # 第二部分：年份拉桿與動態矩陣
          st.subheader("🗓️ 時間維度因子分析")
          st.write("請拖動下方拉桿切換年份，觀察 Mret、MTNA、Flow 的相關性變化：")
      
         # 建立一個 Streamlit Slider 作為年份拉桿
          available_years = sorted(df_f['caldt'].dt.year.unique())
          selected_year = st.slider("選擇分析年份", 
                              min_value=int(min(available_years)), 
                              max_value=int(max(available_years)), 
                              value=int(max(available_years)))

        # 呼叫矩陣函數
          fig_matrix = render_dynamic_factor_matrix(df_f, selected_year)
          if fig_matrix:
              st.plotly_chart(fig_matrix, width="stretch")
 
              st.info("""
                   **💡 專業金融推論指南：**
                    1. **追漲現象 (Performance-Flow)**：觀察「報酬」對「資金流」的散佈。若呈現正相關斜率，代表投資者傾向於追逐過去表現。
                     2. **規模瓶頸 (Size vs Performance)**：觀察「規模」變大時，「報酬」的垂直分布是否變窄或下移，這可用來討論「基金規模是否是績效的敵人」。
                """)
  
              st.divider()
          
            # --- 3. 資金流敏感度分析 (取代原本的極端測試) ---
          st.subheader("🌊 資金流敏感度：資深經理人更能留住資金嗎？")
          st.markdown("分析在不同報酬水準下，投資者對資深 vs 一般經理人的資金進出行為。")

          if 'net_flow' in df_f.columns:
                # 建立兩欄排版
              col_chart, col_stat = st.columns([3, 2])
                
              with col_chart:
        # 使用 scatter_gl 代替 scatter 以提升效能
        # 如果 px.scatter_gl 在你的環境有問題，可維持 px.scatter 並在下方 update_traces 修改
                fig_flow_sens = px.scatter(
                     df_f, 
                     x="mret", 
                     y="net_flow", 
                     color="seniority_label",
                     trendline="ols",
                     title=f"資金流對報酬之敏感度 (Flow Sensitivity)",
                     labels={"mret": "月報酬率", "net_flow": "淨資金流 (Net Flow)"},
                     color_discrete_map={"資深 (10年以上)": "#2E7D32", "一般資歷": "#C62828"},
                     opacity=0.4,
                     # 移除 render_mode，改在 update_traces 設定
                ) 

                # 2. 【核心修正】安全地逐一更新 Trace
                for trace in fig_flow_sens.data:
               # 判斷是否為 scatter (點) 而不是趨勢線
                    if trace.type == 'scatter':
                        trace.render_mode = 'webgl'  # 手動強制指定
                        trace.marker.size = 4        # 調整大小
                
                fig_flow_sens.update_layout(
                   xaxis_tickformat='.1%',
                   hovermode=False,          # 關閉全域 Hover 偵測，這是滑鼠卡頓的主因
                   uirevision='constant',     # 當你調整其他元件時，保持目前縮放範圍        
                )
    
              st.plotly_chart(fig_flow_sens, width='stretch', theme=None)
                
              with col_stat:
                    # 統計不同資歷的平均資金吸引力
                   flow_stats = df_f.groupby('seniority_label').agg({
                        'net_flow': ['mean', 'std', 'sum'],
                        'mret': 'mean'
                    })
                   flow_stats.columns = ['平均月流向', '流向波動', '總淨流向', '平均回報']
                    
                   st.write("**經理人資歷與資金流統計：**")
                   st.table(flow_stats.style.format({'平均回報': '{:.2%}', '平均月流向': '{:.2f}', '總淨流向': '{:.2f}'}))
                    
                   st.info(f"""
                    **💡 專家觀測點：**
                    - **回歸線斜率**：若斜率越陡，代表投資者對該類經理人的「追漲殺跌」行為越明顯。
                    - **資金黏著度**：觀察在報酬為負（左半區）時，資深經理人的資金流出（Net Flow < 0）是否比一般經理人緩和。
                    """)
          else:
                st.warning("數據中缺少 new_sls 或 redemp 欄位，無法進行資金流敏感度分析。")

          st.divider()
          st.info("""
            **💡 總體行為推論：**
            1. **Performance-Flow Abnormality**：檢視報酬是否為資金流的領先指標。
            2. **Investor Trust**：資深經理人通常擁有更高的「品牌溢價」，這在 Net Flow 的穩定性上可以得到體現。
         """)
          
          
          with tab4:
               st.header("🔍 自定義交叉探索沙盒：動態邊界分析")
    
               # --- 第零層：動態邊界定義 ---
               st.subheader("⚙️ 定義研究邊界")
               exp_col1, exp_col2, exp_col3 = st.columns(3)
     
               with exp_col1:
                  senior_threshold = st.slider("資深經理人年資門檻 (年)", 1.0, 20.0, 10.0, 0.5)
        
               with exp_col2:
                 # 新增：回報門檻拉桿，回應老師「負一點點算低迷嗎」的質疑
                  ret_cutoff = st.slider("市場低迷判定門檻 (月報酬 %)", -5.0, 2.0, 0.0, 0.1) / 100
               with exp_col3:
                  vol_adjustment = st.slider("市場波動判定偏移 (±%)", -50, 50, 0, 5) / 100
      
              # 使用副本進行處理，避免影響原始數據
               df_sandbox = df_f.copy()
    
              # 1. 動態年資計算
               df_sandbox['seniority_label'] = df_sandbox['tenure'].apply(
                   lambda x: f'資深 ({senior_threshold}Y+)' if x >= senior_threshold else '一般資歷'
               )
    
              # 2. 動態市場環境計算
               market_stats = df_sandbox.groupby('caldt')['mret'].agg(['mean', 'std']).reset_index()
               dynamic_vol_limit = market_stats['std'].median() * (1 + vol_adjustment)
    
               def dynamic_label(row):
               # 使用拉桿設定的 ret_cutoff 取代硬碼的 0
                   if row['mean'] < ret_cutoff: return "市場低迷 (雙殺)"
                   if row['std'] > dynamic_vol_limit: return "股票動能強 (高波)"
                   return "債券/穩健強 (低波)"
    
               market_stats['市場環境'] = market_stats.apply(dynamic_label, axis=1)
               if '市場環境' in df_sandbox.columns:
                  df_sandbox = df_sandbox.drop(columns=['市場環境'])
                  df_sandbox = df_sandbox.merge(market_stats[['caldt', '市場環境']], on='caldt')

               # 3. 股債配置比計算
               def get_alloc_label(row):
                   lipp = str(row.get('lipper_class_name', '')).upper()
                   if 'GROWTH' in lipp or '70% TO 90%' in lipp: return "80:20 (激進)"
                   if 'MODERATE' in lipp or '50% TO 70%' in lipp: return "60:40 (平衡)"
                   if 'CONSERVATIVE' in lipp or '30% TO 50%' in lipp: return "40:60 (保守)"
                   return "60:40 (標準)"
               df_sandbox['股債配置比'] = df_sandbox.apply(get_alloc_label, axis=1)

       

               # --- 第一層：雙軸交叉分析 ---
               st.divider()
               st.subheader("📊 雙軸交互統計")
               st.markdown("此圖表呈現原始數據點位，不進行聚合，以便觀察數據的分佈規律。")
               c1, c2, c3 = st.columns(3)
               with c1:
                 # 橫軸：加入來自熱圖的分類維度
                 # 移除 lipper_obj_cd 與 crsp_style 相關
                   x_opts = ['caldt', '年份', '市場環境', 'seniority_label', 'mgmt_name', '股債配置比', 'index_fund_flag', 'dead_flag']
                   x_axis = st.selectbox("選擇橫軸 (X-axis)", options=x_opts, index=0)
    
               with c2:
                  # 縱軸：從熱圖中抓取關鍵量化指標
                   y_metrics_raw = {
                      '月報酬率': 'mret',
                       '資產規模 (MTNA)': 'mtna',
                       '淨資金流 (Net Flow)': 'net_flow',
                       '費用率 (Exp Ratio)': 'exp_ratio',
                       '管理費 (Mgmt Fee)': 'mgmt_fee',
                       '換手率 (Turnover)': 'turn_ratio',
                       '基金年齡 (Age)': 'age'
                   }
                   y_sel = st.selectbox("選擇縱軸 (Y-axis)", options=list(y_metrics_raw.keys()), index=0)
                   y_col = y_metrics_raw[y_sel]
               
               with c3:
                   # 分組：提供具備金融研究意義的標籤
                   color_opts = ['seniority_label', '市場環境', '股債配置比', 'index_fund_flag']
                   color_col = st.selectbox("選擇分組顏色 (Legend)", options=color_opts, index=1)
              # 數據預處理
               if x_axis == '年份':
                   df_sandbox['yr_tmp'] = df_sandbox['caldt'].dt.year
                   final_x = 'yr_tmp'
               else:
                   final_x = x_axis

               # 繪製點位散佈圖 (Scatter)
               fig_raw = px.scatter(
                   df_sandbox,
                   x=final_x,
                   y=y_col,
                   color=color_col,
                   opacity=0.5,
                   marginal_y="violin", # 在側邊增加小提琴圖，顯示分佈密度
                   title=f"原始數據分佈：{x_axis} vs {y_sel}",
                   labels={y_col: y_sel, final_x: x_axis},
                   hover_data=['fund_name', 'mgmt_name']
               )

              # 效能與視覺優化：只對 Scatter 類型套用 WebGL
               for trace in fig_raw.data:
                   # 檢查是否為散佈圖層 (Scatter 或 Scattergl)
                   if hasattr(trace, 'render_mode'):
                       trace.render_mode = 'webgl'
                   
                   # 如果是 Scatter 類型，縮小標記大小
                   if 'marker' in trace and hasattr(trace.marker, 'size'):
                       trace.marker.size = 4
               
               st.plotly_chart(fig_raw, use_container_width=True)
        
               # --- 第三層：專家觀測與結論 ---
               st.info(f"""
                **💡 金融推論：**
                1. 當前低迷門檻設為 **{ret_cutoff:.2%}**，代表全市場平均月報酬低於此值即判定為「雙殺」環境。
                2. 您可以觀察 X 軸切換為 **『費用率』** 或 **『規模』** 時，資深經理人的分佈是否具備規模優勢。
                """)
                
            
                
              # --- 第三層：邊界敏感度分析 (核心學術回應) ---
               st.divider()
               st.subheader("🧪 邊界敏感度與相關性分析")
    
               m1, m2 = st.columns(2)
               senior_val = df_sandbox[df_sandbox['tenure'] >= senior_threshold]['mret'].mean()
               # 臨界區間定在門檻上下 0.5 年
               fringe_mask = (df_sandbox['tenure'] >= senior_threshold - 0.5) & (df_sandbox['tenure'] <= senior_threshold + 0.5)
               fringe_val = df_sandbox[fringe_mask]['mret'].mean()
    
               m1.metric("資深組平均月回報", f"{senior_val:.2%}")
               m2.metric("臨界區間 (±0.5Y) 回報", f"{fringe_val:.2%}")

              # 自動相關性矩陣
               num_df = df_sandbox.select_dtypes(include=[np.number])
               valid_cols = [c for c in num_df.columns if c not in ['crsp_fundno', 'tenure', 'yr']]
               if len(valid_cols) > 1:
                  corr_matrix = num_df[valid_cols].corr()
                  fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r', aspect="auto")
                  st.plotly_chart(fig_corr, use_container_width=True)

        with tab5:
            st.header("原始數據明細")
            st.dataframe(df_f.head(1000), use_container_width=True)
            st.download_button("📥 下載完整過濾後數據", df_f.to_csv(index=False), "balanced_full_analysis.csv")

else:
    st.info("👋 歡迎！請從側邊欄上傳您的 CSV 數據檔案（例如 `balanced_before2010.csv` 等）以啟動分析。")