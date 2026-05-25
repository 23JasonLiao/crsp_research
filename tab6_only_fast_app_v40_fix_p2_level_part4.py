import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import inspect
import warnings
from pathlib import Path
from scipy.stats import skew, kurtosis

st.set_page_config(layout="wide", page_title="Tab6：報酬與 S&P500 對照分析")

APP_DIR = Path(__file__).resolve().parent
DEFAULT_FUND_FILES = [
    APP_DIR / "balanced_before2010.csv",
    APP_DIR / "balanced_after2010.csv",
]
DEFAULT_SP500_FILE = APP_DIR / "sp500_monthly_returns_1871_2026.csv"


@st.cache_resource(show_spinner=False)
def load_and_combine_data(uploaded_files):
    """Memory-safe loader for the Tab6-only app.

    The original CSV files can contain many columns that are not used by Tab6.
    Reading all columns and then caching/copying the whole DataFrame may exhaust
    memory on machines with limited RAM. This loader reads only the fields needed
    by Part 1~Part 5, then converts repeated text columns to category dtype.
    """
    needed_cols = {
        'caldt', 'mgr_dt', 'first_offer_dt', 'flow_report_dt', 'trans_dt',
        'mret', 'mtna', 'exp_ratio', 'mgmt_fee', 'turn_ratio', 'new_sls',
        'rein_sls', 'oth_sls', 'redemp', 'age', 'mgmt_name', 'mgr_name',
        'fund_name', 'crsp_fundno', 'lipper_class_name', 'policy', 'net_flow'
    }
    df_list = []
    for file in uploaded_files:
        try:
            temp_df = pd.read_csv(
                file,
                low_memory=False,
                usecols=lambda c: c in needed_cols
            )
            df_list.append(temp_df)
        except Exception as e:
            fname = getattr(file, 'name', str(file))
            st.error(f"讀取檔案 {fname} 時發生錯誤: {e}")
    if not df_list:
        return None

    df = pd.concat(df_list, ignore_index=True, copy=False)

    for c in ['caldt', 'mgr_dt', 'first_offer_dt', 'flow_report_dt', 'trans_dt']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    for c in ['mret', 'mtna', 'exp_ratio', 'mgmt_fee', 'turn_ratio', 'new_sls', 'rein_sls', 'oth_sls', 'redemp', 'age', 'net_flow']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce', downcast='float')

    if 'mgr_dt' in df.columns:
        df['mgr_dt'] = df['mgr_dt'].fillna(df['caldt'])
    else:
        df['mgr_dt'] = df['caldt']

    if 'mgmt_name' not in df.columns and 'mgr_name' in df.columns:
        df['mgmt_name'] = df['mgr_name']
    if 'mgr_name' not in df.columns:
        df['mgr_name'] = 'Unknown Manager'

    if {'new_sls', 'redemp'}.issubset(df.columns):
        rein = df['rein_sls'] if 'rein_sls' in df.columns else 0
        oth = df['oth_sls'] if 'oth_sls' in df.columns else 0
        df['net_flow'] = (
            df['new_sls'].fillna(0) +
            pd.Series(rein, index=df.index).fillna(0) +
            pd.Series(oth, index=df.index).fillna(0) -
            df['redemp'].fillna(0)
        ).astype('float32')
    elif 'net_flow' not in df.columns:
        df['net_flow'] = np.nan

    df['tenure'] = ((df['caldt'] - df['mgr_dt']).dt.days / 365.25).astype('float32')
    df['tenure'] = df['tenure'].clip(lower=0)
    df['seniority_label'] = (df['tenure'] >= 10).map({True: '資深 (10年以上)', False: '一般資歷'}).astype('category')

    for c in ['mgmt_name', 'mgr_name', 'fund_name', 'lipper_class_name', 'policy']:
        if c in df.columns:
            df[c] = df[c].astype('category')

    keep = df['mret'].notna() & df['caldt'].notna() & df['mgmt_name'].notna()
    return df.loc[keep]


@st.cache_resource(show_spinner=False)
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
    b = b.loc[:, [date_col, ret_col]]
    b.columns = ['caldt', 'sp500_ret']
    b['caldt'] = pd.to_datetime(b['caldt'], errors='coerce')
    b['sp500_ret'] = pd.to_numeric(b['sp500_ret'], errors='coerce', downcast='float')
    return b.dropna(subset=['caldt', 'sp500_ret'])


def safe_percentile_score(series, higher_is_better=True):
    """把任意指標轉成 0~1 分數，方便做投資者偏好加權。"""
    x = pd.to_numeric(series, errors='coerce')
    if x.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index)
    score = x.rank(pct=True)
    if not higher_is_better:
        score = 1 - score
    return score.fillna(0.5)


def _clean_monthly_returns(s):
    """Clean monthly returns before compounding.

    Some CRSP rows can contain extreme or invalid return values. Direct cumprod/prod
    may overflow and produce RuntimeWarning: overflow encountered in accumulate.
    This helper keeps only finite returns with gross return > 0, so log compounding
    can be used safely.
    """
    r = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    return r[r > -0.999999]


def safe_compound_return(s):
    """Compound returns with log1p to avoid NumPy overflow warnings."""
    r = _clean_monthly_returns(s)
    if len(r) == 0:
        return np.nan
    log_sum = float(np.log1p(r.astype(float)).sum())
    if not np.isfinite(log_sum):
        return np.nan
    if log_sum > 700:
        return np.nan
    if log_sum < -700:
        return -1.0
    return float(np.expm1(log_sum))


def safe_annualized_return_from_monthly_mean(mean_m):
    """Annualize average monthly return without invalid exponent/overflow warnings."""
    if pd.isna(mean_m) or not np.isfinite(mean_m) or mean_m <= -0.999999:
        return np.nan
    log_ann = 12.0 * float(np.log1p(float(mean_m)))
    if not np.isfinite(log_ann):
        return np.nan
    if log_ann > 700:
        return np.nan
    if log_ann < -700:
        return -1.0
    return float(np.expm1(log_ann))


def max_drawdown_from_monthly(s):
    """Max drawdown using log wealth to avoid overflow from cumulative products."""
    r = _clean_monthly_returns(s)
    if len(r) == 0:
        return np.nan
    log_wealth = np.log1p(r.astype(float)).cumsum()
    running_peak = np.maximum.accumulate(log_wealth.to_numpy(dtype=float))
    drawdown = np.exp(np.clip(log_wealth.to_numpy(dtype=float) - running_peak, -700, 0)) - 1
    return float(np.nanmin(drawdown)) if len(drawdown) else np.nan




def safe_corr(a, b):
    """Return correlation without NumPy RuntimeWarning when either series is constant/too short.

    pandas.Series.corr eventually calls numpy.corrcoef, which can emit
    RuntimeWarning: invalid value encountered in divide when one side is
    constant.  This manual implementation checks the denominator first.
    """
    pair = pd.DataFrame({'a': pd.to_numeric(a, errors='coerce'), 'b': pd.to_numeric(b, errors='coerce')}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 2:
        return np.nan
    x = pair['a'].to_numpy(dtype=float)
    y = pair['b'].to_numpy(dtype=float)
    x = x - np.nanmean(x)
    y = y - np.nanmean(y)
    denom = np.sqrt(np.nansum(x * x) * np.nansum(y * y))
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    val = np.nansum(x * y) / denom
    return float(val) if np.isfinite(val) else np.nan


def safe_beta(a, b):
    """Return beta without NumPy RuntimeWarning.

    beta is undefined when benchmark variance is zero/too short.  This avoids
    np.cov because np.cov/corrcoef can warn on degenerate arrays.
    """
    pair = pd.DataFrame({'a': pd.to_numeric(a, errors='coerce'), 'b': pd.to_numeric(b, errors='coerce')}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 2:
        return np.nan
    x = pair['a'].to_numpy(dtype=float)
    y = pair['b'].to_numpy(dtype=float)
    x_centered = x - np.nanmean(x)
    y_centered = y - np.nanmean(y)
    var_b = np.nanmean(y_centered * y_centered)
    if not np.isfinite(var_b) or var_b <= 0:
        return np.nan
    cov_ab = np.nanmean(x_centered * y_centered)
    beta = cov_ab / var_b
    return float(beta) if np.isfinite(beta) else np.nan


def safe_skewness(s):
    """Skewness without scipy precision-loss warnings for constant arrays."""
    x = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if x.size < 3 or np.nanstd(x) <= 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        val = skew(x)
    return float(val) if np.isfinite(val) else np.nan


def safe_kurtosis(s):
    """Kurtosis without scipy precision-loss warnings for constant arrays."""
    x = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if x.size < 4 or np.nanstd(x) <= 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        val = kurtosis(x)
    return float(val) if np.isfinite(val) else np.nan




RADAR_METRIC_CONFIG = [
    {'label': '報酬率', 'score_col': '雷達_報酬率', 'raw_col': '年化報酬率', 'higher_is_better': True, 'unit_type': 'pct', 'unit_label': '%'},
    {'label': '超額報酬率', 'score_col': '雷達_超額報酬率', 'raw_col': '平均超額月報酬', 'higher_is_better': True, 'unit_type': 'pct', 'unit_label': '%'},
    {'label': 'Sharpe Ratio', 'score_col': '雷達_Sharpe Ratio', 'raw_col': 'Sharpe Ratio', 'higher_is_better': True, 'unit_type': 'num', 'unit_label': 'ratio'},
    {'label': '勝率', 'score_col': '雷達_勝率', 'raw_col': '打敗S&P500比例', 'higher_is_better': True, 'unit_type': 'pct', 'unit_label': '%'},
    {'label': '回撤控制', 'score_col': '雷達_回撤控制', 'raw_col': 'Max Drawdown', 'higher_is_better': True, 'unit_type': 'pct', 'unit_label': '%'},
    {'label': '波動度', 'score_col': '雷達_波動度', 'raw_col': '年化波動度', 'higher_is_better': False, 'unit_type': 'pct', 'unit_label': '%'},
    {'label': '費用率', 'score_col': '雷達_費用率', 'raw_col': '平均費用率', 'higher_is_better': False, 'unit_type': 'pct', 'unit_label': '%'},
    {'label': 'Flow', 'score_col': '雷達_Flow', 'raw_col': '平均Flow', 'higher_is_better': True, 'unit_type': 'money', 'unit_label': '金額'},
    {'label': '管理年資', 'score_col': '雷達_管理年資', 'raw_col': '平均管理年資', 'higher_is_better': True, 'unit_type': 'years', 'unit_label': '年'},
    {'label': '資產規模', 'score_col': '雷達_資產規模', 'raw_col': '平均MTNA', 'higher_is_better': True, 'unit_type': 'money', 'unit_label': 'MTNA'},
]

RADAR_LABELS = [m['label'] for m in RADAR_METRIC_CONFIG]
RADAR_THETA_LABELS = [f"{m['label']}<br>({m['unit_label']})" for m in RADAR_METRIC_CONFIG]


@st.cache_resource(show_spinner=False)
def build_tab6_return_sp500_scatter_data(df, sp500_df=None, manual_sp500_ann=0.10):
    """
    Tab6 Part 1：建立「基金月報酬率 vs S&P500 月報酬率」散點圖資料。
    每一個點代表一檔基金在某一個月份的觀測值。
    X 軸：基金月報酬率 mret
    Y 軸：S&P500 月報酬率 sp500_ret
    """
    if df is None or df.empty:
        return pd.DataFrame(), "無基金資料"

    scatter_needed_cols = [
        'caldt', 'mret', 'mgmt_name', 'mgr_name', 'crsp_fundno', 'fund_name',
        'net_flow', 'mtna', 'exp_ratio', 'tenure', 'turn_ratio'
    ]
    scatter_existing_cols = [c for c in scatter_needed_cols if c in df.columns]
    out = df.loc[:, scatter_existing_cols].copy(deep=False)
    out['caldt'] = pd.to_datetime(out['caldt'], errors='coerce')
    out = out.dropna(subset=['caldt', 'mret', 'mgmt_name'])

    out['month_key'] = out['caldt'].dt.to_period('M').astype(str)

    benchmark_source = "手動年化 S&P500 轉換為固定月報酬"
    if sp500_df is not None and not sp500_df.empty:
        b = sp500_df.copy()
        b['caldt'] = pd.to_datetime(b['caldt'], errors='coerce')
        b['month_key'] = b['caldt'].dt.to_period('M').astype(str)
        b = b.dropna(subset=['month_key', 'sp500_ret'])
        b = b.groupby('month_key', as_index=False)['sp500_ret'].mean()
        out = out.merge(b, on='month_key', how='left')
        benchmark_source = "上傳的 S&P500 月報酬 CSV"
    else:
        out['sp500_ret'] = (1 + manual_sp500_ann) ** (1 / 12) - 1

    manual_monthly = (1 + manual_sp500_ann) ** (1 / 12) - 1
    out['sp500_ret'] = pd.to_numeric(out['sp500_ret'], errors='coerce').fillna(manual_monthly)

    out['基金超額月報酬'] = out['mret'] - out['sp500_ret']
    out['基金絕對報酬'] = out['mret'].abs()
    out['S&P500絕對報酬'] = out['sp500_ret'].abs()
    if 'net_flow' in out.columns:
        out['Flow絕對值'] = pd.to_numeric(out['net_flow'], errors='coerce').abs()
    else:
        out['net_flow'] = np.nan
        out['Flow絕對值'] = np.nan

    for c in ['fund_name', 'crsp_fundno', 'mgr_name', 'mtna', 'exp_ratio', 'tenure', 'turn_ratio']:
        if c not in out.columns:
            out[c] = np.nan

    keep_cols = [
        'caldt', 'month_key', 'crsp_fundno', 'fund_name', 'mgmt_name', 'mgr_name',
        'mret', 'sp500_ret', '基金超額月報酬', '基金絕對報酬', 'S&P500絕對報酬',
        'net_flow', 'Flow絕對值', 'mtna', 'exp_ratio', 'tenure', 'turn_ratio'
    ]
    out = out.loc[:, keep_cols].dropna(subset=['mret', 'sp500_ret'])
    return out, benchmark_source


@st.cache_resource(show_spinner=False)
def build_tab6_part2_feature_tables(selected_scatter_df):
    """Tab6 Part 2：依 Part1 目前高亮/框選資料建立月資料特徵與基金層級可計算因子。"""
    if selected_scatter_df is None or selected_scatter_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    d = selected_scatter_df.copy(deep=False)
    for c in ['mret', 'sp500_ret', '基金超額月報酬', 'net_flow', 'Flow絕對值', 'mtna', 'exp_ratio', 'tenure', 'turn_ratio']:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')

    monthly = pd.DataFrame(index=d.index)
    monthly['__tab6_row_id'] = d.get('tab6_row_id', pd.Series(d.index, index=d.index))
    monthly['基金家族'] = d.get('mgmt_name')
    monthly['基金代號'] = d.get('crsp_fundno')
    monthly['基金經理人'] = d.get('mgr_name')
    monthly['基金月報酬率'] = d.get('mret')
    monthly['S&P500月報酬率'] = d.get('sp500_ret')
    monthly['基金超額月報酬'] = d.get('基金超額月報酬', d.get('mret') - d.get('sp500_ret'))
    monthly['基金月報酬絕對值'] = d.get('mret').abs()
    monthly['S&P500月報酬絕對值'] = d.get('sp500_ret').abs()
    monthly['Net Flow'] = d.get('net_flow')
    monthly['Flow絕對值'] = d.get('Flow絕對值', d.get('net_flow').abs())
    monthly['MTNA資產規模'] = d.get('mtna')
    monthly['費用率'] = d.get('exp_ratio')
    monthly['管理年資'] = d.get('tenure')
    monthly['換手率'] = d.get('turn_ratio')
    monthly['觀測年份'] = pd.to_datetime(d.get('caldt'), errors='coerce').dt.year
    monthly = monthly.replace([np.inf, -np.inf], np.nan)

    rows = []
    if 'crsp_fundno' not in d.columns:
        return monthly, pd.DataFrame()

    for fund_id, g in d.groupby('crsp_fundno'):
        g = g.sort_values('caldt') if 'caldt' in g.columns else g.copy()
        r = pd.to_numeric(g['mret'], errors='coerce').dropna()
        rb = g[['mret', 'sp500_ret']].dropna() if {'mret','sp500_ret'}.issubset(g.columns) else pd.DataFrame()
        r_aligned = pd.to_numeric(rb['mret'], errors='coerce') if not rb.empty else r
        b_aligned = pd.to_numeric(rb['sp500_ret'], errors='coerce') if not rb.empty else pd.Series(dtype=float)

        obs = len(r)
        if obs == 0:
            continue
        mean_m = r.mean()
        ann_ret = safe_annualized_return_from_monthly_mean(mean_m) if pd.notna(mean_m) else np.nan
        ann_vol = r.std() * np.sqrt(12) if obs > 1 else np.nan
        sharpe = (ann_ret - 0.01) / ann_vol if pd.notna(ann_vol) and ann_vol > 0 else np.nan
        mdd = max_drawdown_from_monthly(r)
        downside = r[r < 0]
        downside_vol = downside.std() * np.sqrt(12) if len(downside) > 1 else np.nan
        sortino = (ann_ret - 0.01) / downside_vol if pd.notna(downside_vol) and downside_vol > 0 else np.nan
        cumulative_ret = safe_compound_return(r)
        win_rate = (r > 0).mean()
        skewness = safe_skewness(r) if obs >= 3 else np.nan
        kurt = safe_kurtosis(r) if obs >= 4 else np.nan
        var95 = np.percentile(r, 5) if obs >= 2 else np.nan
        cvar95 = r[r <= var95].mean() if obs >= 2 and (r <= var95).any() else np.nan

        if not r_aligned.empty and len(r_aligned) == len(b_aligned) and len(r_aligned) > 1:
            excess = r_aligned - b_aligned
            ann_excess = safe_annualized_return_from_monthly_mean(excess.mean())
            tracking_error = excess.std() * np.sqrt(12)
            info_ratio = ann_excess / tracking_error if pd.notna(tracking_error) and tracking_error > 0 else np.nan
            beat_sp500_rate = (r_aligned > b_aligned).mean()
            beta = safe_beta(r_aligned, b_aligned)
            alpha_monthly = r_aligned.mean() - beta * b_aligned.mean() if pd.notna(beta) else np.nan
            alpha_ann = safe_annualized_return_from_monthly_mean(alpha_monthly) if pd.notna(alpha_monthly) else np.nan
            upside_capture = r_aligned[b_aligned > 0].mean() / b_aligned[b_aligned > 0].mean() if (b_aligned > 0).any() and b_aligned[b_aligned > 0].mean() != 0 else np.nan
            downside_capture = r_aligned[b_aligned < 0].mean() / b_aligned[b_aligned < 0].mean() if (b_aligned < 0).any() and b_aligned[b_aligned < 0].mean() != 0 else np.nan
            corr_sp500 = safe_corr(r_aligned, b_aligned)
        else:
            ann_excess = tracking_error = info_ratio = beat_sp500_rate = beta = alpha_ann = upside_capture = downside_capture = corr_sp500 = np.nan

        rows.append({
            '基金代號': fund_id,
            '基金家族': g['mgmt_name'].dropna().iloc[0] if 'mgmt_name' in g.columns and g['mgmt_name'].notna().any() else np.nan,
            '基金名稱': g['fund_name'].dropna().iloc[0] if 'fund_name' in g.columns and g['fund_name'].notna().any() else np.nan,
            '基金經理人': g['mgr_name'].dropna().mode().iloc[0] if 'mgr_name' in g.columns and g['mgr_name'].notna().any() and not g['mgr_name'].dropna().mode().empty else (g['mgr_name'].dropna().iloc[0] if 'mgr_name' in g.columns and g['mgr_name'].notna().any() else np.nan),
            '觀測月數': obs,
            '平均月報酬率': mean_m,
            '年化報酬率': ann_ret,
            '累積報酬率': cumulative_ret,
            '年化波動度': ann_vol,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': mdd,
            '月勝率': win_rate,
            '打敗S&P500月比例': beat_sp500_rate,
            '年化超額報酬': ann_excess,
            'Tracking Error': tracking_error,
            'Information Ratio': info_ratio,
            'Beta vs S&P500': beta,
            'Alpha年化近似': alpha_ann,
            'Upside Capture': upside_capture,
            'Downside Capture': downside_capture,
            '與S&P500相關係數': corr_sp500,
            'VaR 95%月度': var95,
            'CVaR 95%月度': cvar95,
            '偏度 Skewness': skewness,
            '峰度 Kurtosis': kurt,
            '平均MTNA': g['mtna'].mean() if 'mtna' in g.columns else np.nan,
            '平均Net Flow': g['net_flow'].mean() if 'net_flow' in g.columns else np.nan,
            '累積Net Flow': g['net_flow'].sum() if 'net_flow' in g.columns else np.nan,
            '平均費用率': g['exp_ratio'].mean() if 'exp_ratio' in g.columns else np.nan,
            '平均管理年資': g['tenure'].mean() if 'tenure' in g.columns else np.nan,
            '平均換手率': g['turn_ratio'].mean() if 'turn_ratio' in g.columns else np.nan,
        })

    fund_factors = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    return monthly, fund_factors


@st.cache_resource(show_spinner=False)
def build_tab6_manager_radar_base(raw_df):
    """Tab6 Part 4：依目前 Part 3 連動樣本建立經理人雷達圖基準表。"""
    if raw_df is None or raw_df.empty or 'mgr_name' not in raw_df.columns:
        return pd.DataFrame()

    d = raw_df.copy()
    for c in ['mret', 'sp500_ret', '基金超額月報酬', 'net_flow', 'mtna', 'exp_ratio', 'tenure', 'turn_ratio']:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    if '基金超額月報酬' not in d.columns and {'mret', 'sp500_ret'}.issubset(d.columns):
        d['基金超額月報酬'] = d['mret'] - d['sp500_ret']

    rows = []
    for mgr, g in d.assign(_manager=d['mgr_name'].astype('string').fillna('Unknown Manager').astype(str)).groupby('_manager'):
        r = pd.to_numeric(g.get('mret'), errors='coerce').dropna()
        if r.empty:
            continue
        sp = pd.to_numeric(g.get('sp500_ret'), errors='coerce') if 'sp500_ret' in g.columns else pd.Series(index=g.index, dtype=float)
        aligned = pd.DataFrame({'r': pd.to_numeric(g.get('mret'), errors='coerce'), 'sp': sp}).dropna()
        mean_m = r.mean()
        ann_ret = safe_annualized_return_from_monthly_mean(mean_m) if pd.notna(mean_m) else np.nan
        ann_vol = r.std() * np.sqrt(12) if len(r) > 1 else np.nan
        sharpe = (ann_ret - 0.01) / ann_vol if pd.notna(ann_vol) and ann_vol > 0 else np.nan
        mdd = max_drawdown_from_monthly(r)
        excess_m = aligned['r'].mean() - aligned['sp'].mean() if len(aligned) else np.nan
        beat_rate = (aligned['r'] > aligned['sp']).mean() if len(aligned) else np.nan
        rows.append({
            '基金經理人': mgr,
            '觀測點數': len(g),
            '基金數': g['crsp_fundno'].nunique() if 'crsp_fundno' in g.columns else np.nan,
            '基金家族數': g['mgmt_name'].nunique() if 'mgmt_name' in g.columns else np.nan,
            '平均月報酬率': mean_m,
            '年化報酬率': ann_ret,
            '平均超額月報酬': excess_m,
            '打敗S&P500比例': beat_rate,
            '年化波動度': ann_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': mdd,
            '平均費用率': g['exp_ratio'].mean() if 'exp_ratio' in g.columns else np.nan,
            '平均Flow': g['net_flow'].mean() if 'net_flow' in g.columns else np.nan,
            '平均MTNA': g['mtna'].mean() if 'mtna' in g.columns else np.nan,
            '平均管理年資': g['tenure'].mean() if 'tenure' in g.columns else np.nan,
            '平均換手率': g['turn_ratio'].mean() if 'turn_ratio' in g.columns else np.nan,
        })
    base = pd.DataFrame(rows)
    if base.empty:
        return base

    for metric in RADAR_METRIC_CONFIG:
        label = metric['label']
        col = metric['raw_col']
        higher = metric['higher_is_better']
        if col in base.columns:
            base[f'雷達_{label}'] = safe_percentile_score(base[col], higher_is_better=higher)
        else:
            base[f'雷達_{label}'] = 0.5
    return base


def make_tab6_manager_radar_record(base_df, manager_name, source_label, source_key):
    """把單一經理人的當下雷達資料轉成可放入 session_state 的記憶紀錄。"""
    if base_df is None or base_df.empty:
        return None
    row = base_df[base_df['基金經理人'].astype(str) == str(manager_name)]
    if row.empty:
        return None
    row = row.iloc[0]
    scores = {
        metric['label']: (
            float(row.get(f"雷達_{metric['label']}", 0.5))
            if pd.notna(row.get(f"雷達_{metric['label']}", np.nan))
            else 0.5
        )
        for metric in RADAR_METRIC_CONFIG
    }
    raw_cols = ['觀測點數', '基金數', '基金家族數', '年化報酬率', '平均超額月報酬', '打敗S&P500比例', '年化波動度', 'Sharpe Ratio', 'Max Drawdown', '平均費用率', '平均Flow', '平均MTNA', '平均管理年資', '平均換手率']
    raw = {}
    for c in raw_cols:
        val = row.get(c, np.nan)
        if isinstance(val, (np.integer, np.floating)):
            val = float(val)
        raw[c] = val
    return {
        'record_key': f"{manager_name}__{source_key}",
        '基金經理人': str(manager_name),
        '來源': source_label,
        '雷達分數': scores,
        '原始指標': raw,
    }


HORIZON_DISPLAY_OPTIONS = ['月', '1年', '3年', '5年']
HORIZON_DISPLAY_TO_KEY = {'月': '月資料', '1年': '1年', '3年': '3年', '5年': '5年'}
HORIZON_KEY_TO_DISPLAY = {v: k for k, v in HORIZON_DISPLAY_TO_KEY.items()}

PLOT_COLORS = {
    'A_dark': 'rgba(13, 71, 161, 0.95)',
    'B_dark': 'rgba(191, 54, 12, 0.92)'
}


def _ensure_string_series(s, fill='Unknown Manager'):
    return s.astype('string').fillna(fill).astype(str)


@st.cache_resource(show_spinner=False)
def build_tab6_horizon_scatter_tables(df, sp500_df=None, manual_sp500_ann=0.10):
    """Precompute Part1 scatter data for monthly and trailing 1/3/5-year windows.

    Each row remains a fund-month observation.  For 1/3/5-year views, the point is
    the trailing-window annualized fund return versus trailing-window annualized
    S&P500 return ending at that month.
    """
    base, benchmark_source = build_tab6_return_sp500_scatter_data(df, sp500_df=sp500_df, manual_sp500_ann=manual_sp500_ann)
    if base is None or base.empty:
        return {}, benchmark_source

    base = base.reset_index(drop=True)
    base['tab6_row_id'] = np.arange(len(base), dtype=np.int64)
    base['caldt'] = pd.to_datetime(base['caldt'], errors='coerce')
    base = base.dropna(subset=['caldt', 'mret', 'sp500_ret'])
    base = base.sort_values(['crsp_fundno', 'caldt']).reset_index(drop=True)
    base['tab6_row_id'] = np.arange(len(base), dtype=np.int64)
    base['時間色彩'] = base['caldt'].dt.year + (base['caldt'].dt.month - 1) / 12.0

    safe_mret = pd.to_numeric(base['mret'], errors='coerce').where(lambda x: x > -0.999999)
    safe_sp = pd.to_numeric(base['sp500_ret'], errors='coerce').where(lambda x: x > -0.999999)
    base['_log_mret'] = np.log1p(safe_mret)
    base['_log_sp500'] = np.log1p(safe_sp)

    tables = {}
    monthly = base.copy(deep=False)
    monthly['period_label'] = '月資料'
    monthly['x_ret'] = monthly['sp500_ret'].astype('float32')
    monthly['y_ret'] = monthly['mret'].astype('float32')
    monthly['基金超額月報酬'] = (monthly['y_ret'] - monthly['x_ret']).astype('float32')
    tables['月資料'] = monthly.drop(columns=['_log_mret', '_log_sp500'], errors='ignore')

    for label, window in [('1年', 12), ('3年', 36), ('5年', 60)]:
        d = base.copy(deep=False)
        g = d.groupby('crsp_fundno', sort=False, observed=True)
        roll_sum_fund = g['_log_mret'].rolling(window=window, min_periods=max(3, int(window * 0.7))).sum().reset_index(level=0, drop=True)
        roll_count_fund = g['_log_mret'].rolling(window=window, min_periods=max(3, int(window * 0.7))).count().reset_index(level=0, drop=True)
        bench_by_month = d[['month_key', '_log_sp500']].drop_duplicates('month_key').sort_values('month_key')
        bench_by_month['_sp_roll_sum'] = bench_by_month['_log_sp500'].rolling(window=window, min_periods=max(3, int(window * 0.7))).sum()
        bench_by_month['_sp_roll_count'] = bench_by_month['_log_sp500'].rolling(window=window, min_periods=max(3, int(window * 0.7))).count()
        sp_map_sum = dict(zip(bench_by_month['month_key'], bench_by_month['_sp_roll_sum']))
        sp_map_count = dict(zip(bench_by_month['month_key'], bench_by_month['_sp_roll_count']))
        sp_sum = d['month_key'].map(sp_map_sum)
        sp_count = d['month_key'].map(sp_map_count)

        ann_factor_fund = 12.0 / roll_count_fund.replace(0, np.nan)
        ann_factor_sp = 12.0 / sp_count.replace(0, np.nan)
        d = d.copy()
        d['x_ret'] = np.expm1(sp_sum * ann_factor_sp).astype('float32')
        d['y_ret'] = np.expm1(roll_sum_fund * ann_factor_fund).astype('float32')
        d['sp500_ret'] = d['x_ret']
        d['mret'] = d['y_ret']
        d['基金超額月報酬'] = (d['y_ret'] - d['x_ret']).astype('float32')
        d['period_label'] = label
        d['觀測月數_視窗'] = roll_count_fund.astype('float32')
        d = d.dropna(subset=['x_ret', 'y_ret']).replace([np.inf, -np.inf], np.nan).dropna(subset=['x_ret', 'y_ret'])
        tables[label] = d.drop(columns=['_log_mret', '_log_sp500'], errors='ignore')

    return tables, benchmark_source


@st.cache_resource(show_spinner=False)
def build_tab6_family_feature_table(scope_df):
    """Build family-level features for Part2."""
    if scope_df is None or scope_df.empty or 'mgmt_name' not in scope_df.columns:
        return pd.DataFrame()
    d = scope_df.copy(deep=False)
    for c in ['mret', 'sp500_ret', '基金超額月報酬', 'net_flow', 'mtna', 'exp_ratio', 'turn_ratio', 'tenure']:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    rows = []
    for fam, g in d.groupby('mgmt_name', observed=True):
        rows.append({
            '基金家族': str(fam),
            '觀測點數': int(len(g)),
            '基金數': int(g['crsp_fundno'].nunique()) if 'crsp_fundno' in g.columns else np.nan,
            '經理人數': int(g['mgr_name'].nunique()) if 'mgr_name' in g.columns else np.nan,
            '家族平均報酬率': g['mret'].mean(),
            '家族平均超額報酬': g['基金超額月報酬'].mean() if '基金超額月報酬' in g.columns else np.nan,
            '家族平均費用率': g['exp_ratio'].mean() if 'exp_ratio' in g.columns else np.nan,
            '家族平均規模': g['mtna'].mean() if 'mtna' in g.columns else np.nan,
            '家族平均淨申購': g['net_flow'].mean() if 'net_flow' in g.columns else np.nan,
            '家族累積淨申購': g['net_flow'].sum() if 'net_flow' in g.columns else np.nan,
            '家族平均換手率': g['turn_ratio'].mean() if 'turn_ratio' in g.columns else np.nan,
            '家族平均管理年資': g['tenure'].mean() if 'tenure' in g.columns else np.nan,
        })
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def extract_selection_bbox(event):
    """Extract a bounding box from Streamlit Plotly selection event."""
    if event is None:
        return None
    try:
        boxes = event.selection.box if event.selection else []
        if boxes:
            b = boxes[0]
            if isinstance(b, dict):
                if 'x' in b and 'y' in b:
                    xs, ys = b['x'], b['y']
                    return {'x0': float(min(xs)), 'x1': float(max(xs)), 'y0': float(min(ys)), 'y1': float(max(ys))}
                vals_x = [b.get('x0'), b.get('x1')]
                vals_y = [b.get('y0'), b.get('y1')]
                if all(v is not None for v in vals_x + vals_y):
                    return {'x0': float(min(vals_x)), 'x1': float(max(vals_x)), 'y0': float(min(vals_y)), 'y1': float(max(vals_y))}
    except Exception:
        pass
    xs, ys = [], []
    try:
        points = event.selection.points if event.selection else []
    except Exception:
        points = []
    for p in points:
        try:
            xs.append(float(p.get('x') if isinstance(p, dict) else p.x))
            ys.append(float(p.get('y') if isinstance(p, dict) else p.y))
        except Exception:
            pass
    if xs and ys:
        return {'x0': min(xs), 'x1': max(xs), 'y0': min(ys), 'y1': max(ys)}
    return None


def mask_from_box(df, box, x_col='x_ret', y_col='y_ret'):
    if box is None or df is None or df.empty:
        return pd.Series(False, index=df.index if df is not None else [])
    x0, x1 = sorted([float(box['x0']), float(box['x1'])])
    y0, y1 = sorted([float(box['y0']), float(box['y1'])])
    return df[x_col].between(x0, x1, inclusive='both') & df[y_col].between(y0, y1, inclusive='both')


def nice_axis_range(s, q_low=0.005, q_high=0.995, pad_ratio=0.08):
    vals = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return None
    lo = float(vals.quantile(q_low))
    hi = float(vals.quantile(q_high))
    if lo == hi:
        pad = max(abs(lo) * 0.1, 0.01)
        return [lo - pad, hi + pad]
    pad = (hi - lo) * pad_ratio
    return [lo - pad, hi + pad]


def shared_hist_edges(dataframes, col, bins=30):
    vals = []
    for d in dataframes:
        if d is not None and not d.empty and col in d.columns:
            v = pd.to_numeric(d[col], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
            if not v.empty:
                vals.append(v)
    if not vals:
        return None
    x = pd.concat(vals, ignore_index=True)
    lo, hi = float(x.quantile(0.005)), float(x.quantile(0.995))
    if lo == hi:
        lo, hi = float(x.min()), float(x.max())
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return np.linspace(lo, hi, bins + 1)


def make_manager_count_data(raw_df):
    if raw_df is None or raw_df.empty or 'mgr_name' not in raw_df.columns:
        return pd.DataFrame(columns=['基金經理人', '次數'])
    return (raw_df.assign(_manager=_ensure_string_series(raw_df['mgr_name']))
            .groupby('_manager').size().reset_index(name='次數')
            .rename(columns={'_manager': '基金經理人'})
            .sort_values('次數', ascending=False))


def build_tab6_part2_feature_tables_v32(selected_scatter_df):
    """Part2 tables with added family-level monthly features."""
    monthly, fund_factors = build_tab6_part2_feature_tables(selected_scatter_df)
    if selected_scatter_df is None or selected_scatter_df.empty or monthly is None or monthly.empty:
        return monthly, fund_factors
    d = selected_scatter_df.copy(deep=False)
    if 'mgmt_name' in d.columns:
        for c in ['mret', 'exp_ratio', 'mtna', 'net_flow']:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors='coerce')
        fam_stats = d.groupby('mgmt_name', observed=True).agg(
            家族平均報酬率=('mret', 'mean'),
            家族平均費用率=('exp_ratio', 'mean'),
            家族平均規模=('mtna', 'mean'),
            家族平均淨申購=('net_flow', 'mean'),
        )
        fam_key = d['mgmt_name']
        monthly['家族平均報酬率'] = fam_key.map(fam_stats['家族平均報酬率'])
        monthly['家族平均費用率'] = fam_key.map(fam_stats['家族平均費用率'])
        monthly['家族平均規模'] = fam_key.map(fam_stats['家族平均規模'])
        monthly['家族平均淨申購'] = fam_key.map(fam_stats['家族平均淨申購'])
    return monthly.replace([np.inf, -np.inf], np.nan), fund_factors


def make_part1_scatter_v32(df, period_label, boxes=None, zoom_to_dense=True):
    """Part1 scatter for A/B selection.

    v36 修正重點：設定 A 之後，不再把未選點轉成黑白/灰色。
    也就是不管 A/B 是否已設定，底圖永遠維持「全部點 = 年份色彩」，
    只用藍框/橘框表示 A/B 選區；這樣使用者可以很清楚地繼續框選 B。
    """
    fig = go.Figure()

    cmin = float(df['時間色彩'].min()) if '時間色彩' in df.columns else None
    cmax = float(df['時間色彩'].max()) if '時間色彩' in df.columns else None

    fig.add_trace(go.Scattergl(
        x=df['x_ret'], y=df['y_ret'], mode='markers',
        marker=dict(
            color=df.get('時間色彩', None),
            colorscale='Viridis',
            cmin=cmin,
            cmax=cmax,
            size=4,
            opacity=0.82,
            line=dict(width=0),
            showscale=True,
            colorbar=dict(title='年份', len=0.48, thickness=12)
        ),
        name='全部點（顏色=時間）',
        hoverinfo='skip',
        showlegend=True
    ))

    for name, box in (boxes or {}).items():
        if box is None:
            continue
        x0, x1 = sorted([box['x0'], box['x1']])
        y0, y1 = sorted([box['y0'], box['y1']])
        if name == 'A':
            color = 'rgba(30,136,229,1)'
            fill = 'rgba(30,136,229,0.035)'
            label = '選區 A'
        else:
            color = 'rgba(255,112,67,1)'
            fill = 'rgba(255,112,67,0.035)'
            label = '選區 B'
        fig.add_shape(
            type='rect', x0=x0, x1=x1, y0=y0, y1=y1,
            xref='x', yref='y',
            line=dict(color=color, width=3),
            fillcolor=fill,
            layer='above'
        )
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(color=color, symbol='square-open', size=12),
            name=label,
            hoverinfo='skip',
            showlegend=True
        ))

    fig.add_vline(x=0, line_dash='dash', line_color='gray')
    fig.add_hline(y=0, line_dash='dash', line_color='gray')
    rng_min = np.nanmin([df['x_ret'].min(), df['y_ret'].min()])
    rng_max = np.nanmax([df['x_ret'].max(), df['y_ret'].max()])
    if np.isfinite(rng_min) and np.isfinite(rng_max):
        fig.add_trace(go.Scatter(
            x=[rng_min, rng_max], y=[rng_min, rng_max], mode='lines',
            line=dict(dash='dot', color='black'),
            name='基金報酬 = S&P500',
            hoverinfo='skip'
        ))

    x_title = 'S&P500 月報酬率' if period_label == '月資料' else f'S&P500 {period_label}年化報酬率'
    y_title = '基金月報酬率' if period_label == '月資料' else f'基金 {period_label}年化報酬率'
    fig.update_layout(
        title=f'Part 1：{x_title} × {y_title}',
        xaxis_title=x_title,
        yaxis_title=y_title,
        xaxis_tickformat='.1%',
        yaxis_tickformat='.1%',
        dragmode='select',
        hovermode=False,
        height=560,
        margin=dict(l=50, r=30, t=70, b=55),
        uirevision=f'p1_v36_{period_label}'
    )
    if zoom_to_dense:
        xr = nice_axis_range(df['x_ret'])
        yr = nice_axis_range(df['y_ret'])
        if xr:
            fig.update_xaxes(range=xr)
        if yr:
            fig.update_yaxes(range=yr)
    return fig

def extract_x_range_from_hist_event(event):
    if event is None:
        return None
    try:
        boxes = event.selection.box if event.selection else []
        if boxes:
            b = boxes[0]
            if isinstance(b, dict):
                if 'x' in b and isinstance(b['x'], (list, tuple)) and len(b['x']) >= 2:
                    return (float(min(b['x'])), float(max(b['x'])))
                vals = [b.get('x0'), b.get('x1')]
                if vals[0] is not None and vals[1] is not None:
                    return (float(min(vals)), float(max(vals)))
    except Exception:
        pass
    xs = []
    try:
        points = event.selection.points if event.selection else []
    except Exception:
        points = []
    for p in points:
        try:
            xs.append(float(p.get('x') if isinstance(p, dict) else p.x))
        except Exception:
            pass
    return (min(xs), max(xs)) if xs else None


def get_level_tables(raw_df):
    monthly, fund = build_tab6_part2_feature_tables_v32(raw_df)
    family = build_tab6_family_feature_table(raw_df)
    return monthly, fund, family


def get_table_for_level(raw_df, level):
    monthly, fund, family = get_level_tables(raw_df)
    if level == '月資料特徵':
        return monthly
    if level == '基金層級因子':
        return fund
    return family


def regions_signature(regions):
    if not regions:
        return 'none'
    return repr([(r.get('level'), r.get('feature'), tuple(np.round(r.get('x_range', (np.nan, np.nan)), 10))) for r in regions])


def filter_raw_by_regions(raw_df, level, regions):
    """Union of Part2 histogram ranges, mapped back to raw rows.

    level can be one of the three Part2 levels, or None / '全部層級'.
    When level is None, ranges selected from monthly, fund-level, and
    family-level histograms are all applied together.  This is what makes
    the three Part2 blocks move together after the user selects any one of
    the blocks.
    """
    if raw_df is None or raw_df.empty or not regions:
        return pd.DataFrame(columns=raw_df.columns if raw_df is not None else [])
    selected_parts = []
    use_all_levels = level is None or level == '全部層級'
    table_cache = {}
    for reg in regions:
        if not reg:
            continue
        reg_level = reg.get('level')
        if not use_all_levels and reg_level != level:
            continue
        if reg_level not in ['月資料特徵', '基金層級因子', '基金家族層級特徵']:
            continue
        feature = reg.get('feature')
        xr = reg.get('x_range')
        if feature is None or xr is None:
            continue
        if reg_level not in table_cache:
            table_cache[reg_level] = get_table_for_level(raw_df, reg_level)
        table = table_cache[reg_level]
        if table is None or table.empty or feature not in table.columns:
            continue
        lo, hi = sorted([float(xr[0]), float(xr[1])])
        vals = pd.to_numeric(table[feature], errors='coerce')
        picked = table.loc[vals.between(lo, hi, inclusive='both')]
        if picked.empty:
            continue
        if reg_level == '月資料特徵':
            ids = picked['__tab6_row_id'].dropna().astype(int).unique() if '__tab6_row_id' in picked.columns else []
            part = raw_df.loc[raw_df['tab6_row_id'].isin(ids)] if len(ids) else pd.DataFrame(columns=raw_df.columns)
        elif reg_level == '基金層級因子':
            fund_ids = picked['基金代號'].dropna().unique() if '基金代號' in picked.columns else []
            part = raw_df.loc[raw_df['crsp_fundno'].isin(fund_ids)] if len(fund_ids) else pd.DataFrame(columns=raw_df.columns)
        else:
            fams = picked['基金家族'].dropna().astype(str).unique() if '基金家族' in picked.columns else []
            part = raw_df.loc[raw_df['mgmt_name'].astype(str).isin(fams)] if len(fams) else pd.DataFrame(columns=raw_df.columns)
        if not part.empty:
            selected_parts.append(part)
    if not selected_parts:
        return pd.DataFrame(columns=raw_df.columns)
    out = pd.concat(selected_parts, ignore_index=False)
    if 'tab6_row_id' in out.columns:
        out = out.drop_duplicates('tab6_row_id')
    else:
        out = out.drop_duplicates()
    return out



def make_selectable_hist_v32(all_a, all_b, sel_a, sel_b, col, title, compare_mode=False, cumulative=False, direct_regions=None):
    """Histogram drawing with v29-like interaction but stronger colors.

    Before any Part2 selection, bars keep visible colors.
    After a selection exists, the full distribution becomes pale background,
    linked selected samples are light blue/orange, and the currently framed
    feature/range is drawn in dark blue/orange.
    """
    fig = go.Figure()
    edges = shared_hist_edges([all_a, all_b if compare_mode else None, sel_a, sel_b if compare_mode else None], col, bins=30)
    if edges is None:
        fig.update_layout(title=title, height=270)
        return fig

    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges) * 0.92

    def counts_for(data):
        if data is None or data.empty or col not in data.columns:
            return None
        x = pd.to_numeric(data[col], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if x.empty:
            return None
        counts, _ = np.histogram(x, bins=edges)
        if cumulative:
            counts = np.cumsum(counts)
        return counts

    def add_bar_counts(counts, name, color, opacity=1.0, line_color='rgba(255,255,255,0)', line_w=0.15):
        if counts is None:
            return
        fig.add_trace(go.Bar(
            x=centers, y=counts, width=widths, name=name,
            marker=dict(color=color, line=dict(color=line_color, width=line_w)),
            opacity=opacity, hoverinfo='skip'
        ))

    direct = bool(direct_regions)
    has_sel_a = sel_a is not None and not sel_a.empty
    has_sel_b = compare_mode and sel_b is not None and not sel_b.empty
    has_any_selection = direct or has_sel_a or has_sel_b

    if has_any_selection:
        a_bg_color = 'rgba(91, 155, 213, 0.24)'
        b_bg_color = 'rgba(255, 160, 122, 0.22)'
        a_bg_name = 'A 全部分布（淡色背景）'
        b_bg_name = 'B 全部分布（淡色背景）'
    else:
        a_bg_color = 'rgba(91, 155, 213, 0.72)'
        b_bg_color = 'rgba(255, 112, 67, 0.62)'
        a_bg_name = 'A 分布'
        b_bg_name = 'B 分布'

    add_bar_counts(counts_for(all_a), a_bg_name, a_bg_color, opacity=1.0,
                   line_color='rgba(91,155,213,0.72)', line_w=0.2)
    if compare_mode:
        add_bar_counts(counts_for(all_b), b_bg_name, b_bg_color, opacity=1.0,
                       line_color='rgba(255,112,67,0.72)', line_w=0.2)

    if has_sel_a:
        add_bar_counts(
            counts_for(sel_a),
            'A 框選樣本' if direct else 'A 連動樣本',
            PLOT_COLORS['A_dark'] if direct else 'rgba(91, 155, 213, 0.72)',
            opacity=0.96,
            line_color='rgba(8,48,107,1)' if direct else 'rgba(91,155,213,0.95)',
            line_w=0.3
        )
    if compare_mode and has_sel_b:
        add_bar_counts(
            counts_for(sel_b),
            'B 框選樣本' if direct else 'B 連動樣本',
            PLOT_COLORS['B_dark'] if direct else 'rgba(255, 112, 67, 0.62)',
            opacity=0.92,
            line_color='rgba(191,54,12,1)' if direct else 'rgba(255,112,67,0.95)',
            line_w=0.3
        )

    if direct and ((not has_sel_a) and (not compare_mode or not has_sel_b)):
        for reg in direct_regions or []:
            xr = reg.get('x_range')
            if xr is None:
                continue
            lo, hi = sorted([float(xr[0]), float(xr[1])])
            for source, color, nm in [
                (all_a, PLOT_COLORS['A_dark'], 'A 框選區'),
                (all_b if compare_mode else None, PLOT_COLORS['B_dark'], 'B 框選區')
            ]:
                base_counts = counts_for(source)
                if base_counts is None:
                    continue
                sx, sy, sw = [], [], []
                for left, right, c in zip(edges[:-1], edges[1:], base_counts):
                    ov0, ov1 = max(left, lo), min(right, hi)
                    if ov1 > ov0 and c > 0:
                        sx.append((ov0 + ov1) / 2)
                        sy.append(c)
                        sw.append((ov1 - ov0) * 0.92)
                if sx:
                    fig.add_trace(go.Bar(
                        x=sx, y=sy, width=sw, name=nm,
                        marker=dict(color=color, line=dict(color=color, width=0.35)),
                        opacity=0.96, hoverinfo='skip'
                    ))

    fig.update_layout(
        title=title,
        height=280,
        barmode='overlay',
        hovermode=False,
        dragmode='select',
        margin=dict(l=35, r=10, t=45, b=35),
        xaxis_title=col,
        yaxis_title='累積次數' if cumulative else '次數',
        uirevision=f'hist_v37_{col}',
        plot_bgcolor='white',
        legend=dict(orientation='h', y=-0.28, font=dict(size=9))
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.06)', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.06)', zeroline=False)
    return fig

def render_selectable_hist_grid_v32(data_a, data_b, raw_a, raw_b, level, columns, compare_mode=False, cumulative=False):
    clean_cols = [c for c in columns if c in data_a.columns and pd.to_numeric(data_a[c], errors='coerce').replace([np.inf, -np.inf], np.nan).notna().sum() > 0]
    if not clean_cols:
        st.warning('目前沒有可顯示的 histogram 欄位。')
        return
    pending = st.session_state.get('p2_pending_regions', []) or []
    applied = st.session_state.get('p2_applied_regions', []) or []
    latest = st.session_state.get('p2_latest_region')
    regions_for_visual = []
    if latest: regions_for_visual.append(latest)
    regions_for_visual += pending + applied
    active_regions = pending + ([latest] if latest else []) + applied
    sel_raw_a = filter_raw_by_regions(raw_a, None, active_regions) if active_regions else pd.DataFrame(columns=raw_a.columns)
    sel_raw_b = filter_raw_by_regions(raw_b, None, active_regions) if compare_mode and active_regions else pd.DataFrame(columns=raw_b.columns if raw_b is not None else [])
    sel_a = get_table_for_level(sel_raw_a, level) if not sel_raw_a.empty else pd.DataFrame(columns=data_a.columns)
    sel_b = get_table_for_level(sel_raw_b, level) if compare_mode and not sel_raw_b.empty else pd.DataFrame(columns=data_b.columns if data_b is not None else [])

    supports_selection = 'on_select' in inspect.signature(st.plotly_chart).parameters
    key_sig = str(abs(hash(regions_signature(regions_for_visual))))
    for start in range(0, len(clean_cols), 4):
        ui_cols = st.columns(4)
        for i, col in enumerate(clean_cols[start:start+4]):
            with ui_cols[i]:
                direct_regions = [r for r in regions_for_visual if r.get('level') == level and r.get('feature') == col]
                fig = make_selectable_hist_v32(data_a, data_b, sel_a, sel_b, col, col, compare_mode=compare_mode,
                                               cumulative=cumulative, direct_regions=direct_regions)
                if supports_selection:
                    ev = st.plotly_chart(fig, on_select='rerun', selection_mode=('box', 'lasso'), width='stretch',
                                         config={'displaylogo': False, 'modeBarButtonsToAdd': ['select2d', 'lasso2d'], 'modeBarButtonsToRemove': ['hoverClosestCartesian', 'hoverCompareCartesian']},
                                         key=f'p2_hist_v33_{level}_{col}_{key_sig}_{st.session_state.get("p2_hist_nonce", 0)}')
                    xr = extract_x_range_from_hist_event(ev)
                    if xr is not None:
                        new_region = {'level': level, 'feature': col, 'x_range': (float(xr[0]), float(xr[1]))}
                        new_sig = repr(new_region)
                        if st.session_state.get('p2_latest_region_sig') != new_sig:
                            st.session_state['p2_latest_region'] = new_region
                            st.session_state['p2_latest_region_sig'] = new_sig
                            st.rerun()
                else:
                    st.plotly_chart(fig, width='stretch', config={'displaylogo': False})


def make_manager_select_chart_v32(count_a, count_b=None, compare_mode=False):
    """Part3 manager bar chart.

    Two-region mode:
    - Blue  = manager appears only in region A after Part2 filtering.
    - Orange = manager appears only in region B after Part2 filtering.
    - Red = manager appears in both A and B.
    The bar height is the total appearance count across the relevant region(s), while
    the returned table keeps A/B counts separately for interpretation.
    """
    if compare_mode:
        if count_a is None or count_a.empty:
            a = pd.DataFrame(columns=['基金經理人', '選區A次數'])
        else:
            a = count_a.rename(columns={'次數': '選區A次數'})[['基金經理人', '選區A次數']]

        if count_b is None or count_b.empty:
            b = pd.DataFrame(columns=['基金經理人', '選區B次數'])
        else:
            b = count_b.rename(columns={'次數': '選區B次數'})[['基金經理人', '選區B次數']]

        data = a.merge(b, on='基金經理人', how='outer')
        if data.empty:
            data = pd.DataFrame(columns=['基金經理人', '選區A次數', '選區B次數', '總次數', '所屬選區'])

        data['選區A次數'] = pd.to_numeric(data.get('選區A次數', 0), errors='coerce').fillna(0)
        data['選區B次數'] = pd.to_numeric(data.get('選區B次數', 0), errors='coerce').fillna(0)
        data['總次數'] = data['選區A次數'] + data['選區B次數']

        data['所屬選區'] = np.select(
            [
                (data['選區A次數'] > 0) & (data['選區B次數'] > 0),
                (data['選區A次數'] > 0) & (data['選區B次數'] <= 0),
                (data['選區A次數'] <= 0) & (data['選區B次數'] > 0),
            ],
            ['A&B 都出現', '只屬於 A', '只屬於 B'],
            default='未分類'
        )

        color_map = {
            '只屬於 A': PLOT_COLORS['A_dark'],
            '只屬於 B': PLOT_COLORS['B_dark'],
            'A&B 都出現': 'rgba(229,57,53,0.95)',
            '未分類': 'rgba(120,120,120,0.45)',
        }
        data['顏色'] = data['所屬選區'].map(color_map).fillna('rgba(120,120,120,0.45)')
        data = data.sort_values(['總次數', '選區A次數', '選區B次數'], ascending=False).reset_index(drop=True)

        y = data['總次數'] if not data.empty else []
        colors = data['顏色'].tolist() if not data.empty else []
        title = 'Part 3：兩選區經理人出現次數比較（藍=A；橘=B；紅=A&B 都出現）'
    else:
        data = count_a.copy() if count_a is not None else pd.DataFrame(columns=['基金經理人', '次數'])
        data['總次數'] = pd.to_numeric(data.get('次數', 0), errors='coerce').fillna(0)
        data['所屬選區'] = '只屬於 A'
        data['選區A次數'] = data['總次數']
        data['選區B次數'] = 0
        colors = [PLOT_COLORS['A_dark']] * len(data)
        y = data['總次數']
        title = 'Part 3：單一選區經理人出現次數'

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.75, 0.25],
        subplot_titles=('經理人出現次數', '')
    )

    custom_cols = ['基金經理人', '所屬選區', '選區A次數', '選區B次數', '總次數']
    for c in custom_cols:
        if c not in data.columns:
            data[c] = 0 if c.endswith('次數') else ''

    fig.add_trace(
        go.Bar(
            x=data['基金經理人'],
            y=y,
            marker_color=colors,
            customdata=data[custom_cols].values,
            hoverinfo='skip',
            name='經理人出現次數'
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scattergl(
            x=data['基金經理人'],
            y=[1] * len(data),
            mode='markers',
            marker=dict(size=9, color=colors, opacity=0.85),
            customdata=data[custom_cols].values,
            hoverinfo='skip',
            name='經理人選取列'
        ),
        row=2,
        col=1
    )

    if compare_mode:
        legend_items = [
            ('只屬於 A', PLOT_COLORS['A_dark']),
            ('只屬於 B', PLOT_COLORS['B_dark']),
            ('A&B 都出現', 'rgba(229,57,53,0.95)'),
        ]
        for label, color in legend_items:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(size=12, color=color),
                    name=label,
                    hoverinfo='skip',
                    showlegend=True,
                ),
                row=1,
                col=1
            )

    selected_visual = set(st.session_state.get('p3_latest_managers', []) or []) | set(st.session_state.get('p3_pending_managers', []) or [])
    selected_visual = [m for m in data['基金經理人'].astype(str).tolist() if m in selected_visual]
    if selected_visual:
        fig.add_trace(
            go.Scatter(
                x=selected_visual,
                y=[1] * len(selected_visual),
                mode='markers',
                marker=dict(symbol='square-open', size=18, color='rgba(255,76,76,1)', line=dict(width=3)),
                hoverinfo='skip',
                name='目前框選'
            ),
            row=2,
            col=1
        )

    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text='所選經理人（請在這一列框選）', tickangle=-55, automargin=True, row=2, col=1)
    fig.update_yaxes(title_text='次數', row=1, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, range=[0.6, 1.4], row=2, col=1)
    fig.update_layout(
        title=title,
        height=max(680, min(1500, 500 + 7 * len(data))),
        margin=dict(l=65, r=30, t=90, b=260),
        hovermode=False,
        dragmode='select',
        showlegend=compare_mode,
        legend=dict(orientation='h', y=1.03, x=0.01),
        uirevision='p3_select_v38'
    )
    return fig, data


def parse_manager_selection(event):
    out = []
    try:
        points = event.selection.points if event and event.selection else []
    except Exception:
        points = []
    def extract(cd):
        if cd is None: return None
        if isinstance(cd, str): return cd
        if isinstance(cd, dict):
            for k in ['基金經理人', '0', 0, 'manager', 'mgr_name']:
                if k in cd:
                    v = extract(cd[k])
                    if v is not None: return v
            for v0 in cd.values():
                v = extract(v0)
                if v is not None: return v
        if isinstance(cd, (list, tuple, np.ndarray)) and len(cd):
            return extract(cd[0])
        return str(cd)
    for p in points:
        cd = p.get('customdata', None) if isinstance(p, dict) else getattr(p, 'customdata', None)
        m = extract(cd)
        if m: out.append(str(m))
    return sorted(set(out))


def radar_record_vector(record):
    return np.array([float(record.get('雷達分數', {}).get(k, 0.5)) for k in RADAR_LABELS], dtype=float)


def render_tab6_part4_radar_grouped(records):
    """Render one radar subplot per similarity group.

    Managers are represented by their radar-score vector.  Similar vectors are
    grouped by cosine similarity and displayed in the same polar subplot, so the
    user can compare managers with similar radar shapes directly.
    """
    if not records:
        return go.Figure(), pd.DataFrame()

    vecs = [radar_record_vector(r) for r in records]
    groups = []
    used = set()
    for i in range(len(records)):
        if i in used:
            continue
        used.add(i)
        vi = vecs[i]
        ni = np.linalg.norm(vi)
        group = [i]
        for j in range(i + 1, len(records)):
            if j in used:
                continue
            vj = vecs[j]
            denom = ni * np.linalg.norm(vj)
            sim = float(np.dot(vi, vj) / denom) if denom > 0 else 0.0
            if sim >= 0.92:
                used.add(j)
                group.append(j)
        groups.append(group)

    group_rows = []
    for gid, idxs in enumerate(groups, start=1):
        group_rows.append({
            '相似群組': f'Group {gid}',
            '經理人數': len(idxs),
            '經理人': '、'.join(str(records[i].get('基金經理人')) for i in idxs)
        })
    group_table = pd.DataFrame(group_rows)

    n_groups = len(groups)
    cols = 2 if n_groups > 1 else 1
    rows = int(np.ceil(n_groups / cols))
    specs = [[{'type': 'polar'} for _ in range(cols)] for _ in range(rows)]
    subplot_titles = [f"Group {gid}：相似雷達圖" for gid in range(1, n_groups + 1)]

    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )

    theta = RADAR_THETA_LABELS + RADAR_THETA_LABELS[:1]
    labels = RADAR_LABELS
    for gid, idxs in enumerate(groups, start=1):
        row = (gid - 1) // cols + 1
        col = (gid - 1) % cols + 1
        for i in idxs:
            rec = records[i]
            values = [rec.get('雷達分數', {}).get(c, 0.5) for c in labels]
            values = values + values[:1]
            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=theta,
                    fill='toself',
                    name=f"{rec.get('基金經理人')} | Group {gid}",
                    opacity=0.62,
                    hoverinfo='skip',
                    hovertemplate=None
                ),
                row=row,
                col=col
            )

    for i in range(1, n_groups + 1):
        polar_key = 'polar' if i == 1 else f'polar{i}'
        fig.layout[polar_key].radialaxis.update(visible=True, range=[0, 1], tickformat='.0%', title='相對分位數')

    fig.update_layout(
        title='Part 4：依雷達圖向量相似度分組的經理人雷達圖',
        showlegend=True,
        height=max(620, 500 * rows),
        margin=dict(l=40, r=40, t=95, b=40)
    )
    return fig, group_table


st.title('📌 Tab6：報酬與 S&P500 對照分析（v38：Part3 A/B 顏色分類版）')
st.caption('此版保留 v36 的 Part1 A/B 彩色底圖；加強 Part2 histogram 顏色可讀性；兩選區模式下 Part3 以藍色/橘色/紅色分別表示只屬於 A、只屬於 B、A/B 都出現。')

st.sidebar.subheader('資料來源')
data_source_mode = st.sidebar.radio('選擇資料來源', ['使用內建 CSV（最快）', '自行上傳 CSV'], index=0)
manual_sp500_ann = st.sidebar.number_input('若 S&P500 CSV 不存在，手動設定 benchmark 年化報酬率', value=0.10, min_value=-1.0, max_value=2.0, step=0.01, format='%.2f')

if data_source_mode == '使用內建 CSV（最快）':
    uploaded_files = [p for p in DEFAULT_FUND_FILES if p.exists()]
    sp500_file = DEFAULT_SP500_FILE if DEFAULT_SP500_FILE.exists() else None
    missing_files = [str(p.name) for p in DEFAULT_FUND_FILES + [DEFAULT_SP500_FILE] if not p.exists()]
    if missing_files:
        st.sidebar.warning('找不到內建檔案：' + ', '.join(missing_files))
    else:
        st.sidebar.success('已使用內建 CSV，不需上傳檔案。')
else:
    uploaded_files = st.sidebar.file_uploader('上傳平衡型基金 CSV (可選多個)', type='csv', accept_multiple_files=True)
    sp500_file = st.sidebar.file_uploader('可選：上傳 S&P500 月報酬 CSV', type='csv', accept_multiple_files=False)

if not uploaded_files:
    st.info('尚未載入資料。請放入內建 CSV 或從側邊欄上傳。')
    st.stop()

df = load_and_combine_data(uploaded_files)
sp500_df = load_sp500_benchmark(sp500_file)
if df is None or df.empty:
    st.warning('資料不足，無法建立 Tab6。')
    st.stop()

all_mgmt = sorted([str(x) for x in df['mgmt_name'].dropna().unique()])
selected_mgmt = st.sidebar.multiselect('選擇管理公司（可不選）', options=all_mgmt)
df_f = df.loc[df['mgmt_name'].astype(str).isin(selected_mgmt)] if selected_mgmt else df

for k, v in {
    'p1_box_A': None, 'p1_box_B': None, 'p1_latest_box': None,
    'p2_latest_region': None, 'p2_pending_regions': [], 'p2_applied_regions': [],
    'p3_latest_managers': [], 'p3_pending_managers': [],
    'p1_chart_nonce': 0, 'p2_hist_nonce': 0,
    'tab6_part4_manager_records': []
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

scatter_tables, benchmark_source = build_tab6_horizon_scatter_tables(df_f, sp500_df=sp500_df, manual_sp500_ann=manual_sp500_ann)
if not scatter_tables:
    st.warning('無法建立 Part1 資料。')
    st.stop()

st.header('Part 1：S&P500 報酬率 × 基金報酬率散點圖')
st.caption(f'S&P500 benchmark 來源：{benchmark_source}。月資料是一檔基金在一個月的點；1/3/5 年是同一個月結束時的 trailing window 年化報酬。')

control_cols = st.columns([1.1, 1.5, 2.4, 1])
with control_cols[0]:
    horizon_display = st.radio(
        'Part1/Part2 時間窗',
        HORIZON_DISPLAY_OPTIONS,
        index=0,
        horizontal=True,
        key='main_horizon_display'
    )
    period_label = HORIZON_DISPLAY_TO_KEY[horizon_display]
with control_cols[1]:
    selection_mode = st.radio(
        'Part1 選區模式',
        ['單一選區', '兩選區比較'],
        index=0,
        horizontal=True,
        key='analysis_mode_v34'
    )
with control_cols[2]:
    st.caption('流程：先在 Part 1 框選 A/B 並套用，才會出現 Part 2；Part 2 後才有 Part 3；Part 3 選經理人才有 Part 4。')
with control_cols[3]:
    if st.button('全部重來', key='reset_all_v34'):
        for k in list(st.session_state.keys()):
            if k.startswith('p1_') or k.startswith('p2_') or k.startswith('p3_') or k.startswith('tab6_part4'):
                st.session_state.pop(k, None)
        st.rerun()

p1_context = (period_label, selection_mode, tuple(selected_mgmt))
if st.session_state.get('p1_selection_context_v34') != p1_context:
    for k in ['p1_box_A','p1_box_B','p1_latest_box','p1_latest_sig','p1_applied',
              'p2_latest_region','p2_pending_regions','p2_applied_regions','p3_latest_managers','p3_pending_managers']:
        if k in st.session_state:
            st.session_state[k] = [] if isinstance(st.session_state.get(k), list) else None
    st.session_state['p1_selection_context_v34'] = p1_context
    st.session_state['p1_chart_nonce'] = st.session_state.get('p1_chart_nonce', 0) + 1
    st.session_state['p2_hist_nonce'] = st.session_state.get('p2_hist_nonce', 0) + 1

zoom_dense = True

df_plot = scatter_tables[period_label]
mask_a = mask_from_box(df_plot, st.session_state.get('p1_box_A'))
mask_b = mask_from_box(df_plot, st.session_state.get('p1_box_B'))
raw_a = df_plot.loc[mask_a].copy() if mask_a.any() else pd.DataFrame(columns=df_plot.columns)
raw_b = df_plot.loc[mask_b].copy() if mask_b.any() else pd.DataFrame(columns=df_plot.columns)
compare_mode = selection_mode.startswith('兩選區') and st.session_state.get('p1_box_A') and st.session_state.get('p1_box_B') and not raw_a.empty and not raw_b.empty
fig_p1 = make_part1_scatter_v32(df_plot, period_label, boxes={'A': st.session_state.get('p1_box_A'), 'B': st.session_state.get('p1_box_B') if compare_mode else None}, zoom_to_dense=zoom_dense)
sel_event = st.plotly_chart(fig_p1, on_select='rerun', selection_mode=('box', 'lasso'), width='stretch',
                            config={'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['select2d', 'lasso2d'], 'modeBarButtonsToRemove': ['hoverClosestCartesian', 'hoverCompareCartesian']}, key=f'p1_scatter_v33_{period_label}_{selection_mode}_{st.session_state.get("p1_chart_nonce", 0)}')
latest_box = extract_selection_bbox(sel_event)
if latest_box is not None:
    st.session_state['p1_latest_box'] = latest_box

p1b = st.columns([1, 1, 1, 1, 2])
with p1b[0]:
    if st.button('設為選區 A', key='set_a_v34'):
        if st.session_state.get('p1_latest_box'):
            st.session_state['p1_box_A'] = st.session_state['p1_latest_box']
            st.session_state['p1_latest_box'] = None
            st.session_state['p1_applied'] = False
            st.session_state['p1_chart_nonce'] = st.session_state.get('p1_chart_nonce', 0) + 1
            st.session_state['p2_hist_nonce'] = st.session_state.get('p2_hist_nonce', 0) + 1
            st.session_state['p2_latest_region'] = None; st.session_state['p2_pending_regions'] = []; st.session_state['p2_applied_regions'] = []
            st.session_state['p3_latest_managers'] = []; st.session_state['p3_pending_managers'] = []
            st.rerun()
        else:
            st.warning('請先在圖上框選一個區域。')
with p1b[1]:
    if selection_mode.startswith('兩選區'):
        if st.button('設為選區 B', key='set_b_v34'):
            if st.session_state.get('p1_latest_box'):
                st.session_state['p1_box_B'] = st.session_state['p1_latest_box']
                st.session_state['p1_latest_box'] = None
                st.session_state['p1_applied'] = False
                st.session_state['p1_chart_nonce'] = st.session_state.get('p1_chart_nonce', 0) + 1
                st.session_state['p2_hist_nonce'] = st.session_state.get('p2_hist_nonce', 0) + 1
                st.session_state['p2_latest_region'] = None; st.session_state['p2_pending_regions'] = []; st.session_state['p2_applied_regions'] = []
                st.session_state['p3_latest_managers'] = []; st.session_state['p3_pending_managers'] = []
                st.rerun()
            else:
                st.warning('請先在圖上框選一個區域。')
with p1b[2]:
    if st.button('套用 Part1 選區', key='apply_p1_v34'):
        has_a = st.session_state.get('p1_box_A') is not None and not raw_a.empty
        has_b = st.session_state.get('p1_box_B') is not None and not raw_b.empty
        if selection_mode.startswith('單一') and has_a:
            st.session_state['p1_applied'] = True
            st.session_state['p2_latest_region'] = None; st.session_state['p2_pending_regions'] = []; st.session_state['p2_applied_regions'] = []
            st.session_state['p3_latest_managers'] = []; st.session_state['p3_pending_managers'] = []
            st.rerun()
        elif selection_mode.startswith('兩選區') and has_a and has_b:
            st.session_state['p1_applied'] = True
            st.session_state['p2_latest_region'] = None; st.session_state['p2_pending_regions'] = []; st.session_state['p2_applied_regions'] = []
            st.session_state['p3_latest_managers'] = []; st.session_state['p3_pending_managers'] = []
            st.rerun()
        else:
            st.warning('請先設定需要的選區。單一模式需要 A；兩區比較需要 A 和 B。')
with p1b[3]:
    if st.button('清空 Part1 選區', key='clear_p1_v34'):
        for k in ['p1_box_A','p1_box_B','p1_latest_box','p1_latest_sig','p1_applied',
                  'p2_latest_region','p2_pending_regions','p2_applied_regions','p3_latest_managers','p3_pending_managers']:
            if k in st.session_state:
                st.session_state[k] = [] if isinstance(st.session_state.get(k), list) else None
        st.session_state['p1_chart_nonce'] = st.session_state.get('p1_chart_nonce', 0) + 1
        st.session_state['p2_hist_nonce'] = st.session_state.get('p2_hist_nonce', 0) + 1
        st.rerun()
with p1b[4]:
    st.caption(f"目前最新框選：{'有' if st.session_state.get('p1_latest_box') else '無'}；A：{len(raw_a):,} 點；B：{len(raw_b):,} 點。")

m1, m2, m3, m4 = st.columns(4)
m1.metric('全部點數', f'{len(df_plot):,}')
m2.metric('選區 A 點數', f'{len(raw_a):,}' if not raw_a.empty else '-')
m3.metric('選區 B 點數', f'{len(raw_b):,}' if not raw_b.empty else '-')
beat_a = (raw_a['y_ret'] > raw_a['x_ret']).mean() if not raw_a.empty else np.nan
m4.metric('A 打敗 S&P500 比例', f'{beat_a:.2%}' if pd.notna(beat_a) else '-')

if raw_a.empty:
    st.info('請先在 Part1 框選一個區域，並按「設為選區 A」。')
    st.stop()
if selection_mode.startswith('兩選區') and raw_b.empty:
    st.info('你目前選擇兩區比較，請再框選第二個區域並按「設為選區 B」。')
    st.stop()
if not st.session_state.get('p1_applied'):
    st.info('請確認選區後按「套用 Part1 選區」，Part2 才會出現。')
    st.stop()

st.divider()
st.header('Part 2：框選樣本的特徵與因子分布')
st.caption('三個層級會同時顯示：月資料特徵、基金層級因子、基金家族層級特徵。你在任一層級框選後，三個層級都會一起連動變化；單選區與兩選區都相同。')

p2_period = period_label
p2c1, p2c2 = st.columns([1, 3])
with p2c1:
    st.caption(f'目前使用上方統一時間窗：{HORIZON_KEY_TO_DISPLAY.get(p2_period, p2_period)}')
with p2c2:
    cumulative_hist = st.checkbox('Y 軸改為累積次數', value=False, key='p2_cum')

p2_df = scatter_tables[p2_period]
mask_a2 = mask_from_box(p2_df, st.session_state.get('p1_box_A'))
mask_b2 = mask_from_box(p2_df, st.session_state.get('p1_box_B'))
raw_a2 = p2_df.loc[mask_a2].copy() if mask_a2.any() else pd.DataFrame(columns=p2_df.columns)
raw_b2 = p2_df.loc[mask_b2].copy() if mask_b2.any() else pd.DataFrame(columns=p2_df.columns)
compare_mode_p2 = selection_mode.startswith('兩選區') and not raw_b2.empty

LEVEL_CONFIGS = {
    '月資料特徵': ['基金月報酬率', 'S&P500月報酬率', '基金超額月報酬', '基金月報酬絕對值', 'Net Flow', 'Flow絕對值', 'MTNA資產規模', '費用率', '管理年資', '換手率', '觀測年份', '家族平均報酬率', '家族平均費用率', '家族平均規模', '家族平均淨申購'],
    '基金層級因子': ['觀測月數', '平均月報酬率', '年化報酬率', '累積報酬率', '年化波動度', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', '月勝率', '打敗S&P500月比例', '年化超額報酬', 'Tracking Error', 'Information Ratio', 'Beta vs S&P500', 'Alpha年化近似', 'Upside Capture', '平均MTNA', '平均Net Flow', '累積Net Flow', '平均費用率', '平均管理年資', '平均換手率'],
    '基金家族層級特徵': ['觀測點數', '基金數', '經理人數', '家族平均報酬率', '家族平均超額報酬', '家族平均費用率', '家族平均規模', '家族平均淨申購', '家族累積淨申購', '家族平均換手率', '家族平均管理年資'],
}

p2_tables_a = {lvl: get_table_for_level(raw_a2, lvl) for lvl in LEVEL_CONFIGS.keys()}
p2_tables_b = {lvl: (get_table_for_level(raw_b2, lvl) if compare_mode_p2 else pd.DataFrame()) for lvl in LEVEL_CONFIGS.keys()}

s1, s2, s3, s4 = st.columns(4)
s1.metric('A 點數', f'{len(raw_a2):,}')
s2.metric('A 基金數', f'{raw_a2["crsp_fundno"].nunique():,}' if 'crsp_fundno' in raw_a2.columns else '-')
s3.metric('B 點數', f'{len(raw_b2):,}' if compare_mode_p2 else '-')
s4.metric('B 基金數', f'{raw_b2["crsp_fundno"].nunique():,}' if compare_mode_p2 and 'crsp_fundno' in raw_b2.columns else '-')

p2b1, p2b2, p2b3, p2info = st.columns([1.2, 1.2, 1, 2.5])
with p2b1:
    if st.button('加入目前 Part2 框選到暫存'):
        if st.session_state.get('p2_latest_region'):
            pending = st.session_state.get('p2_pending_regions', []) or []
            sigs = {repr(x) for x in pending}
            if repr(st.session_state['p2_latest_region']) not in sigs:
                pending.append(st.session_state['p2_latest_region'])
            st.session_state['p2_pending_regions'] = pending
            st.session_state['p2_hist_nonce'] = st.session_state.get('p2_hist_nonce', 0) + 1
            st.rerun()
        else:
            st.warning('請先在下面任一張 histogram 上框選一段。')
with p2b2:
    if st.button('套用 Part2 暫存多選區'):
        if st.session_state.get('p2_pending_regions'):
            st.session_state['p2_applied_regions'] = st.session_state.get('p2_pending_regions', [])
            st.session_state['p2_latest_region'] = None
            st.session_state['p2_pending_regions'] = []
            st.session_state['p3_latest_managers'] = []
            st.session_state['p3_pending_managers'] = []
            st.session_state['p2_hist_nonce'] = st.session_state.get('p2_hist_nonce', 0) + 1
            st.rerun()
        else:
            st.warning('Part2 暫存區目前是空的。')
with p2b3:
    if st.button('清空 Part2 暫存/套用'):
        st.session_state['p2_latest_region'] = None
        st.session_state['p2_pending_regions'] = []
        st.session_state['p2_applied_regions'] = []
        st.session_state['p3_latest_managers'] = []
        st.session_state['p3_pending_managers'] = []
        st.session_state['p2_hist_nonce'] = st.session_state.get('p2_hist_nonce', 0) + 1
        st.rerun()
with p2info:
    latest = st.session_state.get('p2_latest_region')
    latest_txt = '無' if not latest else f"{latest.get('level')} / {latest.get('feature')} {latest.get('x_range')}"
    st.caption(f"最新框選：{latest_txt}；暫存：{len(st.session_state.get('p2_pending_regions', []))} 段；已套用：{len(st.session_state.get('p2_applied_regions', []))} 段。")

for level_name, default_cols in LEVEL_CONFIGS.items():
    st.subheader(f'Part 2：{level_name}')
    data_a = p2_tables_a[level_name]
    data_b = p2_tables_b[level_name] if compare_mode_p2 else pd.DataFrame()
    available_cols = [c for c in default_cols if c in data_a.columns]
    default_show = available_cols[:16]
    selected_cols = st.multiselect(
        f'選擇要顯示的 {level_name} Histogram',
        available_cols,
        default=default_show,
        key=f'p2_cols_all_v39_{level_name}_{p2_period}_{selection_mode}'
    )
    render_selectable_hist_grid_v32(
        data_a,
        data_b,
        raw_a2,
        raw_b2,
        level_name,
        selected_cols,
        compare_mode=compare_mode_p2,
        cumulative=cumulative_hist
    )

applied_regions = st.session_state.get('p2_applied_regions', []) or []
if not applied_regions:
    st.info('請在 Part2 任一層級的 histogram 框選區間 → 加入暫存 → 套用，Part3 才會出現。')
    st.stop()

selected_raw_a = filter_raw_by_regions(raw_a2, None, applied_regions)
selected_raw_b = filter_raw_by_regions(raw_b2, None, applied_regions) if compare_mode_p2 else pd.DataFrame(columns=raw_b2.columns)
if selected_raw_a.empty and (not compare_mode_p2 or selected_raw_b.empty):
    st.warning('Part2 已套用的區間沒有對應到資料，請重新框選。')
    st.stop()

st.divider()
st.header('Part 3：根據 Part2 篩選後的經理人構成圖')
count_a = make_manager_count_data(selected_raw_a)
count_b = make_manager_count_data(selected_raw_b) if compare_mode_p2 else pd.DataFrame()
if compare_mode_p2:
    p3_scenario = '兩區比較：A vs B 經理人長條圖'
    compare_p3 = True
    st.caption('目前是兩選區比較模式，因此 Part3 只顯示 A vs B 經理人長條圖。')
else:
    p3_scenario = '單一選區 A 的經理人長條圖'
    compare_p3 = False
    st.caption('目前是單一選區模式，因此 Part3 顯示選區 A 的經理人長條圖。')
fig_mgr, mgr_table = make_manager_select_chart_v32(count_a, count_b, compare_mode=compare_p3)
ev_mgr = st.plotly_chart(fig_mgr, on_select='rerun', selection_mode=('box','lasso'), width='stretch',
                         config={'displaylogo': False, 'modeBarButtonsToAdd': ['select2d', 'lasso2d'], 'modeBarButtonsToRemove': ['hoverClosestCartesian', 'hoverCompareCartesian']},
                         key=f'p3_mgr_v32_{p3_scenario}_{regions_signature(applied_regions)}')
latest_mgrs = parse_manager_selection(ev_mgr)
if latest_mgrs and st.session_state.get('p3_latest_managers') != latest_mgrs:
    st.session_state['p3_latest_managers'] = latest_mgrs
    st.rerun()

p3b1, p3b2, p3b3, p3info = st.columns([1.2,1.2,1,2.5])
with p3b1:
    if st.button('加入目前 Part3 經理人到暫存'):
        latest = st.session_state.get('p3_latest_managers', []) or []
        if latest:
            pending = set(st.session_state.get('p3_pending_managers', []) or [])
            pending.update(latest)
            st.session_state['p3_pending_managers'] = sorted(pending)
            st.rerun()
        else:
            st.warning('請先在 Part3 下方經理人選取列框選一位或多位經理人。')
with p3b2:
    if st.button('套用 Part3 暫存經理人到 Part4'):
        pending = sorted(set(st.session_state.get('p3_pending_managers', []) or []))
        if not pending:
            st.warning('Part3 暫存經理人目前是空的。')
        else:
            p2_level_label = '全部層級'
            if compare_p3:
                raw_parts = [
                    x for x in [selected_raw_a, selected_raw_b]
                    if x is not None and not x.empty
                ]
                base_raw = (
                    pd.concat(raw_parts, ignore_index=True)
                    if raw_parts else pd.DataFrame(columns=raw_a2.columns)
                )
            else:
                base_raw = selected_raw_a if selected_raw_a is not None else pd.DataFrame(columns=raw_a2.columns)
            radar_base = build_tab6_manager_radar_base(base_raw)
            source_label = f'{p3_scenario} | {p2_level_label} | {p2_period}'
            source_key = f'{p3_scenario}|{p2_level_label}|{p2_period}|{regions_signature(applied_regions)}|{len(base_raw)}'
            existing = {r.get('record_key') for r in st.session_state['tab6_part4_manager_records']}
            added = 0
            for mgr in pending:
                rec = make_tab6_manager_radar_record(radar_base, mgr, source_label, source_key)
                if rec is not None and rec['record_key'] not in existing:
                    st.session_state['tab6_part4_manager_records'].append(rec)
                    existing.add(rec['record_key'])
                    added += 1
            st.session_state['p3_latest_managers'] = []
            st.session_state['p3_pending_managers'] = []
            st.success(f'已加入 {added} 位經理人到 Part4。')
            st.rerun()
with p3b3:
    if st.button('清空 Part3 暫存'):
        st.session_state['p3_latest_managers'] = []
        st.session_state['p3_pending_managers'] = []
        st.rerun()
with p3info:
    st.caption(f"最新框選經理人：{len(st.session_state.get('p3_latest_managers', []))} 位；暫存：{len(st.session_state.get('p3_pending_managers', []))} 位。")

with st.expander('顯示 Part3 經理人統計表', expanded=False):
    st.dataframe(mgr_table, width='stretch')

if not st.session_state.get('tab6_part4_manager_records'):
    st.info('請在 Part3 框選經理人並套用到 Part4，Part4 雷達圖才會出現。')
    st.stop()

st.divider()
st.header('Part 4：經理人特徵雷達圖（具記憶性 + 向量相似分群）')
st.caption('Part4 會保留你加入過的經理人，只有按清空才會刪除。雷達圖特徵會被視為向量，並依 cosine similarity 自動把相似經理人放在一起。')
clear_col, info_col = st.columns([1, 4])
with clear_col:
    if st.button('清空 Part4 記憶'):
        st.session_state['tab6_part4_manager_records'] = []
        st.rerun()
with info_col:
    st.write(f"目前 Part4 記憶清單共有 **{len(st.session_state['tab6_part4_manager_records'])}** 筆經理人 / 條件紀錄。")
fig_radar, group_table = render_tab6_part4_radar_grouped(st.session_state['tab6_part4_manager_records'])
st.plotly_chart(fig_radar, width='stretch', config={'displaylogo': False})
with st.expander('顯示雷達圖向量相似分群', expanded=True):
    st.dataframe(group_table, width='stretch')
rows = []
for rec in st.session_state['tab6_part4_manager_records']:
    row = {'基金經理人': rec.get('基金經理人'), '來源': rec.get('來源')}
    row.update({f'雷達_{k}': v for k, v in rec.get('雷達分數', {}).items()})
    row.update(rec.get('原始指標', {}))
    rows.append(row)
with st.expander('顯示 Part4 原始指標與相對分數', expanded=False):
    st.dataframe(pd.DataFrame(rows), width='stretch')
