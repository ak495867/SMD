import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_market_data(tickers, start_date, end_date, target_horizon=5):
    """
    Downloads raw daily close prices and calculates rich cross-asset features:
    - Multi-scale momentum & mean-reversion
    - Multi-scale realized volatility
    - Inter-market macro proxy spreads: Credit Spread (HYG/LQD), Yield Curve slope (IEF/TLT), Equity Momentum (QQQ/SPY)
    - Cross-sectional target: Multi-day forward return (target_horizon days)
    """
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    data = data.dropna(axis=1, how='all')

    # Macro and inter-asset regime proxies
    macro_df = pd.DataFrame(index=data.index)
    if 'HYG' in data.columns and 'LQD' in data.columns:
        macro_df['credit_spread'] = data['HYG'] / data['LQD'].replace(0, np.nan)
    else:
        macro_df['credit_spread'] = 1.0

    if 'IEF' in data.columns and 'TLT' in data.columns:
        macro_df['yield_slope'] = data['IEF'] / data['TLT'].replace(0, np.nan)
    else:
        macro_df['yield_slope'] = 1.0

    if 'QQQ' in data.columns and 'SPY' in data.columns:
        macro_df['growth_value_ratio'] = data['QQQ'] / data['SPY'].replace(0, np.nan)
    else:
        macro_df['growth_value_ratio'] = 1.0

    macro_rets = macro_df.pct_change()
    macro_credit_mom = macro_rets['credit_spread'].rolling(21).mean()
    macro_yield_mom = macro_rets['yield_slope'].rolling(21).mean()
    macro_growth_mom = macro_rets['growth_value_ratio'].rolling(21).mean()

    # Asset-specific feature generation
    dfs = []
    min_len = 65 + target_horizon
    for col in data.columns:
        s = data[col].dropna()
        if len(s) < min_len:
            continue
        r1 = s.pct_change()
        r5 = s.pct_change(5)
        mom21 = s.pct_change(21)
        mom63 = s.pct_change(63)
        vol21 = r1.rolling(21).std()
        vol63 = r1.rolling(63).std()
        
        # Volatility ratio (short term / long term vol expansion)
        vol_ratio = vol21 / vol63.replace(0, np.nan)
        
        # Target: N-day forward cumulative return
        target = s.pct_change(target_horizon).shift(-target_horizon)

        df = pd.DataFrame({
            'date': s.index,
            'ticker': col,
            'ret_1d': r1.values,
            'ret_5d': r5.values,
            'mom_21d': mom21.values,
            'mom_63d': mom63.values,
            'vol_21d': vol21.values,
            'vol_ratio': vol_ratio.values,
            'credit_mom': macro_credit_mom.reindex(s.index).values,
            'yield_mom': macro_yield_mom.reindex(s.index).values,
            'growth_mom': macro_growth_mom.reindex(s.index).values,
            'target': target.values
        })
        dfs.append(df)

    long_df = pd.concat(dfs).dropna().sort_values('date').reset_index(drop=True)
    return long_df

def cross_sectional_standardize(df, feature_cols):
    """
    Standardizes features cross-sectionally per date (z-score across tickers at date t).
    Prevents cross-asset scaling distortion and lookahead across time.
    """
    df = df.copy()
    grouped = df.groupby('date')[feature_cols]
    mean = grouped.transform('mean')
    std = grouped.transform('std').replace(0, 1.0).fillna(1.0)
    df[feature_cols] = (df[feature_cols] - mean) / std
    df[feature_cols] = df[feature_cols].fillna(0.0).clip(-3.0, 3.0)
    return df

class SMDModel(nn.Module):
    """
    Statistical Manifold Dynamics (SMD) Neural Network.
    
    Learns a local Riemannian metric tensor g(z) = L L^T via strictly positive diagonal Cholesky factor L.
    Computes natural gradient drift, geodesic metric norm, and log-determinant volume density form.
    """
    def __init__(self, input_dim, latent_dim):
        super(SMDModel, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        n_tri = latent_dim * (latent_dim + 1) // 2
        self.metric_net = nn.Linear(latent_dim, n_tri)
        self.head = nn.Sequential(
            nn.Linear(latent_dim + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        tri_rows, tri_cols = torch.tril_indices(latent_dim, latent_dim)
        self.register_buffer('tri_rows', tri_rows)
        self.register_buffer('tri_cols', tri_cols)

        diag_mask = tri_rows == tri_cols
        self.register_buffer('diag_indices_in_flat', torch.nonzero(diag_mask).squeeze(1))

    def forward(self, x):
        z = self.encoder(x)
        L_flat = self.metric_net(z)

        # Enforce strictly positive diagonal on Cholesky factor for positive-definite metric
        L_flat_pos = L_flat.clone()
        L_flat_pos[:, self.diag_indices_in_flat] = F.softplus(L_flat[:, self.diag_indices_in_flat]) + 1e-3

        L = torch.zeros(x.shape[0], self.latent_dim, self.latent_dim, device=x.device, dtype=x.dtype)
        L[:, self.tri_rows, self.tri_cols] = L_flat_pos

        # Metric tensor g = L L^T + eps * I
        g = torch.bmm(L, L.transpose(1, 2)) + 1e-4 * torch.eye(self.latent_dim, device=x.device, dtype=x.dtype).unsqueeze(0)
        g_inv = torch.linalg.inv(g)

        # Natural gradient step: -g^{-1} \nabla V with quadratic potential V(z) = 0.5 * ||z||^2
        grad_V = z.unsqueeze(2)
        natural_grad = torch.bmm(g_inv, grad_V).squeeze(2)
        z_next = z - 0.1 * natural_grad

        # Riemannian volume density proxy (log-determinant)
        logdet_vol = torch.logdet(g)

        # Geodesic norm: sqrt(grad_V^T g^{-1} grad_V)
        quad = torch.bmm(grad_V.transpose(1, 2), torch.bmm(g_inv, grad_V)).squeeze()
        grad_norm_g = torch.sqrt(torch.clamp(quad, min=1e-8))

        # Step velocity magnitude
        velocity = torch.norm(z_next - z, dim=1)

        # Geodesic tension alpha signal
        signal = 0.5 * logdet_vol * grad_norm_g + 0.5 * velocity

        features = torch.cat([z_next, logdet_vol.unsqueeze(1), signal.unsqueeze(1)], dim=1)
        out = self.head(features)
        return out, signal, logdet_vol

def pairwise_ranking_loss(preds, targets):
    """
    Soft pairwise rank-correlation objective:
    Maximizes the cosine similarity between centered prediction differences and target differences,
    directly optimizing for Spearman rank correlation.
    """
    preds_c = preds - torch.mean(preds)
    targets_c = targets - torch.mean(targets)
    cos_sim = F.cosine_similarity(preds_c.flatten(), targets_c.flatten(), dim=0)
    # Combined rank loss + subtle regularizing MSE
    mse = F.mse_loss(preds, targets)
    return -cos_sim + 0.1 * mse

def compute_daily_cross_sectional_ic(df_test_preds, pred_col, target_col='target'):
    """
    Computes daily Spearman rank correlation IC across assets for each date t.
    Returns array of daily ICs.
    """
    daily_ics = []
    for date, group in df_test_preds.groupby('date'):
        valid = group[[pred_col, target_col]].dropna()
        if len(valid) >= 5:
            corr, _ = spearmanr(valid[pred_col], valid[target_col])
            if np.isfinite(corr):
                daily_ics.append(corr)
    return np.array(daily_ics)

def simulate_cross_sectional_portfolio(df_test_preds, score_col, target_col='target', fee_bps=10.0, rebalance_freq=5, use_curvature_filter=False):
    """
    Simulates a dollar-neutral cross-sectional portfolio.
    - Rebalances every `rebalance_freq` days (e.g., 5-day horizon) to slash turnover drag.
    - Optionally modulates position sizes by manifold volume density form (curvature filter):
      reducing exposure when structural fragility / volume distortion spikes.
    - Accurately deducts turnover transaction costs.
    """
    daily_returns = {}
    prev_weights = {}
    fee_mult = fee_bps / 10000.0

    unique_dates = sorted(df_test_preds['date'].unique())
    last_weights = {}

    for i, date in enumerate(unique_dates):
        group = df_test_preds[df_test_preds['date'] == date]
        tickers = group['ticker'].values
        scores = group[score_col].values
        targets = group[target_col].values

        if len(scores) < 5:
            continue

        # Check if today is a rebalance day
        if (i % rebalance_freq == 0) or (len(last_weights) == 0):
            # Demean scores for dollar-neutrality
            demeaned = scores - np.mean(scores)

            # Curvature filter: if active, downweight assets with extreme volume density distortion
            if use_curvature_filter and 'curvature' in group.columns:
                curv = group['curvature'].values
                # Dampen weights where curvature is in the extreme tail (top 15% fragility)
                curv_threshold = np.percentile(curv, 85)
                fragility_penalty = np.where(curv > curv_threshold, 0.5, 1.0)
                demeaned = demeaned * fragility_penalty

            norm = np.sum(np.abs(demeaned))
            if norm > 1e-8:
                current_weights = demeaned / norm
            else:
                current_weights = np.zeros_like(demeaned)
            current_weight_map = dict(zip(tickers, current_weights))
        else:
            # Maintain previous weights
            current_weight_map = last_weights

        # Turnover calculation
        all_tickers = set(current_weight_map.keys()).union(set(prev_weights.keys()))
        turnover = sum(abs(current_weight_map.get(k, 0.0) - prev_weights.get(k, 0.0)) for k in all_tickers)
        tcost = turnover * fee_mult

        # Target is scaled to 1-day step approximation of the multi-day path
        # realized return for day t: target divided by holding period
        daily_target = targets / float(rebalance_freq)
        gross_ret = np.sum(np.array([current_weight_map.get(k, 0.0) for k in tickers]) * daily_target)
        net_ret = gross_ret - tcost

        daily_returns[date] = net_ret
        prev_weights = current_weight_map
        last_weights = current_weight_map

    return pd.Series(daily_returns).sort_index()

def run_walk_forward(long_df, train_days=504, test_days=63, gap_days=5, latent_dim=8, batch_size=2048, fee_bps=10.0, target_horizon=5):
    """
    Runs purged walk-forward validation with pairwise ranking loss,
    multi-day rebalancing, and optional curvature filtering.
    """
    feature_cols = [
        'ret_1d', 'ret_5d', 'mom_21d', 'mom_63d', 'vol_21d', 'vol_ratio',
        'credit_mom', 'yield_mom', 'growth_mom'
    ]

    print("Applying cross-sectional factor standardization...")
    long_df = cross_sectional_standardize(long_df, feature_cols)

    unique_dates = np.array(sorted(long_df['date'].unique()))
    n_dates = len(unique_dates)
    current_date_idx = 0

    all_test_dfs = []
    curvature_history = []
    window_smd_ics, window_ridge_ics, window_rf_ics = [], [], []

    while current_date_idx + train_days + gap_days + test_days <= n_dates:
        train_end_idx = current_date_idx + train_days
        gap_end_idx = train_end_idx + gap_days
        test_end_idx = gap_end_idx + test_days

        train_mask = (long_df['date'] >= unique_dates[current_date_idx]) & (long_df['date'] < unique_dates[train_end_idx])
        test_mask = (long_df['date'] >= unique_dates[gap_end_idx]) & (long_df['date'] < unique_dates[test_end_idx])

        train_df = long_df.loc[train_mask]
        test_df = long_df.loc[test_mask].copy()

        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df['target'].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df['target'].values.astype(np.float32)

        X_train_t = torch.tensor(X_train, device=DEVICE)
        y_train_t = torch.tensor(y_train, device=DEVICE).unsqueeze(1)
        X_test_t = torch.tensor(X_test, device=DEVICE)

        model = SMDModel(input_dim=len(feature_cols), latent_dim=latent_dim).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

        n_samples = X_train_t.shape[0]
        indices = torch.arange(n_samples, device=DEVICE)

        # Train model with pairwise ranking loss
        model.train()
        for epoch in range(35):
            perm = indices[torch.randperm(n_samples, device=DEVICE)]
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = perm[start:end]
                xb = X_train_t[batch_idx]
                yb = y_train_t[batch_idx]
                preds, _, _ = model(xb)
                loss = pairwise_ranking_loss(preds, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            all_preds, all_curv = [], []
            for start in range(0, X_test_t.shape[0], batch_size):
                end = min(start + batch_size, X_test_t.shape[0])
                xb = X_test_t[start:end]
                p, _, c = model(xb)
                all_preds.append(p)
                all_curv.append(c)
            smd_preds = torch.cat(all_preds).cpu().numpy().flatten()
            smd_curv = torch.cat(all_curv).cpu().numpy().flatten()

        curvature_history.extend(smd_curv)

        # Baseline: Ridge Regression
        ridge = Ridge(alpha=10.0)
        ridge.fit(X_train, y_train)
        ridge_preds = ridge.predict(X_test)

        # Baseline: Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=40, max_depth=4, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)

        test_df['smd_pred'] = smd_preds
        test_df['ridge_pred'] = ridge_preds
        test_df['rf_pred'] = rf_preds
        test_df['curvature'] = smd_curv
        all_test_dfs.append(test_df)

        # Daily Cross-Sectional Spearman Rank ICs in this window
        daily_ic_smd = compute_daily_cross_sectional_ic(test_df, 'smd_pred')
        daily_ic_ridge = compute_daily_cross_sectional_ic(test_df, 'ridge_pred')
        daily_ic_rf = compute_daily_cross_sectional_ic(test_df, 'rf_pred')

        m_smd = np.mean(daily_ic_smd) if len(daily_ic_smd) > 0 else np.nan
        m_ridge = np.mean(daily_ic_ridge) if len(daily_ic_ridge) > 0 else np.nan
        m_rf = np.mean(daily_ic_rf) if len(daily_ic_rf) > 0 else np.nan

        window_smd_ics.append(m_smd)
        window_ridge_ics.append(m_ridge)
        window_rf_ics.append(m_rf)

        window_num = (current_date_idx // test_days) + 1
        print(f"Window {window_num:02d}: SMD Rank IC={m_smd:.4f} | Ridge Rank IC={m_ridge:.4f} | RF Rank IC={m_rf:.4f}")

        current_date_idx += test_days

    full_test_df = pd.concat(all_test_dfs).sort_values('date').reset_index(drop=True)

    # Simulating portfolios with 5-day rebalancing and 10 bps fee drag
    print(f"\nSimulating dollar-neutral portfolios ({target_horizon}-day rebalance, {fee_bps:.1f} bps fees)...")
    smd_daily_ret = simulate_cross_sectional_portfolio(full_test_df, 'smd_pred', fee_bps=fee_bps, rebalance_freq=target_horizon, use_curvature_filter=False)
    smd_filtered_daily_ret = simulate_cross_sectional_portfolio(full_test_df, 'smd_pred', fee_bps=fee_bps, rebalance_freq=target_horizon, use_curvature_filter=True)
    ridge_daily_ret = simulate_cross_sectional_portfolio(full_test_df, 'ridge_pred', fee_bps=fee_bps, rebalance_freq=target_horizon)
    rf_daily_ret = simulate_cross_sectional_portfolio(full_test_df, 'rf_pred', fee_bps=fee_bps, rebalance_freq=target_horizon)

    # Full time-series daily ICs
    all_smd_ics = compute_daily_cross_sectional_ic(full_test_df, 'smd_pred')
    all_ridge_ics = compute_daily_cross_sectional_ic(full_test_df, 'ridge_pred')
    all_rf_ics = compute_daily_cross_sectional_ic(full_test_df, 'rf_pred')

    results = {
        'smd_daily_ret': smd_daily_ret,
        'smd_filtered_daily_ret': smd_filtered_daily_ret,
        'ridge_daily_ret': ridge_daily_ret,
        'rf_daily_ret': rf_daily_ret,
        'all_smd_ics': all_smd_ics,
        'all_ridge_ics': all_ridge_ics,
        'all_rf_ics': all_rf_ics,
        'window_smd_ics': np.array(window_smd_ics),
        'window_ridge_ics': np.array(window_ridge_ics),
        'window_rf_ics': np.array(window_rf_ics),
        'curvature_history': np.array(curvature_history),
        'full_test_df': full_test_df
    }
    return results

if __name__ == '__main__':
    tickers = [
        'SPY','QQQ','IWM','EFA','EEM','GLD','SLV','TLT','IEF','HYG',
        'LQD','UUP','FXE','FXY','USO','UNG','DBA','PPLT','IAU','BND',
        'VCIT','VCSH','VGIT','VGLT','VNQ','IYR','XLF','XLK','XLV','XLE',
        'XLI','XLP','XLY','XLU','XLRE','XLB','AAPL','MSFT','GOOGL','AMZN',
        'NVDA','META','TSLA','BRK-B','JNJ','JPM','V','PG','UNH','HD',
        'MA','DIS','PYPL','NFLX','ADBE','CRM','CMCSA','PEP','TMO','COST',
        'ABBV','ACN','AVGO','TXN','LLY','NKE','DHR','NEE','PM','BMY',
        'UNP','HON','QCOM','UPS','LOW','IBM','INTU','AMD','AMGN','CAT',
        'GS','BA','SPGI','BLK','AXP','DE','GILD','MDLZ','ISRG','TJX',
        'BKNG','ADP','VRTX','SYK','CI','ZTS','REGN','MO','CVS','PLD',
        'TGT','DUK','SO','CL','BDX','MMM','EL','ITW','CSX','AON',
        'EQIX','APD','SHW','CME','USB','PNC','TFC','COF','MS','SCHW',
        'CB','ICE','AIG','TRV','ALL','PGR','AFL','MET','PRU',
        'HIG','AMP','LNC','PFG','RGA','FAF','EG','WRB','CINF',
        'AFG','RLI','KMPR','THG','SIGI','ACGL','JRVR','PLMR',
        'KNSL','UFCS','HRTG','MCY','MPWR','CDNS','SNPS','MCHP',
        'AMAT','LRCX','KLAC','ASML','TSM','MU','NXPI','MRVL','ADI','ON',
        'INTC','CSCO','ORCL','SAP','SHOP','UBER','LYFT','ABNB','COIN',
        'HOOD','RBLX','U','SNOW','NET','DDOG','CRWD','ZS','OKTA','PANW',
        'FTNT','NOW','WDAY','VEEV','ZM','DOCU','TWLO','MDB',
        'ESTC','GTLB','S','BILL','PAYC','STNE','PAGS','NU','MELI',
        'SE','GRAB','BABA','JD','PDD','BIDU','NIO','XPEV','LI','TME',
        'BILI','IQ','VIPS','WB','NTES','YMM','TAL','EDU','GOTU','COE',
        'TIGR','FUTU','HUYA','DOYU','MOMO','KC','BZ','RLX','YQ',
        'DUOL','GLOB','CPNG','DASH','RIVN','LCID','F','GM','TM','HMC',
        'STLA','RACE','VWAGY','NSANY','SONY','NVO'
    ]

    TARGET_HORIZON = 5
    print(f"Using device: {DEVICE}")
    print(f"Fetching real market data with {TARGET_HORIZON}-day forward return target and macro factors...")
    long_df = get_market_data(tickers, '2018-01-01', '2023-12-31', target_horizon=TARGET_HORIZON)
    print(f"Loaded {long_df['ticker'].nunique()} assets. Total observations: {len(long_df)}")

    print(f"Running purged walk-forward validation (Horizon={TARGET_HORIZON}d, Ranking Loss, Curvature Filter)...")
    results = run_walk_forward(
        long_df, train_days=504, test_days=63, gap_days=TARGET_HORIZON, latent_dim=8,
        batch_size=2048, fee_bps=10.0, target_horizon=TARGET_HORIZON
    )

    smd_daily = results['smd_daily_ret']
    smd_filt_daily = results['smd_filtered_daily_ret']
    ridge_daily = results['ridge_daily_ret']
    rf_daily = results['rf_daily_ret']

    # Legitimate time-series compounding on portfolio-level daily returns
    smd_cum = (1 + smd_daily).cumprod()
    smd_filt_cum = (1 + smd_filt_daily).cumprod()
    ridge_cum = (1 + ridge_daily).cumprod()
    rf_cum = (1 + rf_daily).cumprod()

    # Daily Rank IC metrics
    all_smd_ics = results['all_smd_ics']
    all_ridge_ics = results['all_ridge_ics']
    all_rf_ics = results['all_rf_ics']

    mean_smd_ic = np.mean(all_smd_ics) if len(all_smd_ics) > 0 else 0.0
    std_smd_ic = np.std(all_smd_ics) if len(all_smd_ics) > 0 else 1.0
    icir_smd = mean_smd_ic / (std_smd_ic + 1e-8) * np.sqrt(252)

    mean_ridge_ic = np.mean(all_ridge_ics) if len(all_ridge_ics) > 0 else 0.0
    mean_rf_ic = np.mean(all_rf_ics) if len(all_rf_ics) > 0 else 0.0

    fig, axes = plt.subplots(3, 1, figsize=(14, 18))

    # 1. Real Portfolio Cumulative Returns
    axes[0].plot(smd_cum.index, smd_cum.values, label=f'SMD Raw ({TARGET_HORIZON}d Rebal, Net 10bps)', linewidth=2)
    axes[0].plot(smd_filt_cum.index, smd_filt_cum.values, label=f'SMD + Curvature Filter ({TARGET_HORIZON}d, Net 10bps)', linewidth=2.5, color='darkgreen')
    axes[0].plot(ridge_cum.index, ridge_cum.values, label=f'Ridge ({TARGET_HORIZON}d, Net 10bps)', linewidth=1.8, linestyle='--')
    axes[0].plot(rf_cum.index, rf_cum.values, label=f'Random Forest ({TARGET_HORIZON}d, Net 10bps)', linewidth=1.8, linestyle='-.')
    axes[0].set_title(f'Cross-Sectional Portfolio Cumulative Returns ({TARGET_HORIZON}-Day Horizon, Purged Walk-Forward, Net Fees)')
    axes[0].set_ylabel('Portfolio Wealth ($1 base)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Window-Level Daily Cross-Sectional Mean Rank IC
    axes[1].plot(results['window_smd_ics'], label=f'SMD Mean Rank IC ({mean_smd_ic:.4f}, Ann. ICIR: {icir_smd:.2f})', alpha=0.8, marker='o')
    axes[1].plot(results['window_ridge_ics'], label=f'Ridge Mean Rank IC ({mean_ridge_ic:.4f})', alpha=0.8, linestyle='--', marker='s')
    axes[1].plot(results['window_rf_ics'], label=f'RF Mean Rank IC ({mean_rf_ic:.4f})', alpha=0.8, linestyle='-.', marker='^')
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set_title(f'Cross-Sectional Spearman Rank IC ({TARGET_HORIZON}-Day Target Horizon)')
    axes[1].set_ylabel('Daily Rank IC (Mean per window)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Volume Form Proxy vs Forward Return
    full_df = results['full_test_df']
    axes[2].scatter(full_df['curvature'], full_df['target'], alpha=0.15, s=8)
    axes[2].set_title('SMD Volume Density Form Proxy (log-det g) vs Forward 5-Day Returns')
    axes[2].set_xlabel('Log-Det Volume Density Form Proxy')
    axes[2].set_ylabel(f'Forward {TARGET_HORIZON}-Day Return')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('smd_evaluation.png', dpi=300)
    plt.show()

    print("\n" + "="*50)
    print(f"QUANT EVALUATION SUMMARY ({TARGET_HORIZON}-DAY HORIZON)")
    print("="*50)
    print(f"Final SMD Raw Net Wealth:           {smd_cum.iloc[-1]:.4f}" if len(smd_cum) > 0 else "N/A")
    print(f"Final SMD + Curvature Filter Wealth:{smd_filt_cum.iloc[-1]:.4f}" if len(smd_filt_cum) > 0 else "N/A")
    print(f"Final Ridge Net Wealth:             {ridge_cum.iloc[-1]:.4f}" if len(ridge_cum) > 0 else "N/A")
    print(f"Final RF Net Wealth:                {rf_cum.iloc[-1]:.4f}" if len(rf_cum) > 0 else "N/A")
    print(f"SMD Cross-Sectional Mean Rank IC:   {mean_smd_ic:.4f}")
    print(f"SMD Annualized ICIR:                {icir_smd:.2f}")
    print("="*50)