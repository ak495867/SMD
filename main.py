import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_market_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    data = data.dropna(axis=1, how='all')
    rets = data.pct_change()
    vol = rets.rolling(21).std()
    mom = rets.rolling(63).sum()
    target = rets.shift(-1)
    dfs = []
    for col in data.columns:
        df = pd.DataFrame({
            'date': rets.index,
            'ticker': col,
            'ret': rets[col].values,
            'vol': vol[col].values,
            'mom': mom[col].values,
            'target': target[col].values
        })
        dfs.append(df)
    long_df = pd.concat(dfs).dropna().sort_values('date').reset_index(drop=True)
    return long_df

class SMDModel(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SMDModel, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
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

    def forward(self, x):
        z = self.encoder(x)
        L_flat = self.metric_net(z)
        L = torch.zeros(x.shape[0], self.latent_dim, self.latent_dim, device=x.device, dtype=x.dtype)
        L[:, self.tri_rows, self.tri_cols] = L_flat
        g = torch.bmm(L, L.transpose(1, 2)) + 1e-4 * torch.eye(self.latent_dim, device=x.device, dtype=x.dtype).unsqueeze(0)
        g_inv = torch.linalg.inv(g)
        grad_V = z.unsqueeze(2)
        z_next = z - 0.1 * torch.bmm(g_inv, grad_V).squeeze(2)
        curvature = torch.logdet(g)
        quad = torch.bmm(grad_V.transpose(1, 2), torch.bmm(g_inv, grad_V)).squeeze()
        grad_norm_g = torch.sqrt(torch.clamp(quad, min=1e-8))
        acceleration = torch.norm(z_next - z, dim=1)
        signal = 0.5 * curvature * grad_norm_g + 0.5 * acceleration
        features = torch.cat([z_next, curvature.unsqueeze(1), signal.unsqueeze(1)], dim=1)
        out = self.head(features)
        return out, signal, curvature

def run_walk_forward(long_df, train_days=504, test_days=63, gap_days=5, latent_dim=8, batch_size=4096):
    unique_dates = long_df['date'].unique()
    smd_returns, ridge_returns, rf_returns = [], [], []
    smd_ic, ridge_ic, rf_ic = [], [], []
    curvature_history = []
    feature_cols = ['ret', 'vol', 'mom']
    n_dates = len(unique_dates)
    current_date_idx = 0

    while current_date_idx + train_days + gap_days + test_days <= n_dates:
        train_end_idx = current_date_idx + train_days
        gap_end_idx = train_end_idx + gap_days
        test_end_idx = gap_end_idx + test_days

        train_mask = (long_df['date'] >= unique_dates[current_date_idx]) & (long_df['date'] < unique_dates[train_end_idx])
        test_mask = (long_df['date'] >= unique_dates[gap_end_idx]) & (long_df['date'] < unique_dates[test_end_idx])

        train_df = long_df.loc[train_mask]
        test_df = long_df.loc[test_mask]

        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df['target'].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df['target'].values.astype(np.float32)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_t = torch.tensor(X_train_scaled, device=DEVICE)
        y_train_t = torch.tensor(y_train, device=DEVICE).unsqueeze(1)
        X_test_t = torch.tensor(X_test_scaled, device=DEVICE)

        model = SMDModel(input_dim=3, latent_dim=latent_dim).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.MSELoss()

        n_samples = X_train_t.shape[0]
        indices = torch.arange(n_samples, device=DEVICE)

        for epoch in range(50):
            perm = indices[torch.randperm(n_samples, device=DEVICE)]
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = perm[start:end]
                xb = X_train_t[batch_idx]
                yb = y_train_t[batch_idx]
                preds, _, _ = model(xb)
                loss = criterion(preds, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

        model.eval()
        with torch.no_grad():
            all_preds = []
            all_curv = []
            for start in range(0, X_test_t.shape[0], batch_size):
                end = min(start + batch_size, X_test_t.shape[0])
                xb = X_test_t[start:end]
                p, _, c = model(xb)
                all_preds.append(p)
                all_curv.append(c)
            smd_preds = torch.cat(all_preds).cpu().numpy().flatten()
            smd_curv = torch.cat(all_curv).cpu().numpy()

        curvature_history.extend(smd_curv)

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)
        ridge_preds = ridge.predict(X_test_scaled)

        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        rf_preds = rf.predict(X_test_scaled)

        smd_ret = np.sign(smd_preds) * y_test
        ridge_ret = np.sign(ridge_preds) * y_test
        rf_ret = np.sign(rf_preds) * y_test

        smd_returns.extend(smd_ret)
        ridge_returns.extend(ridge_ret)
        rf_returns.extend(rf_ret)

        valid_mask_smd = np.isfinite(smd_preds) & np.isfinite(y_test)
        valid_mask_ridge = np.isfinite(ridge_preds) & np.isfinite(y_test)
        valid_mask_rf = np.isfinite(rf_preds) & np.isfinite(y_test)

        ic_smd = np.corrcoef(smd_preds[valid_mask_smd], y_test[valid_mask_smd])[0, 1] if valid_mask_smd.sum() > 2 else np.nan
        ic_ridge = np.corrcoef(ridge_preds[valid_mask_ridge], y_test[valid_mask_ridge])[0, 1] if valid_mask_ridge.sum() > 2 else np.nan
        ic_rf = np.corrcoef(rf_preds[valid_mask_rf], y_test[valid_mask_rf])[0, 1] if valid_mask_rf.sum() > 2 else np.nan

        smd_ic.append(ic_smd)
        ridge_ic.append(ic_ridge)
        rf_ic.append(ic_rf)

        current_date_idx += test_days
        print(f"Window {current_date_idx // test_days}: SMD IC={ic_smd:.4f} | Ridge IC={ic_ridge:.4f} | RF IC={ic_rf:.4f}")

    return (np.array(smd_returns), np.array(ridge_returns), np.array(rf_returns),
            np.array(smd_ic), np.array(ridge_ic), np.array(rf_ic), np.array(curvature_history))

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

    print(f"Using device: {DEVICE}")
    print("Fetching real market data...")
    long_df = get_market_data(tickers, '2018-01-01', '2023-12-31')
    print(f"Loaded {long_df['ticker'].nunique()} assets. Total observations: {len(long_df)}")

    print("Running purged walk-forward validation on GPU...")
    smd_ret, ridge_ret, rf_ret, smd_ic, ridge_ic, rf_ic, curv_hist = run_walk_forward(
        long_df, train_days=504, test_days=63, gap_days=5, latent_dim=8, batch_size=4096
    )

    smd_cum = np.cumprod(1 + smd_ret * 0.01)
    ridge_cum = np.cumprod(1 + ridge_ret * 0.01)
    rf_cum = np.cumprod(1 + rf_ret * 0.01)

    fig, axes = plt.subplots(3, 1, figsize=(14, 18))

    axes[0].plot(smd_cum, label='SMD', linewidth=2)
    axes[0].plot(ridge_cum, label='Ridge', linewidth=2, linestyle='--')
    axes[0].plot(rf_cum, label='Random Forest', linewidth=2, linestyle='-.')
    axes[0].set_title('Cumulative Returns (Purged Walk-Forward)')
    axes[0].set_ylabel('Portfolio Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(smd_ic, label='SMD IC', alpha=0.7)
    axes[1].plot(ridge_ic, label='Ridge IC', alpha=0.7)
    axes[1].plot(rf_ic, label='RF IC', alpha=0.7)
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set_title('Information Coefficient (IC) per Test Window')
    axes[1].set_ylabel('Pearson Correlation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(curv_hist, smd_ret, alpha=0.3, s=10)
    axes[2].set_title('SMD Returns vs Latent Manifold Curvature')
    axes[2].set_xlabel('Curvature Proxy (log-det of Fisher Metric)')
    axes[2].set_ylabel('Forward Return')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('smd_evaluation.png', dpi=300)
    plt.show()

    print(f"Final SMD Cumulative Return: {smd_cum[-1]:.4f}")
    print(f"Final Ridge Cumulative Return: {ridge_cum[-1]:.4f}")
    print(f"Final RF Cumulative Return: {rf_cum[-1]:.4f}")
    print(f"Mean SMD IC: {np.nanmean(smd_ic):.4f}")