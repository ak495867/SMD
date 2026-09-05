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
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

def get_market_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    data = data.dropna(axis=1, how='all')
    returns = data.pct_change().dropna()
    vol = returns.rolling(21).std()
    mom = returns.rolling(63).sum()
    features = pd.concat([returns, vol, mom], axis=1).dropna()
    features.columns = [f"{col}_{feat}" for col in data.columns for feat in ['ret', 'vol', 'mom']]
    target = returns.shift(-1).dropna()
    aligned_features = features.loc[target.index]
    return aligned_features, target, data.columns.tolist()

class SMDModel(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SMDModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        self.metric_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * (latent_dim + 1) // 2),
            nn.Tanh()
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.latent_dim = latent_dim

    def forward(self, x):
        z = self.encoder(x)
        L_flat = self.metric_net(z)
        L = torch.zeros(x.shape[0], self.latent_dim, self.latent_dim, device=x.device)
        idx = 0
        for i in range(self.latent_dim):
            L[:, i, :i+1] = L_flat[:, idx:idx+i+1]
            idx += i + 1
        g = torch.bmm(L, L.transpose(1, 2)) + 1e-4 * torch.eye(self.latent_dim, device=x.device).unsqueeze(0)
        g_inv = torch.linalg.inv(g)
        grad_V = z
        z_next = z - 0.1 * torch.bmm(g_inv, grad_V.unsqueeze(2)).squeeze(2)
        curvature = torch.logdet(g)
        grad_norm_g = torch.sqrt((grad_V.unsqueeze(1) @ g_inv @ grad_V.unsqueeze(2)).squeeze() + 1e-8)
        acceleration = torch.norm(z_next - z, dim=1)
        signal = 0.5 * curvature * grad_norm_g + 0.5 * acceleration
        out = self.head(torch.cat([z_next, curvature.unsqueeze(1), signal.unsqueeze(1)], dim=1))
        return out, signal, curvature

def run_walk_forward(features, target, train_size=252, test_size=63, gap=5, latent_dim=8):
    n_samples = len(features)
    smd_returns = []
    ridge_returns = []
    rf_returns = []
    smd_ic = []
    ridge_ic = []
    rf_ic = []
    curvature_history = []
    
    for i in range(0, n_samples - train_size - gap - test_size + 1, test_size):
        train_end = i + train_size
        test_start = train_end + gap
        test_end = test_start + test_size
        
        X_train, y_train = features.iloc[i:train_end].values, target.iloc[i:train_end].values
        X_test, y_test = features.iloc[test_start:test_end].values, target.iloc[test_start:test_end].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32)
        
        model = SMDModel(input_dim=X_train_t.shape[1], latent_dim=latent_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            preds, _, _ = model(X_train_t)
            loss = mean_squared_error(preds.detach().numpy(), y_train_t.numpy())
            loss_tensor = nn.MSELoss()(preds, y_train_t)
            loss_tensor.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            smd_preds, smd_signals, smd_curv = model(X_test_t)
            smd_preds = smd_preds.numpy().flatten()
            smd_signals = smd_signals.numpy()
            smd_curv = smd_curv.numpy()
            
        curvature_history.extend(smd_curv)
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)
        ridge_preds = ridge.predict(X_test_scaled)
        
        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        rf_preds = rf.predict(X_test_scaled)
        
        smd_ret = np.sign(smd_preds) * y_test.flatten()
        ridge_ret = np.sign(ridge_preds) * y_test.flatten()
        rf_ret = np.sign(rf_preds) * y_test.flatten()
        
        smd_returns.extend(smd_ret)
        ridge_returns.extend(ridge_ret)
        rf_returns.extend(rf_ret)
        
        smd_ic.append(np.corrcoef(smd_preds, y_test.flatten())[0, 1])
        ridge_ic.append(np.corrcoef(ridge_preds, y_test.flatten())[0, 1])
        rf_ic.append(np.corrcoef(rf_preds, y_test.flatten())[0, 1])

    return (np.array(smd_returns), np.array(ridge_returns), np.array(rf_returns),
            np.array(smd_ic), np.array(ridge_ic), np.array(rf_ic), np.array(curvature_history))

if __name__ == '__main__':
    tickers = [
        'SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'GLD', 'SLV', 'TLT', 'IEF', 'HYG', 
        'LQD', 'UUP', 'FXE', 'FXY', 'USO', 'UNG', 'DBA', 'PPLT', 'IAU', 'BND', 
        'VCIT', 'VCSH', 'VGIT', 'VGLT', 'VNQ', 'IYR', 'XLF', 'XLK', 'XLV', 'XLE', 
        'XLI', 'XLP', 'XLY', 'XLU', 'XLRE', 'XLB', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 
        'NVDA', 'META', 'TSLA', 'BRK-B', 'JNJ', 'JPM', 'V', 'PG', 'UNH', 'HD', 
        'MA', 'DIS', 'PYPL', 'NFLX', 'ADBE', 'CRM', 'CMCSA', 'PEP', 'TMO', 'COST', 
        'ABBV', 'ACN', 'AVGO', 'TXN', 'LLY', 'NKE', 'DHR', 'NEE', 'PM', 'BMY', 
        'UNP', 'HON', 'QCOM', 'UPS', 'LOW', 'IBM', 'INTU', 'AMD', 'AMGN', 'CAT', 
        'GS', 'BA', 'SPGI', 'BLK', 'AXP', 'DE', 'GILD', 'MDLZ', 'ISRG', 'TJX', 
        'BKNG', 'ADP', 'VRTX', 'SYK', 'CI', 'ZTS', 'REGN', 'MO', 'CVS', 'PLD', 
        'TGT', 'DUK', 'SO', 'CL', 'BDX', 'MMM', 'EL', 'ITW', 'CSX', 'AON', 
        'EQIX', 'APD', 'SHW', 'CME', 'USB', 'PNC', 'TFC', 'COF', 'MS', 'SCHW',
        'CB', 'MMC', 'ICE', 'AIG', 'TRV', 'ALL', 'PGR', 'AFL', 'MET', 'PRU',
        'HIG', 'AMP', 'LNC', 'PFG', 'TMK', 'RGA', 'FAF', 'EG', 'WRB', 'CINF',
        'AFG', 'Y', 'RLI', 'KMPR', 'THG', 'SIGI', 'ACGL', 'ESGR', 'JRVR', 'PLMR',
        'KNSL', 'UFCS', 'HRTG', 'MCY', 'ARGO', 'MPWR', 'ANSS', 'CDNS', 'SNPS', 'MCHP'
    ]
    
    print("Fetching real market data...")
    features, target, assets = get_market_data(tickers, '2018-01-01', '2023-12-31')
    print(f"Loaded {len(assets)} assets. Shape: {features.shape}")
    
    print("Running purged walk-forward validation...")
    smd_ret, ridge_ret, rf_ret, smd_ic, ridge_ic, rf_ic, curv_hist = run_walk_forward(
        features, target, train_size=504, test_size=63, gap=5, latent_dim=8
    )
    
    smd_cum = np.cumprod(1 + smd_ret * 0.01)
    ridge_cum = np.cumprod(1 + ridge_ret * 0.01)
    rf_cum = np.cumprod(1 + rf_ret * 0.01)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    axes[0].plot(smd_cum, label='SMD (Statistical Manifold Dynamics)', linewidth=2)
    axes[0].plot(ridge_cum, label='Ridge Regression Baseline', linewidth=2, linestyle='--')
    axes[0].plot(rf_cum, label='Random Forest Baseline', linewidth=2, linestyle='-.')
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
    axes[2].set_title('SMD Returns vs Latent Manifold Curvature (Log-Det)')
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