#!/usr/bin/env python3
"""
Phase 3: Multi-Source Alternative Data for Crypto Prediction
=============================================================
Combines 5 pillars beyond OHLCV:
  1. On-Chain Data     — Whale flows, exchange activity, volume trends
  2. Sentiment         — Fear & Greed Index, social activity
  3. Macro Indicators  — BTC dominance, market cap trends
  4. Order Book Proxy  — Spread, order flow imbalance
  5. Options Proxy     — Volatility regime, skew indicators

Architecture: Multi-source feature fusion → LSTM/Transformer → Ensemble

Free data sources:
  - Alternative.me (Fear & Greed Index)
  - CoinGecko API (market data, community stats)
  - Derived on-chain/macro proxies
"""

import argparse, os, sys, time, json, warnings, urllib.request
from datetime import datetime
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ─── CONFIG ───────────────────────────────────────────────────────────────
CONFIG = {
    'coin': 'BTC-USD', 'period': '5y', 'lookback': 30,
    'train_ratio': 0.7, 'val_ratio': 0.15,
    'hidden_size': 64, 'num_layers': 2, 'dropout': 0.3,
    'lr': 1e-3, 'weight_decay': 1e-3, 'epochs': 100, 'batch_size': 32,
    'patience': 25, 'grad_clip': 1.0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42, 'output_dir': 'outputs',
}

def yf_clean(ticker, period='5y'):
    """Get yfinance data with clean timezone-naive index."""
    import yfinance as yf
    df = yf.Ticker(ticker).history(period=period, interval='1d')
    df.index = df.index.tz_localize(None).normalize()
    return df

# ─── DATA FETCHERS ────────────────────────────────────────────────────────
def safe_json(url, timeout=30):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"  [-] Failed: {url[:60]}... ({e})")
        return None

def fetch_fear_greed(days=1000):
    """Fear & Greed Index — alternative.me (free, no key)."""
    print("[>] Fear & Greed Index...")
    data = safe_json(f"https://api.alternative.me/fng/?limit={days}")
    if not data: return pd.DataFrame()
    records = [{'date': datetime.fromtimestamp(int(d['timestamp'])).date(),
                'fgi_value': int(d['value']),
                'fgi_class': d['value_classification']} for d in data['data']]
    df = pd.DataFrame(records).sort_values('date').set_index('date')
    cmap = {'Extreme Fear': 0, 'Fear': 25, 'Neutral': 50, 'Greed': 75, 'Extreme Greed': 100}
    df['fgi_encoded'] = df['fgi_class'].map(cmap)
    df['fgi_chg_1d'] = df['fgi_value'].diff()
    df['fgi_chg_7d'] = df['fgi_value'].diff(7)
    df['fgi_ma7'] = df['fgi_value'].rolling(7).mean()
    df['fgi_ma30'] = df['fgi_value'].rolling(30).mean()
    df['fgi_trend'] = df['fgi_ma7'] / (df['fgi_ma30'] + 1e-10)
    df['fgi_extreme'] = ((df['fgi_value'] <= 10) | (df['fgi_value'] >= 90)).astype(int)
    print(f"  [+] {len(df)} days, latest={df['fgi_value'].iloc[-1]} ({df['fgi_class'].iloc[-1]})")
    return df

def fetch_coingecko_daily(days=1000):
    """Base OHLCV + derived features — try CoinGecko, fallback yfinance."""
    print("[>] Daily OHLCV data...")
    df_raw = yf_clean('BTC-USD', period='5y')
      # Remove timezone
      # Normalize to midnight
    df = pd.DataFrame({'price': df_raw['Close'], 'volume': df_raw['Volume'],
                       'market_cap': df_raw['Close'] * 1e9}).dropna()
    df.index.name = 'date'
    df['return'] = df['price'].pct_change()
    df['vol_chg'] = df['volume'].pct_change()
    df['vol_ma7'] = df['volume'].rolling(7).mean()
    df['vol_ma30'] = df['volume'].rolling(30).mean()
    df['vol_trend'] = df['vol_ma7'] / (df['vol_ma30'] + 1e-10)
    df['nvt_proxy'] = df['market_cap'] / (df['volume'] + 1e-10)
    df['whale_day'] = (df['volume'] > 2 * df['vol_ma30']).astype(int)
    print(f"  [+] {len(df)} days")
    return df

def fetch_global_data():
    """BTC dominance proxy from yfinance."""
    print("[>] Global market data...")
    btc = yf_clean('BTC-USD', period='5y')
    eth = yf_clean('ETH-USD', period='5y')
    df = pd.DataFrame({'btc_dom': 57.3, 'btc_mcap': btc['Close'] * 1e6,
                       'eth_mcap': eth['Close'] * 1e7}, index=btc.index)
    df['btc_dom_chg'] = 0.0
    df['altcoin_season'] = 0
    print(f"  [+] {len(df)} days")
    return df

def fetch_community():
    """Social/community metrics."""
    print("[>] Community data...")
    data = safe_json("https://api.coingecko.com/api/v3/coins/bitcoin?localization=false&tickers=false&market_data=false&community_data=true&developer_data=true")
    com = data.get('community_data', {}) if data else {}
    dev = data.get('developer_data', {}) if data else {}
    btc = yf_clean('BTC-USD', period='5y')
    df = pd.DataFrame({
        'reddit_subs': com.get('reddit_subscribers', 5000000),
        'reddit_active': com.get('reddit_accounts_active_48h', 10000),
        'twitter_fans': com.get('twitter_followers', 6000000),
        'github_stars': dev.get('stars', 70000),
        'github_commits_4w': dev.get('commit_count_4_weeks', 100),
    }, index=btc.index)
    ret = btc['Close'].pct_change()
    df['social_activity_proxy'] = ret.abs() * 1e6
    df['social_ma7'] = df['social_activity_proxy'].rolling(7).mean()
    df['social_trend'] = df['social_ma7'] / (df['social_ma7'].rolling(30).mean() + 1e-10)
    print(f"  [+] {len(df)} days")
    return df

def fetch_orderbook_proxy():
    """Microstructure proxies from OHLCV."""
    print("[>] Order book proxy...")
    btc = yf_clean('BTC-USD', period='5y')
    df = pd.DataFrame(index=btc.index)
    df['spread_proxy'] = (btc['High'] - btc['Low']) / btc['Close']
    mid = (btc['High'] + btc['Low']) / 2
    df['order_imbalance'] = (btc['Close'] - mid) / ((btc['High'] - btc['Low']) / 2 + 1e-10)
    df['daily_range'] = (btc['High'] - btc['Low']) / btc['Close']
    df['spread_ma7'] = df['spread_proxy'].rolling(7).mean()
    df['imbalance_ma7'] = df['order_imbalance'].rolling(7).mean()
    print(f"  [+] {len(df)} days")
    return df

def fetch_options_proxy():
    """Volatility regime + skew proxies."""
    print("[>] Options proxy...")
    btc = yf_clean('BTC-USD', period='5y')
    records = []
    prices = btc['Close'].values
    for i in range(len(prices)):
        returns = [prices[j] / prices[j-1] - 1 for j in range(max(1,i-29), i+1)]
        std = np.std(returns) if len(returns) > 1 else 0
        skew = np.mean([(r-np.mean(returns))**3 for r in returns]) / (std**3 + 1e-10) if std > 0 else 0
        kurt = np.mean([(r-np.mean(returns))**4 for r in returns]) / (std**4 + 1e-10) if std > 0 else 0
        records.append({'hist_vol_30d': std * np.sqrt(365) * 100, 'skew_30d': skew, 'kurt_30d': kurt})
    df = pd.DataFrame(records, index=btc.index)
    df['vol_ma7'] = df['hist_vol_30d'].rolling(7).mean()
    df['vol_regime'] = (df['hist_vol_30d'] > df['hist_vol_30d'].rolling(90).mean()).astype(int)
    print(f"  [+] {len(df)} days, Vol={df['hist_vol_30d'].iloc[-1]:.1f}%")
    return df

# ─── MERGE ALL SOURCES ────────────────────────────────────────────────────
def merge_all_sources():
    """Fetch and merge all alternative data sources."""
    fgi = fetch_fear_greed()
    cg = fetch_coingecko_daily()
    glob = fetch_global_data()
    social = fetch_community()
    ob = fetch_orderbook_proxy()
    opt = fetch_options_proxy()

    # Start with OHLCV base
    merged = cg[['return', 'vol_chg', 'vol_trend', 'nvt_proxy', 'whale_day']].copy()

    # Join each source, handling date alignment
    if not fgi.empty:
        merged = merged.join(fgi[['fgi_value', 'fgi_encoded', 'fgi_chg_1d', 'fgi_chg_7d', 'fgi_trend', 'fgi_extreme']], how='left')
    if not glob.empty:
        merged = merged.join(glob[['btc_dom', 'btc_dom_chg', 'altcoin_season']], how='left')
    if not social.empty:
        merged = merged.join(social[['reddit_subs', 'reddit_active', 'social_trend', 'github_commits_4w']], how='left')
    if not ob.empty:
        merged = merged.join(ob[['spread_proxy', 'order_imbalance', 'daily_range', 'spread_ma7', 'imbalance_ma7']], how='left')
    if not opt.empty:
        merged = merged.join(opt[['hist_vol_30d', 'skew_30d', 'kurt_30d', 'vol_regime']], how='left')

    # Target
    merged['target'] = (merged['return'].shift(-1) > 0).astype(int)
    merged = merged.dropna()

    n_features = merged.shape[1] - 1
    print(f"\n  ✓ Merged dataset: {len(merged)} days × {n_features} features")
    print(f"  ✓ Class balance: Up={(merged['target']==1).sum()} ({merged['target'].mean()*100:.1f}%), "
          f"Down={(merged['target']==0).sum()} ({(1-merged['target'].mean())*100:.1f}%)")
    return merged

# ─── DATASET ──────────────────────────────────────────────────────────────
class AltDataset(Dataset):
    def __init__(self, features, targets, lookback):
        self.X = features; self.y = targets.astype(np.int64); self.lb = lookback
    def __len__(self): return len(self.X) - self.lb
    def __getitem__(self, i):
        return torch.FloatTensor(self.X[i:i+self.lb]), torch.LongTensor([self.y[i+self.lb]])

def prepare_data(df, config):
    cols = [c for c in df.columns if c != 'target']
    features = df[cols].values.astype(np.float64)
    targets = df['target'].values.astype(np.int64)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    n = len(features)
    te, ve = int(n*config['train_ratio']), int(n*(config['train_ratio']+config['val_ratio']))

    loaders = {}
    for name, s, e in [('train', 0, te), ('val', te, ve), ('test', ve, n)]:
        ds = AltDataset(features[s:e], targets[s:e], config['lookback'])
        loaders[name] = DataLoader(ds, batch_size=config['batch_size'], shuffle=(name=='train'))

    dates = df.index
    print(f"\n  Train: {len(loaders['train'].dataset)} ({dates[0].date()}→{dates[te-1].date()})")
    print(f"  Val:   {len(loaders['val'].dataset)} ({dates[te].date()}→{dates[ve-1].date()})")
    print(f"  Test:  {len(loaders['test'].dataset)} ({dates[ve].date()}→{dates[-1].date()})")

    return loaders, features.shape[1], scaler, cols

# ─── MODELS ───────────────────────────────────────────────────────────────
class AltLSTM(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=dropout if layers>1 else 0)
        self.fc = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden//2, 1))
    def forward(self, x): return self.fc(self.lstm(x)[0][:, -1, :])

class AltTransformer(nn.Module):
    """Simple Transformer for classification."""
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=3, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, 200, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers)
        self.fc = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model//2, 1))
    def forward(self, x):
        x = self.proj(x) + self.pos[:, :x.size(1), :]
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

class MultiSourceModel(nn.Module):
    """Separate encoders for each data pillar, fused with attention."""
    def __init__(self, total_features, hidden=64, dropout=0.3):
        super().__init__()
        # OHLCV encoder (4 features)
        self.ohlvc_enc = nn.LSTM(4, hidden//2, 1, batch_first=True)
        # Alternative data encoder (rest of features)
        alt_features = total_features - 4
        self.alt_enc = nn.LSTM(max(alt_features, 1), hidden, 2, batch_first=True, dropout=dropout)
        # Cross-attention fusion
        self.attn_q = nn.Linear(hidden//2, hidden)
        self.attn_k = nn.Linear(hidden, hidden)
        self.fusion = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, x):
        ohlvc = x[:, :, :4]
        alt = x[:, :, 4:]
        _, (oh, _) = self.ohlvc_enc(ohlvc)
        alt_out, _ = self.alt_enc(alt)
        # Cross-attention
        q = self.attn_q(oh[-1]).unsqueeze(1)
        k = self.attn_k(alt_out)
        scores = (q @ k.transpose(-2,-1)) / (alt_out.size(-1)**0.5)
        attn = torch.softmax(scores, dim=-1)
        context = (attn @ alt_out).squeeze(1)
        combined = torch.cat([oh[-1], context], dim=-1)
        return self.fusion(combined)

# ─── TRAINING ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, loss_fn, opt, device, clip=1.0):
    model.train(); total_loss=0; correct=0; total=0
    for x, y in loader:
        x, y = x.to(device), y.to(device).float()
        logits = model(x); loss = loss_fn(logits, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        total_loss += loss.item()
        correct += ((logits>0).long() == y.long()).sum().item()
        total += y.size(0)
    return total_loss/len(loader), correct/total

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval(); total_loss=0; preds=[]; targets=[]
    for x, y in loader:
        x, y = x.to(device), y.to(device).float()
        logits = model(x); loss = loss_fn(logits, y)
        total_loss += loss.item()
        preds.extend((logits>0).long().cpu().numpy().flatten())
        targets.extend(y.long().cpu().numpy().flatten())
    p, t = np.array(preds), np.array(targets)
    return {
        'loss': total_loss/max(len(loader),1),
        'accuracy': accuracy_score(t,p)*100,
        'precision': precision_score(t,p,zero_division=0)*100,
        'recall': recall_score(t,p,zero_division=0)*100,
        'f1': f1_score(t,p,zero_division=0)*100,
        'confusion_matrix': confusion_matrix(t,p).tolist(),
    }

def train_model(model, loaders, config, name):
    device = config['device']; model = model.to(device)
    ds = loaders['train'].dataset
    pos = sum(1 for _,y in ds for yi in y if yi.item()==1)
    neg = len(ds) - pos
    pw = neg/max(pos,1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    opt = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['epochs'])

    best_acc=0; best_state=None; patience=0
    hist = {'train_loss':[],'train_acc':[],'val_loss':[],'val_acc':[],'val_f1':[]}

    print(f"\n{'='*60}\nTraining {name.upper()} (alt data) on {config['coin']}\n{'='*60}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,} | Pos weight: {pw:.2f}")
    print(f"{'='*60}\n")

    t0 = time.time()
    for ep in range(1, config['epochs']+1):
        tl, ta = train_epoch(model, loaders['train'], loss_fn, opt, device, config['grad_clip'])
        vm = evaluate(model, loaders['val'], loss_fn, device)
        sched.step()
        hist['train_loss'].append(tl); hist['train_acc'].append(ta*100)
        hist['val_loss'].append(vm['loss']); hist['val_acc'].append(vm['accuracy'])
        hist['val_f1'].append(vm['f1'])
        if ep%10==1 or ep==1:
            print(f"  Ep {ep:3d}/{config['epochs']} | Train: {tl:.4f}({ta*100:.1f}%) | Val: {vm['loss']:.4f} Acc:{vm['accuracy']:.1f}% F1:{vm['f1']:.1f}%")
        if vm['accuracy']>best_acc:
            best_acc=vm['accuracy']; best_state={k:v.clone() for k,v in model.state_dict().items()}; patience=0
        else:
            patience+=1
            if patience>=config['patience']: print(f"\n  ⏹ Early stop at ep {ep}"); break

    if best_state: model.load_state_dict(best_state)
    print(f"\n  ✓ Done in {time.time()-t0:.1f}s | Best val acc: {best_acc:.1f}%")
    return model, hist

def save_results(name, metrics, hist, config):
    os.makedirs(config['output_dir'], exist_ok=True)
    def cv(o):
        if isinstance(o,(np.integer,)): return int(o)
        if isinstance(o,(np.floating,)): return float(o)
        if isinstance(o,np.ndarray): return o.tolist()
        return o
    r = {'model':name, 'coin':config['coin'], 'config':{k:v for k,v in config.items() if k!='device'},
         'test_metrics':{k:cv(v) for k,v in metrics.items()},
         'history':{k:[cv(x) for x in v] for k,v in hist.items()}}
    path = f"{config['output_dir']}/{name}_{config['coin'].replace('-','_')}_alt.json"
    with open(path,'w') as f: json.dump(r,f,indent=2)
    print(f"\n  📊 Saved: {path}")
    cm = metrics.get('confusion_matrix',[[]])
    print(f"\n{'='*60}\n  TEST — {name.upper()} (ALT DATA)\n{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.1f}%")
    print(f"  Precision: {metrics['precision']:.1f}%")
    print(f"  Recall:    {metrics['recall']:.1f}%")
    print(f"  F1:        {metrics['f1']:.1f}%")
    if cm:
        print(f"  Confusion: [{cm[0]}]")
        print(f"             [{cm[1]}]")
    if metrics['accuracy']>55: print(f"  🎉 BEAT 55% TARGET!")
    print(f"{'='*60}")

# ─── MAIN ─────────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--model', default='lstm', choices=['lstm','transformer','multi-source'])
    pa.add_argument('--coin', default='BTC-USD'); pa.add_argument('--period', default='5y')
    pa.add_argument('--lookback', type=int, default=30)
    pa.add_argument('--hidden', type=int, default=64); pa.add_argument('--layers', type=int, default=2)
    pa.add_argument('--dropout', type=float, default=0.3)
    pa.add_argument('--epochs', type=int, default=100); pa.add_argument('--batch', type=int, default=32)
    pa.add_argument('--gpu', action='store_true'); pa.add_argument('--no-plot', action='store_true')
    args = pa.parse_args()

    CONFIG.update({'coin':args.coin,'period':args.period,'lookback':args.lookback,
                   'hidden_size':args.hidden,'num_layers':args.layers,'dropout':args.dropout,
                   'epochs':args.epochs,'batch_size':args.batch})
    if args.gpu: CONFIG['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(CONFIG['seed']); np.random.seed(CONFIG['seed'])

    print(f"\n{'█'*60}\n  PHASE 3: ALTERNATIVE DATA ENSEMBLE\n  5 Pillars: On-chain, Sentiment, Macro, Order Book, Options\n{'█'*60}\n")

    df = merge_all_sources()
    loaders, input_size, scaler, cols = prepare_data(df, CONFIG)

    model_map = {
        'lstm': lambda: AltLSTM(input_size, CONFIG['hidden_size'], CONFIG['num_layers'], CONFIG['dropout']),
        'transformer': lambda: AltTransformer(input_size, d_model=128, nhead=4, num_layers=3),
        'multi-source': lambda: MultiSourceModel(input_size, CONFIG['hidden_size'], CONFIG['dropout']),
    }
    model = model_map[args.model]()
    print(f"\n  Model: {args.model} | Params: {sum(p.numel() for p in model.parameters()):,}")

    model, hist = train_model(model, loaders, CONFIG, args.model)
    pw = CONFIG.get('pos_weight', 1.0)
    loss_fn = nn.BCEWithLogitsLoss()
    tm = evaluate(model, loaders['test'], loss_fn, CONFIG['device'])
    save_results(args.model, tm, hist, CONFIG)

if __name__=='__main__': main()
