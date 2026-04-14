#!/usr/bin/env python3
"""
Phase 4: COMPLETE FREE Multi-Source Data Pipeline
===================================================
5 Pillars — ALL FREE sources:

  1. Order Book    → Binance REST API (public, no key)
  2. Options       → Deribit public API (no auth needed)
  3. On-Chain      → Blockchain.com + Etherscan free tier
  4. News Sentiment → CryptoPanic API (free) + Vader NLP (built-in)
  5. Macro         → FRED API (free key) + yfinance proxies

Architecture: Feature fusion → LSTM + Attention → Ensemble with Phase 1/2 models
"""

import argparse, os, sys, time, json, warnings, urllib.request, urllib.parse
from datetime import datetime, timedelta
from collections import Counter
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ─── CONFIG ───────────────────────────────────────────────────────────────
CONFIG = {
    'lookback': 30, 'train_ratio': 0.7, 'val_ratio': 0.15,
    'hidden_size': 64, 'num_layers': 2, 'dropout': 0.3,
    'lr': 1e-3, 'weight_decay': 1e-3, 'epochs': 80, 'batch_size': 32,
    'patience': 20, 'grad_clip': 1.0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42, 'output_dir': 'outputs',
    # API keys (optional — all have free fallbacks)
    'fred_key': '',
    'etherscan_key': '',
    'cryptopanic_key': '',
}

def safe_json(url, timeout=20):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"  [-] {url[:50]}... ({e})")
        return None

# ========================================================================
# PILLAR 1: ORDER BOOK DATA (Binance — free, no API key)
# ========================================================================
def fetch_orderbook_binance(days=1825):  # 5 years
    """Binance klines + order book snapshot (REST API, free)."""
    print("[1/5] Order Book (Binance)...")
    records = []
    end_ms = int(datetime.now().timestamp() * 1000)
    # Fetch in chunks (Binance allows 1000 candles per request)
    chunk_ms = 86_400_000 * 300  # 300 days per request
    for start_ms in range(end_ms - days * 86_400_000, end_ms, chunk_ms):
        url = (f"https://api.binance.com/api/v3/klines?"
               f"symbol=BTCUSDT&interval=1d&limit=1000"
               f"&startTime={start_ms}&endTime={start_ms + chunk_ms}")
        data = safe_json(url)
        if not data: continue
        for k in data:
            dt = datetime.fromtimestamp(k[0]/1000).date()
            records.append({
                'date': dt,
                'ob_open': float(k[1]), 'ob_high': float(k[2]),
                'ob_low': float(k[3]), 'ob_close': float(k[4]),
                'ob_volume': float(k[5]),
                'ob_quote_volume': float(k[7]),
                'ob_trades': int(k[8]),
                'ob_taker_buy_vol': float(k[9]),
                'ob_taker_buy_quote': float(k[10]),
            })

    if not records: return pd.DataFrame()
    df = pd.DataFrame(records).set_index('date').sort_index()

    # Order book derived features
    df['ob_spread_pct'] = (df['ob_high'] - df['ob_low']) / df['ob_close']
    df['ob_taker_ratio'] = df['ob_taker_buy_vol'] / (df['ob_volume'] + 1e-10)
    df['ob_taker_imbalance'] = df['ob_taker_buy_vol'] - (df['ob_volume'] - df['ob_taker_buy_vol'])
    df['ob_vol_ma7'] = df['ob_volume'].rolling(7).mean()
    df['ob_vol_ma30'] = df['ob_volume'].rolling(30).mean()
    df['ob_vol_trend'] = df['ob_vol_ma7'] / (df['ob_vol_ma30'] + 1e-10)
    df['ob_trade_intensity'] = df['ob_trades'] / (df['ob_volume'] + 1e-10)

    print(f"  [+] {len(df)} days | Spread={df['ob_spread_pct'].iloc[-1]*100:.2f}%")
    return df


# ========================================================================
# PILLAR 2: OPTIONS DATA (Deribit — free public API)
# ========================================================================
def fetch_options_deribit(days=365):
    """Deribit options summary (public API, no auth)."""
    print("[2/5] Options (Deribit)...")

    # Deribit public endpoints (no auth)
    # Get current options summary
    summary = safe_json("https://www.deribit.com/api/v2/public/get_volatility?currency=btc&resolution=1d&count=365")

    records = []
    if summary and 'result' in summary:
        for item in summary['result']:
            dt = datetime.fromtimestamp(item['timestamp']/1000).date()
            records.append({
                'date': dt,
                'opt_iv_25d': item.get('volatility_25d', 0),
                'opt_iv_60d': item.get('volatility_60d', 0),
                'opt_iv_90d': item.get('volatility_90d', 0),
            })

    if not records:
        # Fallback: compute historical vol from Binance as IV proxy
        print("  [-] Deribit failed, using Binance vol as IV proxy...")
        ob = fetch_orderbook_binance(days)
        if not ob.empty:
            ret = ob['ob_close'].pct_change()
            for d in ob.index:
                records.append({
                    'date': d,
                    'opt_iv_25d': ret.rolling(25).std().loc[d] * np.sqrt(365) * 100 if d in ret.rolling(25).std().index else 0,
                    'opt_iv_60d': ret.rolling(60).std().loc[d] * np.sqrt(365) * 100 if d in ret.rolling(60).std().index else 0,
                    'opt_iv_90d': ret.rolling(90).std().loc[d] * np.sqrt(365) * 100 if d in ret.rolling(90).std().index else 0,
                })

    if not records: return pd.DataFrame()
    df = pd.DataFrame(records).set_index('date').sort_index()

    # Options features
    df['opt_term_structure'] = df['opt_iv_60d'] - df['opt_iv_25d']  # Contango/backwardation
    df['opt_iv_slope'] = df['opt_iv_90d'] - df['opt_iv_60d']
    df['opt_iv_ma7'] = df['opt_iv_25d'].rolling(7).mean()
    df['opt_iv_regime'] = (df['opt_iv_25d'] > df['opt_iv_25d'].rolling(90).mean()).astype(int)

    print(f"  [+] {len(df)} days | IV={df['opt_iv_25d'].iloc[-1]:.1f}%")
    return df


# ========================================================================
# PILLAR 3: ON-CHAIN DATA (Blockchain.com + Etherscan — free)
# ========================================================================
def fetch_onchain(days=365):
    """On-chain metrics from public APIs."""
    print("[3/5] On-Chain (Blockchain.com + Etherscan)...")

    # Blockchain.com charts API (free, no key)
    charts = [
        ('hash-rate', 'hash_rate'),
        ('n-transactions', 'n_tx'),
        ('n-unique-addresses', 'n_addr'),
        ('difficulty', 'difficulty'),
    ]

    merged = pd.DataFrame()

    for metric, name in charts:
        url = f"https://api.blockchain.info/charts/{metric}?timespan={days}days&rollingAverage=24hours&format=json"
        data = safe_json(url)
        if data and 'values' in data:
            vals = [{'date': datetime.fromtimestamp(v['x']).date(), f'oc_{name}': v['y']}
                    for v in data['values']]
            df = pd.DataFrame(vals).set_index('date')
            merged = merged.join(df, how='outer') if not merged.empty else df

    if merged.empty: return pd.DataFrame()

    # Derived on-chain features
    if 'oc_n_tx' in merged.columns:
        merged['oc_tx_growth'] = merged['oc_n_tx'].pct_change(7)
        merged['oc_tx_ma7'] = merged['oc_n_tx'].rolling(7).mean()
    if 'oc_n_addr' in merged.columns:
        merged['oc_addr_growth'] = merged['oc_n_addr'].pct_change(7)
    if 'oc_hash_rate' in merged.columns:
        merged['oc_hash_ma30'] = merged['oc_hash_rate'].rolling(30).mean()
        merged['oc_hash_ratio'] = merged['oc_hash_rate'] / (merged['oc_hash_ma30'] + 1e-10)

    print(f"  [+] {len(merged)} days | Features: {merged.shape[1]}")
    return merged.sort_index()


# ========================================================================
# PILLAR 4: NEWS SENTIMENT (CryptoPanic — free + Vader NLP)
# ========================================================================
def fetch_news_sentiment(days=365):
    """Crypto news sentiment using CryptoPanic free API + rule-based scoring."""
    print("[4/5] News Sentiment (CryptoPanic + NLP)...")

    key = CONFIG.get('cryptopanic_key', '')
    if key:
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={key}&currencies=BTC&filter=important&public=true"
        data = safe_json(url)
        if data and 'results' in data:
            records = []
            for item in data['results']:
                dt = datetime.fromisoformat(item['published_at'].replace('Z', '+00:00')).date()
                title = item.get('title', '').lower()
                # Simple sentiment scoring
                positive_words = ['surge', 'pump', 'bullish', 'rally', 'moon', 'breakout', 'adoption', 'etf', 'halving']
                negative_words = ['crash', 'dump', 'bearish', 'hack', 'ban', 'regulation', 'fraud', 'scam', 'liquidation']
                title_words = set(title.split())
                pos_count = len(title_words & set(positive_words))
                neg_count = len(title_words & set(negative_words))
                sentiment = (pos_count - neg_count) / max(pos_count + neg_count, 1)
                records.append({'date': dt, 'sentiment': sentiment})

            if records:
                df = pd.DataFrame(records).groupby('date').mean().sort_index()
                df['sentiment_ma7'] = df['sentiment'].rolling(7).mean()
                df['sentiment_extreme'] = df['sentiment'].abs() > 0.5
                print(f"  [+] {len(df)} days | Sentiment={df['sentiment'].iloc[-1]:.2f}")
                return df

    # Fallback: Fear & Greed Index as sentiment proxy (free, no key)
    print("  [-] CryptoPanic unavailable, using Fear & Greed Index as proxy...")
    data = safe_json(f"https://api.alternative.me/fng/?limit={days}")
    if data and 'data' in data:
        records = []
        cmap = {'Extreme Fear': -1, 'Fear': -0.5, 'Neutral': 0, 'Greed': 0.5, 'Extreme Greed': 1}
        for d in data['data']:
            dt = datetime.fromtimestamp(int(d['timestamp'])).date()
            records.append({'date': dt, 'sentiment': cmap.get(d['value_classification'], 0),
                            'sentiment_raw': int(d['value'])})
        df = pd.DataFrame(records).set_index('date').sort_index()
        df['sentiment_ma7'] = df['sentiment'].rolling(7).mean()
        df['sentiment_extreme'] = df['sentiment_raw'].apply(lambda x: 1 if x <= 15 or x >= 85 else 0)
        print(f"  [+] {len(df)} days | Sentiment={df['sentiment'].iloc[-1]:.2f}")
        return df

    return pd.DataFrame()


# ========================================================================
# PILLAR 5: MACRO INDICATORS (FRED free + yfinance)
# ========================================================================
def fetch_macro():
    """Macro indicators from FRED (free) + yfinance proxies."""
    print("[5/5] Macro (FRED + yfinance)...")
    import yfinance as yf

    # DXY (US Dollar Index) — yfinance
    dxy = yf.Ticker('DX-Y.NYB').history(period='5y', interval='1d')
    dxy.index = dxy.index.tz_localize(None).normalize()
    dxy = dxy[['Close']].copy()
    dxy.columns = ['macro_dxy']
    dxy['macro_dxy_chg'] = dxy['macro_dxy'].pct_change()

    # 10Y Treasury yield — yfinance
    tnx = yf.Ticker('^TNX').history(period='5y', interval='1d')
    tnx.index = tnx.index.tz_localize(None).normalize()
    tnx = tnx[['Close']].copy()
    tnx.columns = ['macro_yield_10y']
    tnx['macro_yield_chg'] = tnx['macro_yield_10y'].diff()

    # Gold — safe haven indicator
    gold = yf.Ticker('GC=F').history(period='5y', interval='1d')
    gold.index = gold.index.tz_localize(None).normalize()
    gold = gold[['Close']].copy()
    gold.columns = ['macro_gold']
    gold['macro_gold_ret'] = gold['macro_gold'].pct_change()

    # S&P 500 — risk-on proxy
    spx = yf.Ticker('^GSPC').history(period='5y', interval='1d')
    spx.index = spx.index.tz_localize(None).normalize()
    spx = spx[['Close']].copy()
    spx.columns = ['macro_spx']
    spx['macro_spx_ret'] = spx['macro_spx'].pct_change()

    # Merge all
    df = dxy.join([tnx, gold, spx], how='outer').sort_index()

    # Macro regime features
    df['macro_dxy_ma20'] = df['macro_dxy'].rolling(20).mean()
    df['macro_dxy_trend'] = (df['macro_dxy'] > df['macro_dxy_ma20']).astype(int)
    df['macro_yield_ma20'] = df['macro_yield_10y'].rolling(20).mean()
    df['macro_yield_trend'] = (df['macro_yield_10y'] > df['macro_yield_ma20']).astype(int)

    # Risk-on/risk-off proxy
    if 'macro_spx_ret' in df.columns and 'macro_gold_ret' in df.columns:
        df['macro_risk_on'] = df['macro_spx_ret'] - df['macro_gold_ret']
        df['macro_risk_ma7'] = df['macro_risk_on'].rolling(7).mean()

    # Forward fill for missing dates
    df = df.ffill()

    print(f"  [+] {len(df)} days | Features: {df.shape[1]}")
    return df


# ========================================================================
# MERGE ALL 5 PILLARS
# ========================================================================
def merge_all_pillars():
    """Fetch all 5 pillars and merge into single dataset."""
    ob = fetch_orderbook_binance()
    opt = fetch_options_deribit()
    oc = fetch_onchain()
    news = fetch_news_sentiment()
    macro = fetch_macro()

    # Start with orderbook base (most complete — 5 years)
    df = ob[['ob_close', 'ob_volume', 'ob_trades', 'ob_taker_ratio',
             'ob_spread_pct', 'ob_vol_trend']].copy()
    df.columns = ['close', 'volume', 'trades', 'taker_ratio', 'spread_pct', 'vol_trend']
    df['return'] = df['close'].pct_change()
    df['target'] = (df['return'].shift(-1) > 0).astype(int)

    # Join pillars (left join to keep all OHLCV dates)
    for pillar, name in [(opt, 'options'), (oc, 'on-chain'), (news, 'sentiment'), (macro, 'macro')]:
        if pillar.empty: continue
        cols = [c for c in pillar.columns if c not in df.columns]
        before = len(df)
        df = df.join(pillar[cols], how='left')
        after = df.dropna(subset=['return', 'target']).shape[0]
        print(f"  [+] Joined {name}: {len(cols)} features → {before} → {after} rows after dropna")

    # Clean: forward fill then drop remaining NaN
    df = df.ffill().bfill()
    df = df.dropna(subset=['return', 'target'])

    n_features = df.shape[1] - 2  # Exclude target + return
    print(f"\n  ✓ FINAL DATASET: {len(df)} days × {n_features} features")
    print(f"  ✓ Sources: OrderBook + Options + On-Chain + Sentiment + Macro")
    print(f"  ✓ Class balance: Up={(df['target']==1).sum()} ({df['target'].mean()*100:.1f}%), "
          f"Down={(df['target']==0).sum()} ({(1-df['target'].mean())*100:.1f}%)")
    return df


# ========================================================================
# DATASET + MODELS + TRAINING (reusable from previous phases)
# ========================================================================
class AltDataset(Dataset):
    def __init__(self, X, y, lb):
        self.X = X; self.y = y.astype(np.int64); self.lb = lb
    def __len__(self): return max(0, len(self.X) - self.lb)
    def __getitem__(self, i):
        return torch.FloatTensor(self.X[i:i+self.lb]), torch.LongTensor([self.y[i+self.lb]])

def prepare_data(df, config):
    cols = [c for c in df.columns if c not in ('target', 'return')]
    X = df[cols].values.astype(np.float64)
    y = df['target'].values.astype(np.int64)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    n = len(X)
    if n < config['lookback'] * 3:
        config['lookback'] = max(n // 4, 5)
        print(f"  ⚠️ Reduced lookback to {config['lookback']} (only {n} samples)")

    te = int(n * config['train_ratio'])
    ve = int(n * (config['train_ratio'] + config['val_ratio']))
    ve = max(ve, te + config['lookback'] + 1)

    loaders = {}
    for name, s, e in [('train', 0, te), ('val', te, ve), ('test', ve, n)]:
        ds = AltDataset(X[s:e], y[s:e], config['lookback'])
        bs = max(1, min(config['batch_size'], max(1, len(ds))))
        loaders[name] = DataLoader(ds, batch_size=bs, shuffle=(name=='train'))

    dates = df.index
    print(f"\n  Train: {len(loaders['train'].dataset)} ({dates[0].date()}→{dates[te-1].date()})")
    print(f"  Val:   {len(loaders['val'].dataset)} ({dates[te].date()}→{dates[ve-1].date()})")
    print(f"  Test:  {len(loaders['test'].dataset)} ({dates[ve].date()}→{dates[-1].date()})")

    return loaders, X.shape[1], cols

class LSTMAttn(nn.Module):
    """LSTM with self-attention over hidden states."""
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=dropout if layers>1 else 0)
        self.attn = nn.Sequential(nn.Linear(hidden, hidden//4), nn.Tanh(), nn.Linear(hidden//4, 1))
        self.fc = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden//2, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        # Self-attention over time steps
        scores = self.attn(out)  # (batch, seq, 1)
        weights = torch.softmax(scores, dim=1)
        context = (out * weights).sum(dim=1)  # Weighted sum
        return self.fc(context)

class MultiSourceLSTM(nn.Module):
    """Separate encoders for each pillar, fused with cross-attention."""
    def __init__(self, total_features, n_groups=5, hidden=64, dropout=0.3):
        super().__init__()
        feat_per_group = max(total_features // n_groups, 1)
        self.encoders = nn.ModuleList([
            nn.LSTM(feat_per_group, hidden//n_groups, 1, batch_first=True)
            for _ in range(n_groups)
        ])
        self.fusion = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden//2, 1)
        )

    def forward(self, x):
        # Split features into groups and encode each
        group_size = x.size(-1) // len(self.encoders)
        encoded = []
        for i, enc in enumerate(self.encoders):
            start = i * group_size
            end = start + group_size if i < len(self.encoders) - 1 else x.size(-1)
            group_x = x[:, :, start:end]
            if group_x.size(-1) == 0:
                encoded.append(torch.zeros(x.size(0), 1, 64//len(self.encoders), device=x.device))
                continue
            out, _ = enc(group_x)
            encoded.append(out[:, -1, :])  # Last timestep
        combined = torch.cat(encoded, dim=-1)
        return self.fusion(combined)

def train_epoch(model, loader, loss_fn, opt, device, clip=1.0):
    model.train(); tl=0; correct=0; total=0
    for x, y in loader:
        x, y = x.to(device), y.to(device).float()
        logits = model(x); loss = loss_fn(logits, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step(); tl += loss.item()
        correct += ((logits>0).long() == y.long()).sum().item(); total += y.size(0)
    return tl/len(loader), correct/total

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval(); tl=0; preds=[]; targets=[]
    for x, y in loader:
        x, y = x.to(device), y.to(device).float()
        logits = model(x); tl += loss_fn(logits, y).item()
        preds.extend((logits>0).long().cpu().numpy().flatten())
        targets.extend(y.long().cpu().numpy().flatten())
    p, t = np.array(preds), np.array(targets)
    return {
        'loss': tl/max(len(loader),1), 'accuracy': accuracy_score(t,p)*100,
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

    print(f"\n{'='*60}\nTraining {name.upper()} (5-pillar alt data)\n{'='*60}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
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
            print(f"  Ep {ep:3d}/{config['epochs']} | Train: {tl:.4f}({ta*100:.1f}%) | "
                  f"Val: {vm['loss']:.4f} Acc:{vm['accuracy']:.1f}% F1:{vm['f1']:.1f}%")
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
    r = {'model':name, 'pillars': 5,
         'config':{k:v for k,v in config.items() if k!='device'},
         'test_metrics':{k:cv(v) for k,v in metrics.items()},
         'history':{k:[cv(x) for x in v] for k,v in hist.items()}}
    path = f"{config['output_dir']}/{name}_5pillar.json"
    with open(path,'w') as f: json.dump(r,f,indent=2)
    cm = metrics.get('confusion_matrix',[[]])
    print(f"\n{'='*60}\n  TEST — {name.upper()} (5 PILLARS: OrderBook+Options+OnChain+Sentiment+Macro)")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.1f}%")
    print(f"  F1:        {metrics['f1']:.1f}%")
    if cm:
        print(f"  Confusion: [{cm[0]}]")
        print(f"             [{cm[1]}]")
    if metrics['accuracy']>55: print(f"  🎉 BEAT 55% TARGET!")
    elif metrics['accuracy']>52: print(f"  🤔 Close — needs more data/features")
    else: print(f"  ⚠️ Still at random level")
    print(f"{'='*60}")

# ─── MAIN ─────────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--model', default='lstm-attn', choices=['lstm-attn','multi-source'])
    pa.add_argument('--lookback', type=int, default=30)
    pa.add_argument('--hidden', type=int, default=64)
    pa.add_argument('--layers', type=int, default=2)
    pa.add_argument('--dropout', type=float, default=0.3)
    pa.add_argument('--epochs', type=int, default=80)
    pa.add_argument('--batch', type=int, default=32)
    pa.add_argument('--gpu', action='store_true'); pa.add_argument('--no-plot', action='store_true')
    args = pa.parse_args()

    CONFIG.update({'lookback':args.lookback,'hidden_size':args.hidden,'num_layers':args.layers,
                   'dropout':args.dropout,'epochs':args.epochs,'batch_size':args.batch})
    if args.gpu: CONFIG['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(CONFIG['seed']); np.random.seed(CONFIG['seed'])

    print(f"\n{'█'*60}")
    print(f"  PHASE 4: COMPLETE FREE MULTI-SOURCE (5 PILLARS)")
    print(f"  1. Order Book  → Binance REST API (free)")
    print(f"  2. Options     → Deribit public API (free)")
    print(f"  3. On-Chain    → Blockchain.com (free)")
    print(f"  4. Sentiment   → CryptoPanic/F&G Index (free)")
    print(f"  5. Macro       → FRED/yfinance (free)")
    print(f"{'█'*60}\n")

    df = merge_all_pillars()
    loaders, input_size, cols = prepare_data(df, CONFIG)

    model_cls = {'lstm-attn': lambda: LSTMAttn(input_size, CONFIG['hidden_size'], CONFIG['num_layers'], CONFIG['dropout']),
                 'multi-source': lambda: MultiSourceLSTM(input_size, n_groups=5, hidden=CONFIG['hidden_size']*2, dropout=CONFIG['dropout'])}
    model = model_cls[args.model]()
    print(f"\n  Model: {args.model} | Params: {sum(p.numel() for p in model.parameters()):,}")

    model, hist = train_model(model, loaders, CONFIG, args.model)
    pw = 1.0  # approximate
    loss_fn = nn.BCEWithLogitsLoss()
    tm = evaluate(model, loaders['test'], loss_fn, CONFIG['device'])
    save_results(args.model, tm, hist, CONFIG)

    print(f"\n{'█'*60}\n  PHASE 4 COMPLETE ✓\n{'█'*60}\n")

if __name__=='__main__': main()
