#!/usr/bin/env python3
"""
Phase 2.5: Daily Classification for Crypto Price Prediction
=============================================================
Strategy: Reduce noise + simpler task = better direction accuracy

Key changes from Phase 2:
  1. DAILY candles (not hourly) — removes microstructure noise
  2. CLASSIFICATION (Up/Down) — easier than predicting exact return
  3. ENSEMBLE (LSTM + Linear + Volume signal) — combine complementary signals
  4. More features (7d/14d/30d returns, volume trends, ATR ratio)
  5. Weighted loss for class imbalance

Goal: Direction accuracy > 55%

Based on D2L Chapters 3, 5, 10
"""

import argparse
import os
import sys
import time
import json
import warnings
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
    'coin': 'BTC-USD',
    'period': '5y',             # 5 years of daily data
    'lookback': 30,             # 30 days lookback
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    # Model
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.3,
    'lr': 1e-3,
    'weight_decay': 1e-3,
    'epochs': 100,
    'batch_size': 32,
    'patience': 25,
    'grad_clip': 1.0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'output_dir': 'outputs',
}


# ─── DATA ─────────────────────────────────────────────────────────────────
def fetch_data(coin, period):
    import yfinance as yf
    print(f"[>] Fetching {coin} (daily, {period})...")
    ticker = yf.Ticker(coin)
    df = ticker.history(period=period, interval='1d')
    if df.empty:
        print(f"[-] No data for {coin}")
        sys.exit(1)
    print(f"[+] Got {len(df)} daily candles: {df.index[0].date()} → {df.index[-1].date()}")
    return df


def engineer_features_daily(df):
    """Features optimized for daily classification."""
    feat = pd.DataFrame(index=df.index)

    # Price
    feat['close'] = df['Close']
    feat['open'] = df['Open']
    feat['high'] = df['High']
    feat['low'] = df['Low']
    feat['volume'] = df['Volume']

    # Daily return
    feat['return_1d'] = feat['close'].pct_change(1)

    # Multi-scale returns
    feat['return_3d'] = feat['close'].pct_change(3)
    feat['return_7d'] = feat['close'].pct_change(7)
    feat['return_14d'] = feat['close'].pct_change(14)
    feat['return_30d'] = feat['close'].pct_change(30)

    # Log returns
    feat['log_return'] = np.log(feat['close'] / feat['close'].shift(1))

    # Volume features
    feat['volume_change'] = feat['volume'].pct_change(1)
    feat['volume_sma5'] = feat['volume'].rolling(5).mean() / (feat['volume'].rolling(20).mean() + 1e-10)
    feat['volume_sma10'] = feat['volume'].rolling(10).mean() / (feat['volume'].rolling(20).mean() + 1e-10)

    # Volatility
    feat['volatility_7d'] = feat['return_1d'].rolling(7).std()
    feat['volatility_14d'] = feat['return_1d'].rolling(14).std()
    feat['volatility_30d'] = feat['return_1d'].rolling(30).std()
    feat['atr_14'] = ((df['High'] - df['Low']).rolling(14).mean()) / df['Close']

    # Range
    feat['daily_range'] = (df['High'] - df['Low']) / (df['Close'] + 1e-10)
    feat['upper_wick'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / (df['Close'] + 1e-10)
    feat['lower_wick'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / (df['Close'] + 1e-10)

    # Moving averages
    feat['sma_5'] = df['Close'] / df['Close'].rolling(5).mean()
    feat['sma_10'] = df['Close'] / df['Close'].rolling(10).mean()
    feat['sma_20'] = df['Close'] / df['Close'].rolling(20).mean()
    feat['sma_50'] = df['Close'] / df['Close'].rolling(50).mean()

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    feat['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    feat['macd'] = ema12 - ema26
    feat['macd_signal'] = feat['macd'].ewm(span=9, adjust=False).mean()
    feat['macd_hist'] = feat['macd'] - feat['macd_signal']

    # Bollinger
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    feat['bb_position'] = (df['Close'] - sma20) / (2 * std20 + 1e-10)
    feat['bb_width'] = (2 * std20) / (sma20 + 1e-10)

    # Momentum
    feat['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    feat['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1

    # Consecutive up/down days
    up_days = (feat['return_1d'] > 0).astype(int)
    feat['consecutive_up'] = up_days.groupby((up_days != up_days.shift()).cumsum()).cumsum()
    down_days = (feat['return_1d'] < 0).astype(int)
    feat['consecutive_down'] = down_days.groupby((down_days != down_days.shift()).cumsum()).cumsum()

    # Target: Binary classification (1 = up, 0 = down)
    feat['target'] = (feat['return_1d'].shift(-1) > 0).astype(int)

    # Clean
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"[+] Features: {feat.shape[1]} columns ({feat.shape[1]-1} input + 1 target)")
    print(f"[+] Samples: {len(feat)}")
    print(f"[+] Class balance: Up={(feat['target']==1).sum()} ({(feat['target'].mean()*100):.1f}%), "
          f"Down={(feat['target']==0).sum()} ({((1-feat['target'].mean())*100):.1f}%)")

    return feat


# ─── DATASET ──────────────────────────────────────────────────────────────
class ClassificationDataset(Dataset):
    def __init__(self, features, targets, lookback):
        self.features = features
        self.targets = targets.astype(np.int64)
        self.lookback = lookback

    def __len__(self):
        return len(self.features) - self.lookback

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.lookback]
        y = self.targets[idx + self.lookback]
        return torch.FloatTensor(x), torch.LongTensor([y])


def prepare_data(df_feat, config):
    feature_cols = [c for c in df_feat.columns if c != 'target']
    features = df_feat[feature_cols].values.astype(np.float64)
    targets = df_feat['target'].values.astype(np.int64)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    n = len(features)
    train_end = int(n * config['train_ratio'])
    val_end = int(n * (config['train_ratio'] + config['val_ratio']))

    train_ds = ClassificationDataset(features[:train_end], targets[:train_end], config['lookback'])
    val_ds = ClassificationDataset(features[train_end:val_end], targets[train_end:val_end], config['lookback'])
    test_ds = ClassificationDataset(features[val_end:], targets[val_end:], config['lookback'])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)

    input_size = features.shape[1]
    dates = df_feat.index

    print(f"\n  Data split:")
    print(f"    Train: {len(train_ds)} sequences ({dates[0].date()} → {dates[train_end-1].date()})")
    print(f"    Val:   {len(val_ds)} sequences ({dates[train_end].date()} → {dates[val_end-1].date()})")
    print(f"    Test:  {len(test_ds)} sequences ({dates[val_end].date()} → {dates[-1].date()})")

    return train_loader, val_loader, test_loader, input_size, scaler, feature_cols


# ─── MODELS ───────────────────────────────────────────────────────────────

class LinearClassifier(nn.Module):
    """Linear model for directional classification."""
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size * 4, 1)  # last + mean + std + diff
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        flat = self._flatten(x)
        return self.linear(flat)

    def _flatten(self, x):
        last = x[:, -1, :]
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        diff = x[:, -1, :] - x[:, 0, :]
        return torch.cat([last, mean, std, diff], dim=1)


class MLPClassifier(nn.Module):
    """MLP for classification."""
    def __init__(self, input_size, hidden_sizes=None, dropout=0.3):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64]
        layers = []
        prev = input_size * 4
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        flat = torch.cat([x[:, -1, :], x.mean(1), x.std(1), x[:, -1, :] - x[:, 0, :]], dim=1)
        return self.network(flat)


class LSTMClassifier(nn.Module):
    """LSTM for directional classification."""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUClassifier(nn.Module):
    """GRU for directional classification."""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class EnsembleClassifier(nn.Module):
    """Ensemble of Linear + LSTM + Volume-based signal."""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.linear = LinearClassifier(input_size)
        self.lstm = LSTMClassifier(input_size, hidden_size, num_layers, dropout)
        self.meta = nn.Sequential(
            nn.Linear(2, 16),  # 2 model outputs
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        linear_out = self.linear(x)
        lstm_out = self.lstm(x)
        combined = torch.cat([linear_out, lstm_out], dim=1)
        return self.meta(combined)


# ─── TRAINING ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal Loss — handles class imbalance, focuses on hard examples."""
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets.float())
        pt = torch.exp(-bce_loss)
        focal = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal.mean()


def train_epoch(model, loader, loss_fn, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    n_batches = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device).float()

        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        preds = (logits > 0).long()
        correct += (preds == y.long()).sum().item()
        total += y.size(0)

    return total_loss / n_batches, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds = []
    all_targets = []

    for x, y in loader:
        x, y = x.to(device), y.to(device).float()
        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += loss.item()
        n_batches += 1

        preds = (logits > 0).long().cpu().numpy().flatten()
        all_preds.extend(preds)
        all_targets.extend(y.long().cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    metrics = {
        'loss': total_loss / max(n_batches, 1),
        'accuracy': accuracy_score(all_targets, all_preds) * 100,
        'precision': precision_score(all_targets, all_preds, zero_division=0) * 100,
        'recall': recall_score(all_targets, all_preds, zero_division=0) * 100,
        'f1': f1_score(all_targets, all_preds, zero_division=0) * 100,
    }

    cm = confusion_matrix(all_targets, all_preds)
    metrics['confusion_matrix'] = cm.tolist()

    return metrics


def train_model(model, train_loader, val_loader, config, model_name):
    device = config['device']
    model = model.to(device)

    # Use weighted BCE or Focal Loss for imbalanced data
    pos_count = sum(1 for _, y in train_loader.dataset for yi in y if yi.item() == 1)
    neg_count = len(train_loader.dataset) - pos_count
    pos_weight = neg_count / max(pos_count, 1)
    print(f"\n  Class weights: Up={pos_count} ({pos_count/len(train_loader.dataset)*100:.1f}%), "
          f"Down={neg_count} ({neg_count/len(train_loader.dataset)*100:.1f}%)")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    best_val_acc = 0
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} on {config['coin']} (DAILY)")
    print(f"{'='*60}")
    print(f"  Device:         {device}")
    print(f"  Hidden Size:    {config['hidden_size']}")
    print(f"  Layers:         {config['num_layers']}")
    print(f"  Dropout:        {config['dropout']}")
    print(f"  Epochs:         {config['epochs']}")
    print(f"  Pos Weight:     {pos_weight:.2f}")
    print(f"  Early Stop:     patience={config['patience']}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, loss_fn, optimizer, device, config['grad_clip']
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc * 100)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{config['epochs']} | "
                  f"Train: {train_loss:.4f} ({train_acc*100:.1f}%) | "
                  f"Val: {val_metrics['loss']:.4f} (Acc: {val_metrics['accuracy']:.1f}%, "
                  f"F1: {val_metrics['f1']:.1f}%) | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\n  ⏹ Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - start_time
    if best_state:
        model.load_state_dict(best_state)

    print(f"\n  ✓ Training complete in {elapsed:.1f}s")
    print(f"  Best val accuracy: {best_val_acc:.1f}%")

    return model, history


# ─── RESULTS ──────────────────────────────────────────────────────────────
def save_results(model_name, metrics, history, config):
    os.makedirs(config['output_dir'], exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        elif isinstance(obj, (np.floating,)): return float(obj)
        elif isinstance(obj, (np.bool_,)): return bool(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, list): return [convert(x) for x in obj]
        return obj

    results = {
        'model': model_name,
        'coin': config['coin'],
        'interval': 'daily',
        'config': {k: v for k, v in config.items() if k not in ['device']},
        'test_metrics': {k: convert(v) for k, v in metrics.items()},
        'training_history': {k: [convert(x) for x in v] for k, v in history.items()},
        'lookback': config['lookback'],
    }

    path = os.path.join(config['output_dir'], f'{model_name}_{config["coin"].replace("-", "_")}_daily.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  📊 Results saved: {path}")

    print(f"\n{'='*60}")
    print(f"  TEST RESULTS — {model_name.upper()} (DAILY)")
    print(f"{'='*60}")
    print(f"  Accuracy:        {metrics['accuracy']:.1f}%")
    print(f"  Precision:       {metrics['precision']:.1f}%")
    print(f"  Recall:          {metrics['recall']:.1f}%")
    print(f"  F1 Score:        {metrics['f1']:.1f}%")
    print(f"  Loss:            {metrics['loss']:.4f}")
    cm = metrics.get('confusion_matrix', [])
    if cm:
        print(f"  Confusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Down  Up")
        print(f"    Actual Down  {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"    Actual Up    {cm[1][0]:4d}  {cm[1][1]:4d}")
    print(f"{'='*60}")


def plot_results(history, model_name, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(config['output_dir'], exist_ok=True)
    base = os.path.join(config['output_dir'], f'{model_name}_{config["coin"].replace("-", "_")}_daily')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'{model_name.upper()} — {config["coin"]} (Daily)', fontsize=14, fontweight='bold')

    epochs = range(1, len(history['train_loss']) + 1)

    ax = axes[0]
    ax.plot(epochs, history['train_loss'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history['train_acc'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_acc'], label='Validation', linewidth=2)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Classification Accuracy')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, history['val_f1'], label='F1 Score', color='green', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('F1 Score (%)')
    ax.set_title('Validation F1 Score')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{base}_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Plots saved: {base}_plots.png")


# ─── MAIN ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Phase 2.5: Daily Classification')
    parser.add_argument('--model', type=str, default='ensemble',
                        choices=['linear', 'mlp', 'lstm', 'gru', 'ensemble'],
                        help='Model type')
    parser.add_argument('--coin', type=str, default='BTC-USD')
    parser.add_argument('--period', type=str, default='5y')
    parser.add_argument('--lookback', type=int, default=30)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    CONFIG['coin'] = args.coin
    CONFIG['period'] = args.period
    CONFIG['lookback'] = args.lookback
    CONFIG['hidden_size'] = args.hidden
    CONFIG['num_layers'] = args.layers
    CONFIG['dropout'] = args.dropout
    CONFIG['epochs'] = args.epochs
    CONFIG['batch_size'] = args.batch
    if args.gpu:
        CONFIG['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])

    print(f"\n{'█'*60}")
    print(f"  PHASE 2.5: DAILY CLASSIFICATION")
    print(f"  Strategy: Daily candles + Up/Down classification")
    print(f"  Goal: Direction accuracy > 55%")
    print(f"{'█'*60}")

    # 1. Data
    df_raw = fetch_data(CONFIG['coin'], CONFIG['period'])
    df_feat = engineer_features_daily(df_raw)
    train_loader, val_loader, test_loader, input_size, scaler, feature_cols = \
        prepare_data(df_feat, CONFIG)

    # 2. Model
    model_map = {
        'linear': lambda: LinearClassifier(input_size),
        'mlp': lambda: MLPClassifier(input_size, [128, 64], CONFIG['dropout']),
        'lstm': lambda: LSTMClassifier(input_size, CONFIG['hidden_size'], CONFIG['num_layers'], CONFIG['dropout']),
        'gru': lambda: GRUClassifier(input_size, CONFIG['hidden_size'], CONFIG['num_layers'], CONFIG['dropout']),
        'ensemble': lambda: EnsembleClassifier(input_size, CONFIG['hidden_size'], CONFIG['num_layers'], CONFIG['dropout']),
    }
    model = model_map[args.model]()

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model: {args.model.upper()}")
    print(f"  Input size: {input_size}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable: {trainable:,}")

    # 3. Train
    model, history = train_model(model, train_loader, val_loader, CONFIG, args.model)

    # 4. Evaluate
    pos_count = sum(1 for _, y in train_loader.dataset for yi in y if yi.item() == 1)
    neg_count = len(train_loader.dataset) - pos_count
    pos_weight = neg_count / max(pos_count, 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=CONFIG['device']))
    test_metrics = evaluate(model, test_loader, loss_fn, CONFIG['device'])

    # 5. Save
    save_results(args.model, test_metrics, history, CONFIG)

    # 6. Plot
    if not args.no_plot:
        try:
            plot_results(history, args.model, CONFIG)
        except Exception as e:
            print(f"\n  ⚠️ Could not generate plots: {e}")

    print(f"\n{'█'*60}")
    print(f"  PHASE 2.5 COMPLETE ✓")
    print(f"  Direction Accuracy: {test_metrics['accuracy']:.1f}%")
    if test_metrics['accuracy'] > 55:
        print(f"  🎉 GOAL ACHIEVED! (> 55%)")
    else:
        print(f"  ⚠️ Below 55% target — crypto is hard to predict")
    print(f"{'█'*60}\n")


if __name__ == '__main__':
    main()
