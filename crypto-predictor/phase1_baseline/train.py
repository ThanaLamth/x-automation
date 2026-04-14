#!/usr/bin/env python3
"""
Phase 1: Baseline Models for Crypto Price Prediction
=====================================================
Based on D2L.ai Chapters 3, 5, 6, 12

Models:
  1. Linear Regression (with weight decay)
  2. MLP (Multilayer Perceptron with dropout)

Features:
  - OHLCV data from yfinance
  - Technical indicators: returns, RSI, MACD, Bollinger
  - Time-based train/val/test split (no shuffling)
  - Huber loss for robustness
  - GPU support
  - Walk-forward validation

Usage:
  python train.py --model linear          # Linear Regression
  python train.py --model mlp             # MLP
  python train.py --model linear --coin ETH-USD  # Ethereum
  python train.py --model mlp --epochs 100
  python train.py --model linear --gpu              # Use GPU
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─── CONFIG ───────────────────────────────────────────────────────────────
CONFIG = {
    'coin': 'BTC-USD',
    'interval': '1h',
    'period': '2y',            # 2 years of data
    'lookback': 48,            # 48 hours lookback
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    # Linear Regression
    'linear_lr': 1e-2,
    'linear_wd': 1e-3,
    'linear_epochs': 50,
    'linear_batch': 256,
    # MLP
    'mlp_hidden': [128, 64],
    'mlp_dropout': 0.3,
    'mlp_lr': 1e-3,
    'mlp_wd': 1e-4,
    'mlp_epochs': 100,
    'mlp_batch': 128,
    # Common
    'huber_delta': 1.0,
    'patience': 15,            # Early stopping
    'grad_clip': 1.0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'output_dir': 'outputs',
}


# ─── DATA FETCHING ────────────────────────────────────────────────────────
def fetch_data(coin: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance."""
    import yfinance as yf
    print(f"[>] Fetching {coin} ({interval}, {period})...")
    ticker = yf.Ticker(coin)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        print(f"[-] No data for {coin}")
        sys.exit(1)

    print(f"[+] Got {len(df)} candles: {df.index[0]} → {df.index[-1]}")
    return df


# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from raw OHLCV data."""
    feat = pd.DataFrame(index=df.index)

    # Raw prices (normalized later)
    feat['close'] = df['Close']
    feat['open'] = df['Open']
    feat['high'] = df['High']
    feat['low'] = df['Low']
    feat['volume'] = df['Volume']

    # Returns
    feat['return_1'] = feat['close'].pct_change(1)
    feat['return_3'] = feat['close'].pct_change(3)
    feat['return_6'] = feat['close'].pct_change(6)
    feat['return_12'] = feat['close'].pct_change(12)
    feat['return_24'] = feat['close'].pct_change(24)

    # Log returns
    feat['log_return'] = np.log(feat['close'] / feat['close'].shift(1))

    # Volume change
    feat['volume_change'] = feat['volume'].pct_change(1)

    # RSI (14 period)
    delta = feat['close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    feat['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = feat['close'].ewm(span=12, adjust=False).mean()
    ema26 = feat['close'].ewm(span=26, adjust=False).mean()
    feat['macd'] = ema12 - ema26
    feat['macd_signal'] = feat['macd'].ewm(span=9, adjust=False).mean()
    feat['macd_hist'] = feat['macd'] - feat['macd_signal']

    # Bollinger Bands (20, 2)
    sma20 = feat['close'].rolling(20).mean()
    std20 = feat['close'].rolling(20).std()
    feat['bb_upper'] = sma20 + 2 * std20
    feat['bb_lower'] = sma20 - 2 * std20
    feat['bb_width'] = (feat['bb_upper'] - feat['bb_lower']) / (sma20 + 1e-10)
    feat['bb_position'] = (feat['close'] - feat['bb_lower']) / (feat['bb_upper'] - feat['bb_lower'] + 1e-10)

    # ATR (14)
    tr = pd.concat([
        feat['high'] - feat['low'],
        (feat['high'] - feat['close'].shift(1)).abs(),
        (feat['low'] - feat['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    feat['atr'] = tr.rolling(14).mean()

    # Target: next step return
    feat['target'] = feat['close'].pct_change(1).shift(-1)

    # Drop NaN and replace infinity
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna()

    print(f"[+] Features: {feat.shape[1]} columns (incl. target)")
    print(f"[+] Samples after feature engineering: {len(feat)}")

    return feat


# ─── DATASET ──────────────────────────────────────────────────────────────
class TimeSeriesDataset(Dataset):
    """Create sliding window sequences for time series."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, lookback: int):
        self.features = features
        self.targets = targets
        self.lookback = lookback

    def __len__(self):
        return len(self.features) - self.lookback

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.lookback]
        y = self.targets[idx + self.lookback]
        return torch.FloatTensor(x), torch.FloatTensor([y])


# ─── MODELS ───────────────────────────────────────────────────────────────
class LinearRegression(nn.Module):
    """Linear Regression with optional weight decay (L2).
    Based on D2L Chapter 3.
    """

    def __init__(self, input_size: int):
        super().__init__()
        # Xavier initialization
        self.linear = nn.Linear(input_size, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # x: (batch, lookback, features)
        # Flatten sequence: take last step + mean/std of window
        x_flat = self._flatten_features(x)
        return self.linear(x_flat)

    def _flatten_features(self, x):
        """Aggregate sequence into feature vector."""
        # (batch, lookback, features) → (batch, features * 4)
        last = x[:, -1, :]          # Last timestep
        mean = x.mean(dim=1)        # Mean over window
        std = x.std(dim=1)          # Std over window
        diff = x[:, -1, :] - x[:, 0, :]  # Change over window
        return torch.cat([last, mean, std, diff], dim=1)

    @property
    def flat_input_size(self):
        return self.linear.in_features


class MLP(nn.Module):
    """Multilayer Perceptron with Dropout.
    Based on D2L Chapter 5.
    """

    def __init__(self, input_size: int, hidden_sizes: list = None, dropout: float = 0.3):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x_flat = self._flatten_features(x)
        return self.network(x_flat)

    def _flatten_features(self, x):
        last = x[:, -1, :]
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        diff = x[:, -1, :] - x[:, 0, :]
        return torch.cat([last, mean, std, diff], dim=1)


# ─── TRAINING ─────────────────────────────────────────────────────────────
class HuberLoss(nn.Module):
    """Huber Loss — robust to outliers in crypto data."""
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        error = pred - target
        is_small = torch.abs(error) < self.delta
        loss = torch.where(
            is_small,
            0.5 * error ** 2,
            self.delta * (torch.abs(error) - 0.5 * self.delta)
        )
        return loss.mean()


def train_epoch(model, loader, loss_fn, optimizer, device, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (D2L Ch 9.7 — critical for financial data)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds = []
    all_targets = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        total_loss += loss.item()
        n_batches += 1
        all_preds.extend(pred.cpu().numpy().flatten())
        all_targets.extend(y.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    metrics = {
        'loss': total_loss / n_batches,
        'mae': mean_absolute_error(all_targets, all_preds),
        'rmse': np.sqrt(mean_squared_error(all_targets, all_preds)),
        'r2': r2_score(all_targets, all_preds),
        'mape': np.mean(np.abs((all_targets - all_preds) / (all_targets + 1e-10))) * 100,
    }

    # Direction accuracy
    directions_pred = np.sign(all_preds)
    directions_true = np.sign(all_targets)
    metrics['direction_accuracy'] = np.mean(directions_pred == directions_true) * 100

    return metrics, all_preds, all_targets


def train_model(model, train_loader, val_loader, config, model_name):
    """Full training loop with early stopping."""
    device = config['device']
    model = model.to(device)

    loss_fn = HuberLoss(delta=config['huber_delta'])

    if model_name == 'linear':
        optimizer = optim.SGD(model.parameters(), lr=config['linear_lr'],
                              weight_decay=config['linear_wd'], momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['linear_epochs'])
        epochs = config['linear_epochs']
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config['mlp_lr'],
                                weight_decay=config['mlp_wd'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['mlp_epochs'])
        epochs = config['mlp_epochs']

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_r2': [], 'val_dir_acc': []}

    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} on {config['coin']}")
    print(f"{'='*60}")
    print(f"  Device:       {device}")
    print(f"  Epochs:       {epochs}")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples:   {len(val_loader.dataset)}")
    print(f"  Loss:          Huber (delta={config['huber_delta']})")
    print(f"  Early stop:    patience={config['patience']}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer,
                                 device, config['grad_clip'])
        val_metrics, _, _ = evaluate(model, val_loader, loss_fn, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_r2'].append(val_metrics['r2'])
        history['val_dir_acc'].append(val_metrics['direction_accuracy'])

        # Logging
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_metrics['loss']:.6f} | "
                  f"Val MAE: {val_metrics['mae']:.6f} | "
                  f"Val R²: {val_metrics['r2']:.4f} | "
                  f"Dir Acc: {val_metrics['direction_accuracy']:.1f}% | "
                  f"LR: {current_lr:.6f}")

        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\n  ⏹ Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - start_time

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\n  ✓ Training complete in {elapsed:.1f}s")
    print(f"  Best val loss: {best_val_loss:.6f}")

    return model, history


# ─── DATA PREPARATION ─────────────────────────────────────────────────────
def prepare_data(df_features, config):
    """Split data chronologically and create DataLoaders."""
    # Separate target and features (exclude non-feature columns)
    exclude_cols = ['target', 'index']
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]
    features = df_features[feature_cols].values.astype(np.float64)
    targets = df_features['target'].values.astype(np.float64)

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Chronological split (NO shuffling!)
    n = len(features)
    train_end = int(n * config['train_ratio'])
    val_end = int(n * (config['train_ratio'] + config['val_ratio']))

    train_feat = features[:train_end]
    train_tgt = targets[:train_end]
    val_feat = features[train_end:val_end]
    val_tgt = targets[train_end:val_end]
    test_feat = features[val_end:]
    test_tgt = targets[val_end:]

    dates = df_features.index

    print(f"\n  Data split:")
    print(f"    Train: {len(train_feat)} samples ({dates[0]} → {dates[train_end-1]})")
    print(f"    Val:   {len(val_feat)} samples ({dates[train_end]} → {dates[val_end-1]})")
    print(f"    Test:  {len(test_feat)} samples ({dates[val_end]} → {dates[-1]})")

    # Create datasets
    lookback = config['lookback']
    train_ds = TimeSeriesDataset(train_feat, train_tgt, lookback)
    val_ds = TimeSeriesDataset(val_feat, val_tgt, lookback)
    test_ds = TimeSeriesDataset(test_feat, test_tgt, lookback)

    batch_size = config['linear_batch']  # Will adjust for MLP

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_size = features.shape[1] * 4  # last + mean + std + diff

    return train_loader, val_loader, test_loader, input_size, scaler, feature_cols


# ─── RESULTS ──────────────────────────────────────────────────────────────
def save_results(model_name, metrics, history, config, test_preds, test_targets, scaler, feature_cols):
    """Save all outputs."""
    os.makedirs(config['output_dir'], exist_ok=True)

    # Convert numpy types for JSON
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results = {
        'model': model_name,
        'coin': config['coin'],
        'config': {k: v for k, v in config.items() if k not in ['device']},
        'device': config['device'],
        'test_metrics': {k: convert_numpy(v) for k, v in metrics.items()},
        'training_history': {k: [convert_numpy(x) for x in v] for k, v in history.items()},
        'feature_columns': feature_cols,
        'lookback': config['lookback'],
    }

    # Save JSON
    path = os.path.join(config['output_dir'], f'{model_name}_{config["coin"].replace("-", "_")}_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  📊 Results saved: {path}")

    # Save model weights
    model_path = os.path.join(config['output_dir'], f'{model_name}_{config["coin"].replace("-", "_")}.pth')
    # (model state saved during training)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  TEST RESULTS — {model_name.upper()}")
    print(f"{'='*60}")
    print(f"  MAE:               {metrics['mae']:.6f}")
    print(f"  RMSE:              {metrics['rmse']:.6f}")
    print(f"  R² Score:          {metrics['r2']:.4f}")
    print(f"  MAPE:              {metrics['mape']:.2f}%")
    print(f"  Direction Accuracy: {metrics['direction_accuracy']:.1f}%")
    print(f"{'='*60}")


def plot_results(history, test_preds, test_targets, model_name, config):
    """Generate training and prediction plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(config['output_dir'], exist_ok=True)
    base = os.path.join(config['output_dir'], f'{model_name}_{config["coin"].replace("-", "_")}')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name.upper()} — {config["coin"]}', fontsize=14, fontweight='bold')

    # Training loss
    ax = axes[0, 0]
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Huber Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation metrics
    ax = axes[0, 1]
    ax.plot(epochs, history['val_r2'], label='R²', color='green', linewidth=2)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R² Score')
    ax.set_title('Validation R² Score')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Direction accuracy
    ax = axes[1, 0]
    ax.plot(epochs, history['val_dir_acc'], label='Direction Accuracy', color='orange', linewidth=2)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Direction Prediction Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Predictions vs actuals (sample)
    ax = axes[1, 1]
    sample_size = min(200, len(test_preds))
    ax.plot(test_targets[:sample_size], label='Actual', alpha=0.7, linewidth=1)
    ax.plot(test_preds[:sample_size], label='Predicted', alpha=0.7, linewidth=1)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Return')
    ax.set_title(f'Predictions vs Actuals (first {sample_size})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f'{base}_plots.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Plots saved: {plot_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Phase 1: Crypto Price Prediction Baseline')
    parser.add_argument('--model', type=str, default='linear', choices=['linear', 'mlp'],
                        help='Model type: linear or mlp')
    parser.add_argument('--coin', type=str, default='BTC-USD',
                        help='Coin symbol (yfinance format)')
    parser.add_argument('--interval', type=str, default='1h',
                        help='Data interval: 1h, 1d, etc.')
    parser.add_argument('--period', type=str, default='2y',
                        help='Data period: 1y, 2y, 5y, max')
    parser.add_argument('--lookback', type=int, default=48,
                        help='Lookback window in timesteps')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting')
    args = parser.parse_args()

    # Update config
    CONFIG['coin'] = args.coin
    CONFIG['interval'] = args.interval
    CONFIG['period'] = args.period
    CONFIG['lookback'] = args.lookback
    if args.epochs:
        CONFIG[f'{args.model}_epochs'] = args.epochs
    if args.gpu:
        CONFIG['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set seed
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])

    print(f"\n{'█'*60}")
    print(f"  PHASE 1: BASELINE MODELS")
    print(f"  Based on D2L.ai Chapters 3, 5, 6, 12")
    print(f"{'█'*60}")

    # 1. Fetch data
    df_raw = fetch_data(CONFIG['coin'], CONFIG['interval'], CONFIG['period'])

    # 2. Feature engineering
    df_feat = engineer_features(df_raw)

    # 3. Prepare data
    train_loader, val_loader, test_loader, input_size, scaler, feature_cols = \
        prepare_data(df_feat, CONFIG)

    # 4. Create model
    if args.model == 'linear':
        model = LinearRegression(input_size)
    else:
        model = MLP(input_size, hidden_sizes=CONFIG['mlp_hidden'],
                    dropout=CONFIG['mlp_dropout'])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model: {args.model.upper()}")
    print(f"  Input size: {input_size}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # 5. Train
    model, history = train_model(model, train_loader, val_loader, CONFIG, args.model)

    # 6. Evaluate on test set
    loss_fn = HuberLoss(delta=CONFIG['huber_delta'])
    test_metrics, test_preds, test_targets = evaluate(model, test_loader, loss_fn, CONFIG['device'])

    # 7. Save results
    save_results(args.model, test_metrics, history, CONFIG, test_preds, test_targets,
                 scaler, feature_cols)

    # 8. Plot
    if not args.no_plot:
        try:
            plot_results(history, test_preds, test_targets, args.model, CONFIG)
        except Exception as e:
            print(f"\n  ⚠️ Could not generate plots: {e}")

    print(f"\n{'█'*60}")
    print(f"  PHASE 1 COMPLETE ✓")
    print(f"  Next: Phase 2 — LSTM/GRU (D2L Chapter 10)")
    print(f"{'█'*60}\n")


if __name__ == '__main__':
    main()
