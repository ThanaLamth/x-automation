#!/usr/bin/env python3
"""
Phase 2: Sequence Models (LSTM / GRU) for Crypto Price Prediction
==================================================================
Based on D2L.ai Chapter 10: Modern Recurrent Neural Networks

Models:
  1. LSTM (Long Short-Term Memory)
  2. GRU (Gated Recurrent Unit)
  3. Stacked LSTM (2-3 layers)
  4. Seq2Seq Encoder-Decoder (multi-step forecasting)

Key D2L concepts applied:
  - LSTM gates: forget, input, cell, output (Section 10.1)
  - GRU gates: reset, update (Section 10.2)
  - Deep RNNs with hierarchical temporal abstraction (10.3)
  - Backpropagation Through Time + Gradient Clipping (9.7)
  - Encoder-Decoder for multi-step forecasting (10.6-10.7)
  - State detachment between batches (9.5)

Usage:
  python train_phase2.py --model lstm
  python train_phase2.py --model gru
  python train_phase2.py --model stacked-lstm --layers 3
  python train_phase2.py --model seq2seq --forecast-horizon 24
  python train_phase2.py --model lstm --coin ETH-USD --epochs 100
  python train_phase2.py --model lstm --gpu
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
    'period': '2y',
    'lookback': 96,             # 96 hours = 4 days
    'forecast_horizon': 1,      # Predict next 1 step (set >1 for multi-step)
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    # Common
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'epochs': 80,
    'batch_size': 64,
    'huber_delta': 1.0,
    'patience': 20,
    'grad_clip': 1.0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'output_dir': 'outputs',
}


# ─── DATA FETCHING & FEATURE ENGINEERING (same as Phase 1) ────────────────
def fetch_data(coin, interval, period):
    import yfinance as yf
    print(f"[>] Fetching {coin} ({interval}, {period})...")
    ticker = yf.Ticker(coin)
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        print(f"[-] No data for {coin}")
        sys.exit(1)
    print(f"[+] Got {len(df)} candles: {df.index[0]} → {df.index[-1]}")
    return df


def engineer_features(df):
    feat = pd.DataFrame(index=df.index)
    feat['close'] = df['Close']
    feat['open'] = df['Open']
    feat['high'] = df['High']
    feat['low'] = df['Low']
    feat['volume'] = df['Volume']
    feat['return_1'] = feat['close'].pct_change(1)
    feat['return_3'] = feat['close'].pct_change(3)
    feat['return_6'] = feat['close'].pct_change(6)
    feat['return_12'] = feat['close'].pct_change(12)
    feat['return_24'] = feat['close'].pct_change(24)
    feat['log_return'] = np.log(feat['close'] / feat['close'].shift(1))
    feat['volume_change'] = feat['volume'].pct_change(1)

    # RSI
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

    # Bollinger
    sma20 = feat['close'].rolling(20).mean()
    std20 = feat['close'].rolling(20).std()
    feat['bb_upper'] = sma20 + 2 * std20
    feat['bb_lower'] = sma20 - 2 * std20
    feat['bb_width'] = (feat['bb_upper'] - feat['bb_lower']) / (sma20 + 1e-10)
    feat['bb_position'] = (feat['close'] - feat['bb_lower']) / (feat['bb_upper'] - feat['bb_lower'] + 1e-10)

    # ATR
    tr = pd.concat([
        feat['high'] - feat['low'],
        (feat['high'] - feat['close'].shift(1)).abs(),
        (feat['low'] - feat['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    feat['atr'] = tr.rolling(14).mean()

    # Target
    feat['target'] = feat['close'].pct_change(1).shift(-1)

    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"[+] Features: {feat.shape[1]} columns, {len(feat)} samples")
    return feat


# ─── DATASET ──────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    """Sliding window for sequence models."""
    def __init__(self, features, targets, lookback, horizon=1):
        self.features = features
        self.targets = targets
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self):
        return len(self.features) - self.lookback - self.horizon + 1

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.lookback]
        if self.horizon == 1:
            y = self.targets[idx + self.lookback]
        else:
            y = self.targets[idx + self.lookback:idx + self.lookback + self.horizon]
        return torch.FloatTensor(x), torch.FloatTensor([y] if self.horizon == 1 else y)


def prepare_data(df_feat, config):
    feature_cols = [c for c in df_feat.columns if c != 'target']
    features = df_feat[feature_cols].values.astype(np.float64)
    targets = df_feat['target'].values.astype(np.float64)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    n = len(features)
    train_end = int(n * config['train_ratio'])
    val_end = int(n * (config['train_ratio'] + config['val_ratio']))

    train_ds = SequenceDataset(
        features[:train_end], targets[:train_end],
        config['lookback'], config['forecast_horizon']
    )
    val_ds = SequenceDataset(
        features[train_end:val_end], targets[train_end:val_end],
        config['lookback'], config['forecast_horizon']
    )
    test_ds = SequenceDataset(
        features[val_end:], targets[val_end:],
        config['lookback'], config['forecast_horizon']
    )

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)

    input_size = features.shape[1]
    dates = df_feat.index

    print(f"\n  Data split:")
    print(f"    Train: {len(train_ds)} sequences ({dates[0]} → {dates[train_end-1]})")
    print(f"    Val:   {len(val_ds)} sequences ({dates[train_end]} → {dates[val_end-1]})")
    print(f"    Test:  {len(test_ds)} sequences ({dates[val_end]} → {dates[-1]})")
    print(f"    Input size: {input_size}, Lookback: {config['lookback']}, Horizon: {config['forecast_horizon']}")

    return train_loader, val_loader, test_loader, input_size, scaler, feature_cols


# ─── MODELS ───────────────────────────────────────────────────────────────

class CryptoLSTM(nn.Module):
    """LSTM for time series forecasting.
    Based on D2L Section 10.1 — LSTM Architecture.

    LSTM Gates:
      - Forget Gate: decides what to discard from cell state
      - Input Gate: decides what new info to store
      - Cell State: long-term memory highway
      - Output Gate: decides what to output

    For crypto:
      - Forget gate filters flash crash noise
      - Cell state preserves bull/bear cycles
      - Output gate adjusts prediction confidence
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x, state=None):
        # x: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x, state)
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


class CryptoGRU(nn.Module):
    """GRU for time series forecasting.
    Based on D2L Section 10.2 — Gated Recurrent Units.

    GRU vs LSTM:
      - Fewer parameters → faster training
      - Reset gate adapts to regime shifts quickly
      - Update gate balances past vs new information
      - Better for high-frequency data (1m/5m/15m)

    For crypto:
      - Reset gate handles sudden liquidity drops
      - Update gate smooths volatile micro-trends
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x, state=None):
        gru_out, h_n = self.gru(x, state)
        last_hidden = gru_out[:, -1, :]
        return self.fc(last_hidden)


class StackedLSTM(nn.Module):
    """Deep LSTM with hierarchical temporal abstraction.
    Based on D2L Section 10.3 — Deep Recurrent Neural Networks.

    Hierarchical learning:
      Layer 1: Local dependencies (tick noise, volume spikes)
      Layer 2: Medium-term (daily momentum, support/resistance)
      Layer 3: Global (weekly cycles, macro trend)
    """
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        # Attention over hidden states
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x, state=None):
        lstm_out, (h_n, c_n) = self.lstm(x, state)
        # Attention-weighted sum of all hidden states
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)
        return self.fc(context)


class Seq2Seq(nn.Module):
    """Encoder-Decoder for multi-step forecasting.
    Based on D2L Section 10.6-10.7 — Encoder-Decoder & Seq2Seq.

    Encoder: compresses historical window into context vector
    Decoder: generates future sequence autoregressively

    Teacher forcing during training:
      Decoder receives actual targets shifted by 1 step
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 forecast_horizon=24, dropout=0.2):
        super().__init__()
        self.forecast_horizon = forecast_horizon

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.decoder = nn.LSTM(
            input_size=1,  # Previous prediction as input
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, enc_x, dec_x=None, teacher_forcing_ratio=0.5):
        # Encode
        enc_out, (h_n, c_n) = self.encoder(enc_x)

        if dec_x is not None:
            # Training with teacher forcing
            dec_out, _ = self.decoder(dec_x, (h_n, c_n))
            return self.fc(dec_out)
        else:
            # Inference — autoregressive
            batch_size = enc_x.size(0)
            device = enc_x.device
            predictions = []

            # Initial input: last known target (zeros for now)
            dec_input = torch.zeros(batch_size, 1, 1, device=device)
            state = (h_n, c_n)

            for _ in range(self.forecast_horizon):
                dec_out, state = self.decoder(dec_input, state)
                pred = self.fc(dec_out[:, -1, :])  # (batch, 1)
                predictions.append(pred)
                dec_input = pred.unsqueeze(1)  # Feed back as next input

            return torch.stack(predictions, dim=1).squeeze(-1)  # (batch, horizon)


# ─── TRAINING ─────────────────────────────────────────────────────────────
class HuberLoss(nn.Module):
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


def train_epoch(model, loader, loss_fn, optimizer, device, grad_clip=1.0,
                model_type='single', teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if model_type == 'seq2seq':
            # For seq2seq, create decoder input from target
            dec_input = torch.cat([y[:, :1].unsqueeze(-1), y[:, :-1].unsqueeze(-1)], dim=1)
            pred = model(x, dec_input=dec_input, teacher_forcing_ratio=teacher_forcing_ratio)
            loss = loss_fn(pred, y)
        else:
            pred = model(x)
            loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping — critical for RNNs (D2L Section 9.7)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, model_type='single'):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds = []
    all_targets = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if model_type == 'seq2seq':
            pred = model(x, dec_input=None)  # Autoregressive inference
        else:
            pred = model(x)

        # Handle different output shapes
        if pred.dim() > y.dim():
            pred = pred.squeeze(-1)
        if pred.shape != y.shape:
            # Take first step for single-step comparison
            if pred.dim() == 2 and y.dim() == 1:
                pred = pred[:, 0]

        loss = loss_fn(pred, y)
        total_loss += loss.item()
        n_batches += 1

        pred_np = pred.cpu().numpy().flatten()
        target_np = y.cpu().numpy().flatten()

        # Ensure same length
        min_len = min(len(pred_np), len(target_np))
        all_preds.extend(pred_np[:min_len])
        all_targets.extend(target_np[:min_len])

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    metrics = {
        'loss': total_loss / max(n_batches, 1),
        'mae': mean_absolute_error(all_targets, all_preds),
        'rmse': np.sqrt(mean_squared_error(all_targets, all_preds)),
        'r2': r2_score(all_targets, all_preds),
        'mape': np.mean(np.abs((all_targets - all_preds) / (np.abs(all_targets) + 1e-10))) * 100,
    }

    directions_pred = np.sign(all_preds)
    directions_true = np.sign(all_targets)
    metrics['direction_accuracy'] = np.mean(directions_pred == directions_true) * 100

    return metrics, all_preds, all_targets


def train_model(model, train_loader, val_loader, config, model_name, model_type='single'):
    device = config['device']
    model = model.to(device)
    loss_fn = HuberLoss(delta=config['huber_delta'])

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_r2': [], 'val_dir_acc': []}

    teacher_forcing_ratio = 0.5

    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} on {config['coin']}")
    print(f"{'='*60}")
    print(f"  Model Type:     {model_type}")
    print(f"  Device:         {device}")
    print(f"  Hidden Size:    {config['hidden_size']}")
    print(f"  Layers:         {config['num_layers']}")
    print(f"  Dropout:        {config['dropout']}")
    print(f"  Epochs:         {config['epochs']}")
    print(f"  Batch Size:     {config['batch_size']}")
    print(f"  Train samples:  {len(train_loader.dataset)}")
    print(f"  Val samples:    {len(val_loader.dataset)}")
    print(f"  Loss:           Huber (delta={config['huber_delta']})")
    print(f"  Gradient Clip:  {config['grad_clip']}")
    print(f"  Early Stop:     patience={config['patience']}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(1, config['epochs'] + 1):
        # Decay teacher forcing for seq2seq
        if model_type == 'seq2seq':
            teacher_forcing_ratio = max(0.0, 0.5 - epoch * 0.005)

        train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, device,
            config['grad_clip'], model_type, teacher_forcing_ratio
        )
        val_metrics, _, _ = evaluate(model, val_loader, loss_fn, device, model_type)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_r2'].append(val_metrics['r2'])
        history['val_dir_acc'].append(val_metrics['direction_accuracy'])

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{config['epochs']} | "
                  f"Train: {train_loss:.6f} | "
                  f"Val Loss: {val_metrics['loss']:.6f} | "
                  f"Val MAE: {val_metrics['mae']:.6f} | "
                  f"Val R²: {val_metrics['r2']:.4f} | "
                  f"Dir Acc: {val_metrics['direction_accuracy']:.1f}% | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

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
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\n  ✓ Training complete in {elapsed:.1f}s")
    print(f"  Best val loss: {best_val_loss:.6f}")

    return model, history


# ─── RESULTS ──────────────────────────────────────────────────────────────
def save_results(model_name, metrics, history, config):
    os.makedirs(config['output_dir'], exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        elif isinstance(obj, (np.floating,)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    results = {
        'model': model_name,
        'coin': config['coin'],
        'config': {k: v for k, v in config.items() if k not in ['device']},
        'device': config['device'],
        'test_metrics': {k: convert(v) for k, v in metrics.items()},
        'training_history': {k: [convert(x) for x in v] for k, v in history.items()},
        'lookback': config['lookback'],
        'forecast_horizon': config['forecast_horizon'],
    }

    path = os.path.join(config['output_dir'], f'{model_name}_{config["coin"].replace("-", "_")}_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  📊 Results saved: {path}")

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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(config['output_dir'], exist_ok=True)
    base = os.path.join(config['output_dir'], f'{model_name}_{config["coin"].replace("-", "_")}')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name.upper()} — {config["coin"]}', fontsize=14, fontweight='bold')

    epochs = range(1, len(history['train_loss']) + 1)

    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Huber Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history['val_r2'], label='R²', color='green', linewidth=2)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('R² Score')
    ax.set_title('Validation R² Score')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history['val_dir_acc'], label='Direction Accuracy', color='orange', linewidth=2)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Direction Prediction Accuracy')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    sample_size = min(200, len(test_preds))
    ax.plot(test_targets[:sample_size], label='Actual', alpha=0.7, linewidth=1)
    ax.plot(test_preds[:sample_size], label='Predicted', alpha=0.7, linewidth=1)
    ax.set_xlabel('Sample'); ax.set_ylabel('Return')
    ax.set_title(f'Predictions vs Actuals (first {sample_size})')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{base}_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Plots saved: {base}_plots.png")


# ─── MAIN ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Phase 2: LSTM/GRU for Crypto Prediction')
    parser.add_argument('--model', type=str, default='lstm',
                        choices=['lstm', 'gru', 'stacked-lstm', 'seq2seq'],
                        help='Model type')
    parser.add_argument('--coin', type=str, default='BTC-USD')
    parser.add_argument('--interval', type=str, default='1h')
    parser.add_argument('--period', type=str, default='2y')
    parser.add_argument('--lookback', type=int, default=96)
    parser.add_argument('--horizon', type=int, default=1, help='Forecast horizon (multi-step)')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    CONFIG['coin'] = args.coin
    CONFIG['interval'] = args.interval
    CONFIG['period'] = args.period
    CONFIG['lookback'] = args.lookback
    CONFIG['forecast_horizon'] = args.horizon
    CONFIG['hidden_size'] = args.hidden
    CONFIG['num_layers'] = args.layers
    CONFIG['dropout'] = args.dropout
    CONFIG['batch_size'] = args.batch
    if args.epochs:
        CONFIG['epochs'] = args.epochs
    if args.gpu:
        CONFIG['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])

    print(f"\n{'█'*60}")
    print(f"  PHASE 2: SEQUENCE MODELS (LSTM / GRU)")
    print(f"  Based on D2L.ai Chapter 10")
    print(f"{'█'*60}")

    # 1. Data
    df_raw = fetch_data(CONFIG['coin'], CONFIG['interval'], CONFIG['period'])
    df_feat = engineer_features(df_raw)
    train_loader, val_loader, test_loader, input_size, scaler, feature_cols = \
        prepare_data(df_feat, CONFIG)

    # 2. Model type mapping
    model_map = {
        'lstm': ('LSTM', 'single'),
        'gru': ('GRU', 'single'),
        'stacked-lstm': ('StackedLSTM', 'single'),
        'seq2seq': ('Seq2Seq', 'seq2seq'),
    }
    model_key, model_type = model_map[args.model]

    # 3. Create model
    if args.model == 'lstm':
        model = CryptoLSTM(input_size, CONFIG['hidden_size'], CONFIG['num_layers'], CONFIG['dropout'])
    elif args.model == 'gru':
        model = CryptoGRU(input_size, CONFIG['hidden_size'], CONFIG['num_layers'], CONFIG['dropout'])
    elif args.model == 'stacked-lstm':
        model = StackedLSTM(input_size, CONFIG['hidden_size'], CONFIG['num_layers'], CONFIG['dropout'])
    elif args.model == 'seq2seq':
        model = Seq2Seq(input_size, CONFIG['hidden_size'], CONFIG['num_layers'],
                       CONFIG['forecast_horizon'], CONFIG['dropout'])

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model: {model_key}")
    print(f"  Input size: {input_size}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable: {trainable:,}")

    # 4. Train
    model, history = train_model(model, train_loader, val_loader, CONFIG, args.model, model_type)

    # 5. Evaluate
    loss_fn = HuberLoss(delta=CONFIG['huber_delta'])
    test_metrics, test_preds, test_targets = evaluate(model, test_loader, loss_fn, CONFIG['device'], model_type)

    # 6. Save
    save_results(args.model, test_metrics, history, CONFIG)

    # 7. Plot
    if not args.no_plot:
        try:
            plot_results(history, test_preds, test_targets, args.model, CONFIG)
        except Exception as e:
            print(f"\n  ⚠️ Could not generate plots: {e}")

    print(f"\n{'█'*60}")
    print(f"  PHASE 2 COMPLETE ✓")
    print(f"  Next: Phase 3 — Transformer (D2L Chapter 11)")
    print(f"{'█'*60}\n")


if __name__ == '__main__':
    main()
