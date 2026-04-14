# Phase 1: Baseline Models — Crypto Price Prediction

Based on **D2L.ai Chapters 3, 5, 6, 12**

## Models

| Model | D2L Chapter | Description |
|---|---|---|
| **Linear Regression** | Ch 3 | Simplest baseline with weight decay (L2) |
| **MLP** | Ch 5 | Nonlinear baseline with dropout + ReLU |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Linear Regression (BTC, 1h candles)
python train.py --model linear

# MLP (more capacity)
python train.py --model mlp

# Ethereum
python train.py --model linear --coin ETH-USD

# More epochs + GPU
python train.py --model mlp --epochs 150 --gpu

# Daily candles (less noise)
python train.py --model linear --interval 1d --period 5y

# Skip plots (headless server)
python train.py --model linear --no-plot
```

## Features Engineered

| Category | Features |
|---|---|
| **Price** | Close, Open, High, Low |
| **Returns** | 1h, 3h, 6h, 12h, 24h returns + log return |
| **Volume** | Volume, volume change % |
| **Momentum** | RSI(14) |
| **Trend** | MACD, MACD Signal, MACD Histogram |
| **Volatility** | Bollinger Width, Bollinger Position |
| **ATR** | 14-period Average True Range |

## Architecture

### Linear Regression
```
Input: (batch, lookback, features)
  ↓ Flatten (last + mean + std + diff)
Linear(4*features, 1)
  ↓
Output: predicted return
```

### MLP
```
Input: (batch, lookback, features)
  ↓ Flatten
Linear → ReLU → Dropout(0.3)
  ↓
Linear → ReLU → Dropout(0.2)
  ↓
Linear(1)
  ↓
Output: predicted return
```

## Training Details

| Parameter | Linear | MLP |
|---|---|---|
| Loss | Huber (δ=1.0) | Huber (δ=1.0) |
| Optimizer | SGD + Momentum | AdamW |
| LR | 1e-2 | 1e-3 |
| Weight Decay | 1e-3 | 1e-4 |
| Epochs | 50 | 100 |
| Batch Size | 256 | 128 |
| Early Stop | Patience 15 | Patience 15 |
| Gradient Clip | 1.0 | 1.0 |
| LR Schedule | Cosine Annealing | Cosine Annealing |

## Output

All results saved to `outputs/` directory:
- `*_results.json` — Full metrics + training history
- `*_plots.png` — Training curves + prediction visualization
- `*.pth` — Best model weights

## Metrics

| Metric | Meaning |
|---|---|
| **MAE** | Mean absolute error (lower = better) |
| **RMSE** | Root mean squared error (penalizes large errors) |
| **R²** | Variance explained (closer to 1 = better, <0 = worse than mean) |
| **MAPE** | Mean absolute percentage error |
| **Direction Accuracy** | % of correct up/down predictions (>50% = skill) |

## Next Steps

After establishing baseline → Phase 2: LSTM/GRU (Chapter 10)
