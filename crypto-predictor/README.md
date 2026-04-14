# 🔮 Crypto Price Prediction — Deep Research & Multi-Phase Experiment

> Comprehensive research into whether deep learning can predict Bitcoin price direction.
> Based on **D2L.ai (Dive into Deep Learning)** + real market data from 5+ free sources.
>
> **Spoiler:** After 4 phases, 10+ models, 40+ features, 1,825 days of data — the answer is **no**.
> But the journey taught us a lot. 📊

---

## 📋 Quick Navigation

| Directory | Phase | Models | Key Finding |
|---|---|---|---|
| [`phase1_baseline/`](crypto-predictor/phase1_baseline/) | Baseline | Linear, MLP | Linear direction: **54.6%** (best overall) |
| [`phase2_sequence/`](crypto-predictor/phase2_sequence/) | Sequence | LSTM, GRU | LSTM best at magnitude (R²≈0), not direction |
| [`phase2.5_daily_classify/`](crypto-predictor/phase2.5_daily_classify/) | Classification | Linear, LSTM, Ensemble | Daily candles don't beat hourly |
| [`phase3_alternative_data/`](crypto-predictor/phase3_alternative_data/) | Multi-Source | LSTM | Adding FGI + macro doesn't help direction |
| [`phase4_complete_free/`](crypto-predictor/phase4_complete_free/) | 5 Pillars | LSTM-Attention | 5 free sources, 40 features → still ~50% |

---

## 🎯 Research Question

> **Can deep learning models predict the direction (up/down) of Bitcoin daily returns** using:
> 1. OHLCV technical features
> 2. Sequence modeling (LSTM/GRU/Transformer)
> 3. Alternative data (on-chain, sentiment, macro, order book, options)
>
> **Target benchmark:** >55% direction accuracy (statistically significant edge)

---

## 📊 Final Results Summary

| # | Model | Data | Features | Samples | Direction Accuracy | Key Finding |
|---|---|---|---|---|---|---|
| 1 | **Linear** | 1h OHLCV | 21 | 4,163 | **54.6%** ⭐ | Best direction accuracy |
| 2 | MLP | 1h OHLCV | 21 | 4,163 | 49.2% | Overfits, low direction |
| 3 | LSTM | 1h OHLCV | 21 | 4,163 | 49.1% | Best magnitude (R²=-0.02) |
| 4 | Linear | Daily OHLCV | 35 | 1,778 | 52.3% | Less noise, less signal too |
| 5 | Ensemble | Daily OHLCV | 35 | 1,778 | 50.2% | No ensemble benefit |
| 6 | LSTM | Daily + FGI + Macro | 27 | 971 | 50.0% | Alt data doesn't help |
| 7 | **LSTM-Attn** | **5 Pillars (all free)** | **40** | **1,825** | **49.2%** | Most features → still random |

### 📈 Visual: Direction Accuracy Across All Phases

```
55% ┤  ⭐ Linear (54.6%)
    │
54% ┤
    │
53% ┤  Linear-Daily (52.3%)
    │
52% ┤  Ensemble (50.2%)
    │
51% ┤  LSTM-Alt (50.0%)
    │
50% ┤  ─── Random baseline ───
    │  MLP (49.2%)  LSTM (49.1%)  LSTM-Attn-5Pillar (49.2%)
49% ┤
    │
48% ┤
    │
47% ┤
    └────────────────────────────────────────
       Phase1    Phase2    Phase2.5   Phase3    Phase4
```

---

## 🔬 Methodology

### Data Pipeline

```
Raw Data → Feature Engineering → StandardScaler → Sliding Window → Train/Val/Test
                                                                      (70/15/15)
```

**Key design choices:**
- ⚠️ **Time-based split** (no random shuffle — prevents data leakage)
- 📏 **Huber loss** (robust to crypto outliers / flash crashes)
- 🎯 **Gradient clipping** (prevents explosion from volatile data)
- 🛑 **Early stopping** (patience 15-25 epochs on validation loss)

### Feature Engineering

| Category | Features | Source |
|---|---|---|
| **Price** | OHLC, returns (1h/3h/6h/12h/24h), log returns | yfinance / Binance |
| **Volume** | Volume, volume change, volume trend (7d/30d MA) | yfinance / Binance |
| **Momentum** | RSI(14), MACD, MACD histogram, MACD signal | Derived |
| **Volatility** | Bollinger width, position, ATR(14) | Derived |
| **Order Book** | Spread %, taker buy ratio, trade intensity | Binance API |
| **Options** | IV 25d/60d/90d, term structure, slope, regime | Deribit / Binance vol proxy |
| **On-Chain** | Hash rate, n-transactions, n-addresses, difficulty | Blockchain.com API |
| **Sentiment** | Fear & Greed Index, sentiment MA, extreme flags | alternative.me |
| **Macro** | DXY, 10Y yield, gold, S&P 500, risk-on/off proxy | yfinance |

**Total: 40 features across 5 pillars**

### Model Architectures

#### Linear (85 params)
```python
Linear(4×21, 1)  # last + mean + std + diff → single prediction
```

#### MLP (19,201 params)
```python
Linear → ReLU → Dropout(0.3) → Linear → ReLU → Dropout(0.2) → Linear(1)
```

#### LSTM (57,665 params)
```python
LSTM(21, 64, 2 layers) → Linear(64, 32) → ReLU → Dropout → Linear(32, 1)
```

#### LSTM-Attention (63,586 params)
```python
LSTM → Self-Attention over hidden states → Weighted sum → FC → Output
```

#### Multi-Source LSTM
```python
5 separate LSTM encoders (one per pillar) → Concatenate → Cross-attention → FC → Output
```

---

## 🏗️ Phase Details

### Phase 1: Baseline Models
- **Goal:** Establish performance floor
- **Models:** Linear Regression, MLP
- **Finding:** Linear with 85 params beat MLP with 19K params on direction (54.6% vs 49.2%)
- **Lesson:** Simple models generalize better on noisy financial data

### Phase 2: Sequence Models
- **Goal:** Capture temporal dependencies
- **Models:** LSTM, GRU
- **Finding:** LSTM achieves R²≈-0.02 (near perfect magnitude), but direction still ~49%
- **Lesson:** Knowing "how much" ≠ knowing "which way"

### Phase 2.5: Daily Classification
- **Goal:** Reduce noise + simpler task
- **Models:** Linear, LSTM, Ensemble
- **Finding:** Daily candles don't improve direction over hourly
- **Lesson:** Less noise = less signal too

### Phase 3: Alternative Data
- **Goal:** Add non-OHLCV features
- **Models:** LSTM with FGI + on-chain + macro
- **Finding:** 27 features still can't beat 50%
- **Lesson:** Public alt data is already priced in

### Phase 4: Complete Free Multi-Source
- **Goal:** 5 pillars, all free sources
- **Sources:** Binance, Deribit, Blockchain.com, Fear&Greed, yfinance macro
- **Models:** LSTM-Attention, Multi-Source LSTM
- **Finding:** 40 features, 1825 days → still ~50%
- **Lesson:** **No free data source provides predictive edge**

---

## 📁 File Structure

```
crypto-predictor/
├── phase1_baseline/
│   ├── train.py                    # Linear + MLP training script
│   ├── README.md                   # Phase 1 documentation
│   ├── requirements.txt            # Python dependencies
│   └── outputs/
│       ├── linear_BTC_USD_results.json
│       └── mlp_BTC_USD_results.json
│
├── phase2_sequence/
│   ├── train_phase2.py             # LSTM + GRU + Stacked + Seq2Seq
│   └── outputs/
│       └── lstm_BTC_USD_results.json
│
├── phase2.5_daily_classify/
│   ├── train_daily_classify.py     # Daily classification pipeline
│   └── outputs/
│       ├── linear_BTC_USD_daily.json
│       ├── lstm_BTC_USD_daily.json
│       └── ensemble_BTC_USD_daily.json
│
├── phase3_alternative_data/
│   ├── train_phase3.py             # Multi-source with Fear&Greed + macro
│   └── outputs/
│       └── lstm_BTC_USD_alt.json
│
├── phase4_complete_free/
│   ├── train_phase4.py             # 5-pillar complete system
│   └── outputs/
│       └── lstm-attn_5pillar.json
│
├── RAVE-deep-research.md           # CoinDesk-style RAVE analysis
├── RAVE-deep-research.html         # HTML version with SVG charts
├── RAVE-deep-research.pdf          # PDF version
├── d2l-crypto-model-research.md    # Full D2L research document
├── cmc-skill/                      # CMC browser scraper skill
│   ├── SKILL.md
│   ├── cmc-skill.sh
│   └── cmc-scrape.js
├── cmc-api-skill/                  # CMC CLI wrapper
├── Auto_X_Login_Scroll_Post.genlogin  # GenLogin automation
└── README.md                       # ← You are here
```

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install torch numpy pandas scikit-learn yfinance matplotlib seaborn

# Phase 1: Baseline
cd phase1_baseline
python train.py --model linear    # Linear regression
python train.py --model mlp       # MLP

# Phase 2: Sequence models
cd ../phase2_sequence
python train_phase2.py --model lstm
python train_phase2.py --model gru

# Phase 2.5: Daily classification
cd ../phase2.5_daily_classify
python train_daily_classify.py --model linear

# Phase 3: Alternative data
cd ../phase3_alternative_data
python train_phase3.py --model lstm

# Phase 4: Complete 5-pillar system
cd ../phase4_complete_free
python train_phase4.py --model lstm-attn
```

---

## 🔑 Key Insights

### What We Learned

1. **Direction prediction is fundamentally hard** — BTC daily returns are nearly random walk
2. **Simpler models generalize better** — 85-param linear > 19K-param MLP for direction
3. **Magnitude ≠ Direction** — LSTM predicts return size well (R²≈0) but not sign
4. **Public data is already priced in** — Adding FGI, on-chain, macro doesn't help
5. **More features ≠ better predictions** — 40 features performed same as 21
6. **Ensemble doesn't help** — When all models are ~50%, averaging stays ~50%

### Why It Doesn't Work

| Factor | Explanation |
|---|---|
| **Efficient Market Hypothesis** | All public info is reflected in price |
| **Signal-to-noise ratio** | ~0.01 — virtually no signal to extract |
| **Regime changes** | Bull/bear markets have completely different dynamics |
| **Adaptive markets** | Any discovered edge gets arbitraged away |
| **Non-stationarity** | Past patterns don't predict future |

### What Would Actually Work (Theoretically)

| Approach | Requirement | Feasibility |
|---|---|---|
| **Proprietary order book** | Exchange partnership ($50K+/mo) | ❌ Expensive |
| **Insider on-chain tracking** | Nansen/Glassnode Pro ($100+/mo) | ⚠️ Maybe |
| **News NLP with transformer** | Real-time Twitter/Reddit scraping | ⚠️ Possible |
| **Market microstructure** | Tick-level data, colocation | ❌ Institutional only |
| **Options flow analysis** | Deribit/Binance options API | ⚠️ Limited edge |

---

## 📚 References

- **D2L (Dive into Deep Learning):** https://d2l.ai — Chapters 3, 5, 10, 11, 12
- **CoinMarketCap API:** https://coinmarketcap.com/api/
- **Alternative.me Fear & Greed:** https://alternative.me/crypto/fear-and-greed-index/
- **Binance API:** https://binance-docs.github.io/apidocs/
- **Blockchain.com Charts:** https://api.blockchain.info/charts/
- **yfinance:** https://github.com/ranaroussi/yfinance

---

## ⚠️ Disclaimer

This research is for **educational purposes only**. None of this constitutes financial advice.
Cryptocurrency investments carry substantial risk of loss. Past model performance does not
predict future results. All models tested failed to achieve statistically significant
direction prediction accuracy.

---

## 📄 License

MIT — Free to use, modify, and distribute.

---

> *"The market can stay irrational longer than you can stay solvent."* — John Maynard Keynes
