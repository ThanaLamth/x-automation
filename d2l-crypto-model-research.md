# Áp Dụng D2L (Dive into Deep Learning) Vào Xây Dựng Mô Hình Dự Báo Giá Crypto

> **Nghiên cứu dựa trên:** Dive into Deep Learning (d2l.ai) — 23 Chapters
> **Mục tiêu:** Xây dựng model dự báo giá crypto từ baseline đến SOTA
> **Ngày:** April 14, 2026

---

## 📋 Mục Lục

1. [Tổng Quan Sách D2L](#1-tổng-quan-sách-d2l)
2. [Cấp Độ 1: Baseline Models](#2-cấp-độ-1-baseline-models)
3. [Cấp Độ 2: Sequence Models (LSTM/GRU)](#3-cấp-độ-2-sequence-models-lstmgru)
4. [Cấp Độ 3: Transformer (State-of-the-Art)](#4-cấp-độ-3-transformer-state-of-the-art)
5. [Cấp Độ 4: Advanced Techniques](#5-cấp-độ-4-advanced-techniques)
6. [Kiến Trúc Đề Xuất Production](#6-kiến-trúc-đề-xuất-production-ready)
7. [Những Cạm Bẫy Phải Tránh](#7-những-cạm-bẫy-phải-tránh)
8. [Lộ Trình Implement](#8-lộ-trình-implement)

---

## 1. Tổng Quan Sách D2L

Sách **Dive into Deep Learning** có **23 chapters chính + 2 appendices**:

| Chapters | Nội Dung | Áp Dụng Cho Crypto |
|----------|----------|-------------------|
| **Ch 1-2** | Intro, Preliminaries (data, linear algebra, calculus) | Nền tảng toán |
| **Ch 3** | Linear Regression | ✅ Baseline model |
| **Ch 4** | Linear Classification | ❌ Không dùng (regression, không phải classification) |
| **Ch 5** | Multilayer Perceptrons | ✅ Baseline phi tuyến |
| **Ch 6** | Builders' Guide (OOP, GPU) | ✅ Design pattern + hardware |
| **Ch 7-8** | CNNs (LeNet → ResNet → DenseNet) | ⚠️ Chỉ dùng cho chart image pattern recognition |
| **Ch 9** | RNNs (vanilla, BPTT) | ❌ Vanilla RNN — vanishing gradient |
| **Ch 10** | Modern RNNs (LSTM, GRU, Seq2Seq, Beam Search) | ✅ **Core architecture chính** |
| **Ch 11** | Attention & Transformers | ✅ **SOTA architecture** |
| **Ch 12** | Optimization (SGD, Adam, LR scheduling) | ✅ Training optimization |
| **Ch 13** | Computational Performance (GPU, async) | ✅ Scaling |
| **Ch 14** | Computer Vision | ⚠️ Chart pattern recognition |
| **Ch 15** | NLP Pretraining (word2vec, BERT) | ✅ Transfer learning across coins |
| **Ch 16** | NLP Applications (sentiment, fine-tuning BERT) | ✅ Fine-tuning strategy |
| **Ch 17** | Reinforcement Learning | ⚠️ Trading strategy optimization |
| **Ch 18** | Gaussian Processes | ⚠️ Uncertainty quantification |
| **Ch 19** | Hyperparameter Optimization | ✅ Auto-tune model |
| **Ch 20** | GANs | ⚠️ Synthetic data generation |
| **Ch 21** | Recommender Systems | ❌ Không liên quan trực tiếp |
| **Ch 22-23** | Math & Tools Appendices | Tham khảo |

---

## 2. Cấp Độ 1: Baseline Models

### 2.1 Linear Regression (Chapter 3)

**Tại sao cần:** Establish performance floor. Mọi model phức tạp hơn PHẢI beat được baseline này.

**Pipeline:**
```python
Features: [lagged_returns(1h, 4h, 24h), volume_change, RSI, MACD]
  → StandardScaler
  → Linear Regression
  → Loss: MSE (hoặc Huber cho robustness)
  → Optimizer: SGD với momentum
  → Weight Decay (L2 regularization)
```

**Loss Functions:**
| Loss | Ưu Điểm | Nhược Điểm | Khi Nào Dùng |
|---|---|---|---|
| **MSE** | Standard, convex | Nhạy với outliers (flash crash) | Baseline |
| **MAE** | Robust với outliers | Gradient không đổi → convergence chậm | Crypto volatile |
| **Huber** | Best of both — quadratic nhỏ, linear lớn | Need tune delta | **Recommended cho crypto** |

**Generalization (Section 3.6):**
- Track training vs validation error
- Underfitting: cả hai đều cao → tăng model capacity
- Overfitting: train thấp, val cao → thêm regularization

---

### 2.2 Multilayer Perceptrons (Chapter 5)

**Tại sao:** MLP bắt được quan hệ phi tuyến mà linear regression bỏ sót.

**Architecture:**
```python
Sequential(
    Linear(input_features, 128),
    ReLU(),
    Dropout(0.3),
    Linear(128, 64),
    ReLU(),
    Dropout(0.2),
    Linear(64, 1)  # Linear output cho regression
)
```

**Regularization cho Financial Data:**

| Technique | Cơ Chế | Khi Nào Dùng |
|---|---|---|
| **Weight Decay (L2)** | Penalize large weights | **Luôn luôn dùng** |
| **Dropout** | Random zero neurons | Khi hidden units > 64 |
| **Early Stopping** | Stop khi val loss không improve | **Bắt buộc cho noisy data** |
| **K-Fold CV** | Robust model selection | Dataset nhỏ |

**Activation Functions:**
- **ReLU** — default choice, tránh vanishing gradient
- **Leaky ReLU** — nếu nhiều neurons "chết" trong training
- **Output layer: LINEAR** (KHÔNG có activation) cho regression

---

### 2.3 Builders' Guide — OOP Design (Chapter 6)

**Tại sao quan trọng:** Clean design → dễ iterate, swap components.

**Design Pattern:**
```python
class PricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.predictor = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        features = self.feature_encoder(x)
        return self.predictor(features)

    def predict(self, x):
        """Inference mode — no gradients"""
        with torch.no_grad():
            return self.forward(x)
```

**Lazy Initialization (Section 6.4):** Parameters allocated on first forward pass → dễ prototype với nhiều feature dimensions khác nhau.

**Model Persistence (Section 6.6):**
```python
# Save
torch.save(model.state_dict(), 'models/baseline_mlp.pth')

# Load
model.load_state_dict(torch.load('models/baseline_mlp.pth'))
```

---

### 2.4 GPU Training (Chapter 13)

**Tại sao:** Hyperparameter sweep + large datasets cần GPU.

**Key Optimizations:**
- Move model + data to GPU: `model.to(device)`, `data.to(device)`
- Minimize CPU↔GPU transfers (preload data)
- Asynchronous computation: overlap preprocessing với forward/backward pass
- Automatic parallelism: framework tự schedule independent ops

---

## 3. Cấp Độ 2: Sequence Models (LSTM/GRU)

### 3.1 Tại Sao RNN Thông Thường Không Dùng (Chapter 9)

**Vanilla RNN Problem:**
```python
# Vanilla RNN cell
H = tanh(x @ W_xh + H_prev @ W_hh + b_h)
```

**BPTT (Backpropagation Through Time):** Unroll RNN qua time steps → chain rule qua nhiều steps → gradients vanish/explode.

**Hệ quả:** Cannot learn dependencies > ~20 time steps. Với crypto data (daily/weekly cycles), điều này là không chấp nhận được.

**✅ Giải pháp: LSTM hoặc GRU**

---

### 3.2 LSTM Architecture (Chapter 10.1)

**Kiến trúc:**
```
Input Gate:    I_t = σ(X_t·W_xi + H_{t-1}·W_hi + b_i)
Forget Gate:   F_t = σ(X_t·W_xf + H_{t-1}·W_hf + b_f)
Cell Candidate: C̃_t = tanh(X_t·W_xc + H_{t-1}·W_hc + b_c)
Cell State:    C_t = F_t ⊙ C_{t-1} + I_t ⊙ C̃_t
Output Gate:   O_t = σ(X_t·W_xo + H_{t-1}·W_ho + b_o)
Hidden State:  H_t = O_t ⊙ tanh(C_t)
```

**Tại sao LSTM phù hợp crypto:**

| Gate | Vai Trò | Ví Dụ Crypto |
|---|---|---|
| **Forget Gate** | Quên thông tin cũ | Bỏ qua flash crash noise |
| **Input Gate** | Nhận thông tin mới | Nhận diện pump mới bắt đầu |
| **Cell State** | Nhớ dài hạn | Giữ bull/bear trend (tuần-tháng) |
| **Output Gate** | Điều chỉnh output | Confidence của prediction |

**Code Implementation:**
```python
class CryptoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state=None):
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x, state)

        # Lấy output của time step cuối cùng
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        last_hidden = self.dropout(last_hidden)

        prediction = self.fc(last_hidden)  # (batch, 1)
        return prediction, (h_n, c_n)
```

---

### 3.3 GRU Architecture (Chapter 10.2)

**Kiến trúc (đơn giản hơn LSTM):**
```
Reset Gate:  R_t = σ(X_t·W_xr + H_{t-1}·W_hr + b_r)
Update Gate: Z_t = σ(X_t·W_xz + H_{t-1}·W_hz + b_z)
Candidate:   H̃_t = tanh(X_t·W_xh + (R_t ⊙ H_{t-1})·W_hh + b_h)
Hidden:      H_t = Z_t ⊙ H_{t-1} + (1 - Z_t) ⊙ H̃_t
```

**GRU vs LSTM cho Crypto:**

| Aspect | LSTM | GRU |
|---|---|---|
| Parameters | Nhiều hơn (~4x) | Ít hơn (~3x) |
| Training Speed | Chậm hơn | **Nhanh hơn 20-30%** |
| Memory | Cần nhiều hơn | Tiết kiệm hơn |
| Accuracy | Tốt hơn chút | Gần bằng LSTM |
| Phù hợp | Daily/4h data | **1m/5m/15m data** |

**Khi nào dùng GRU:** High-frequency trading, resource-constrained deployment, khi cần fast inference.

---

### 3.4 Deep RNNs (Chapter 10.3)

**Stack nhiều layers:**
```python
self.lstm = nn.LSTM(
    input_size=12,
    hidden_size=128,
    num_layers=3,  # 3 layers
    batch_first=True,
    dropout=0.2
)
```

**Hierarchical Learning:**
| Layer | Bắt Được Pattern | Ví Dụ |
|---|---|---|
| **Layer 1** | Local, ngắn hạn | Tick noise, volume spike |
| **Layer 2** | Medium-term | Daily momentum, support/resistance |
| **Layer 3** | Global, dài hạn | Weekly cycles, macro trend |

**Recommended:** 2-3 layers cho crypto. Nhiều hơn → overfitting risk cao.

---

### 3.5 Bidirectional RNNs (Chapter 10.4)

**⚠️ CẢNH BÁO:** Bidirectional RNNs dùng future data → **KHÔNG DÙNG cho live trading**.

**Chỉ dùng cho:**
- Offline backtesting
- Feature engineering (tạo additional features từ full-window context)
- Identifying support/resistance levels relative to surrounding prices

---

### 3.6 Encoder-Decoder + Seq2Seq (Chapter 10.6-10.7)

**Multi-Step Forecasting:**
```
Encoder:  Input  = 96 giờ historical data (OHLCV + indicators)
          Output = Context vector (compressed market state)

Decoder:  Input  = Context vector + BOS token
          Output = 24 giờ future price predictions
```

**Teacherforcing (Training):**
```python
# Training với teacher forcing
encoder_input = batch[:, :96, :]   # 96 hours history
decoder_input = batch[:, 96:120, :] # 24 hours future (shifted)
target = batch[:, 97:121, 0]        # True targets

# Encode
context, (h_n, c_n) = encoder(encoder_input)

# Decode với teacher forcing
output = decoder(decoder_input, (h_n, c_n))
loss = criterion(output, target)
```

**Inference (Live):**
```python
# Live prediction — autoregressive, không có future data
predictions = []
current_input = historical_data  # (1, 96, features)
h_n, c_n = encoder(current_input)

for step in range(24):
    pred, (h_n, c_n) = decoder(predictions[-1] if predictions else None, (h_n, c_n))
    predictions.append(pred)
```

---

### 3.7 Beam Search (Chapter 10.8)

**Ứng dụng:** Generate multiple future trajectories → scenario analysis.

```python
# Generate K=3 trajectories
trajectories = beam_search(decoder, context_vector, k=3, steps=24)

# Result: 3 possible future paths
# Path 1 (bull):   Price → $80K, $82K, $85K, ...
# Path 2 (base):   Price → $75K, $76K, $74K, ...
# Path 3 (bear):   Price → $70K, $68K, $65K, ...
```

**Risk Management:**
- Tính **Value-at-Risk (VaR)** từ worst trajectory
- Dynamic position sizing dựa trên scenario probabilities
- Stop-loss placement dựa trên bear case

---

## 4. Cấp Độ 3: Transformer (State-of-the-Art)

### 4.1 Attention Mechanism — Tại Sao Quan Trọng (Chapter 11.1-11.3)

**Core concept:**
```
Output = Σ (α_i · v_i)
α_i = softmax(score(q, k_i))
```

**Ánh xạ sang Time Series:**
- **Query** = Time step hiện tại muốn predict
- **Keys** = Historical time steps (past prices, volumes)
- **Values** = Observed features tại mỗi historical step
- **Attention weights** = Adaptive kernel — học được historical period nào quan trọng nhất

**So với traditional methods:**

| Method | Kernel | Flexibility |
|---|---|---|
| Moving Average | Fixed (equal weights) | Không linh hoạt |
| Exponential Smoothing | Fixed decay | Chỉ 1 parameter |
| **Attention** | **Learned** | **Tự động learn weights** |

**Attention Scoring Functions:**

| Scoring | Formula | Khi Nào Dùng |
|---|---|---|
| **Dot Product** | qᵀk | Fast nhưng unstable với nhiều features |
| **Scaled Dot Product** | qᵀk/√d | **Default choice** — ổn định với 50+ features |
| **Additive (Bahdanau)** | Wᵥᵀ tanh(Wq·q + Wk·k) | Khi query/key dimensions khác nhau |

---

### 4.2 Multi-Head Attention — Lợi Thế Lớn Nhất (Chapter 11.5)

**Mỗi attention head học một "aspect" khác nhau của market:**

| Head | Focus | Ví Dụ |
|---|---|---|
| **Head 1** | Short-term momentum | 5-10 recent candles |
| **Head 2** | Support/Resistance | Price levels historically causing reversals |
| **Head 3** | Volume anomalies | High-volume periods |
| **Head 4** | Cyclical patterns | Daily/weekly seasonality |
| **Head 5** | Regime shifts | Volatility breakouts |
| **Head 6** | Cross-asset correlation | BTC dominance effect |
| **Head 7** | Liquidity signals | Order book imbalances |
| **Head 8** | Sentiment indicators | Fear & Greed index alignment |

**Implementation:**
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads  # 32 per head
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        Q = self.W_q(x).view(x.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(x.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(x.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        output = (attn @ V).transpose(1, 2).contiguous()
        output = output.view(x.size(0), -1, self.num_heads * self.d_k)
        return self.W_o(output)
```

---

### 4.3 Self-Attention cho Financial Sequences (Chapter 11.6)

**Key advantage:** O(1) sequential operations (fully parallelizable), vs O(n) cho RNNs.

**Long-range dependencies:**
- Price movement hôm nay có thể depend on pattern từ 30 ngày trước
- Self-attention: capture trong **1 step**
- LSTM: gradient decay sau ~20 steps

**Computational cost:** O(n² · d)
- High-frequency data (1000s candles) → expensive
- Solution: chunked windows, restricted attention

---

### 4.4 Positional Encoding (Chapter 11.6)

**Problem:** Self-attention permutation-invariant.

**Sinusoidal encoding:**
```
PE[i, 2j]   = sin(i / 10000^(2j/d))
PE[i, 2j+1] = cos(i / 10000^(2j/d))
```

**Crypto-specific adaptations:**
- Add **time-of-day** encoding (Asian vs US session)
- Add **day-of-week** encoding (weekend volume drop)
- **Learned positional embeddings** nếu data irregular (missing candles)

---

### 4.5 Transformer Architecture cho Forecasting (Chapter 11.7)

**KHÔNG dùng Encoder-only (BERT-style)** — bidirectional = lookahead bias!

**Dùng Decoder-only (GPT-style)** — causal masking:

```python
class CryptoTransformer(nn.Module):
    def __init__(self, input_size=12, d_model=256, num_heads=8,
                 num_layers=4, ff_dim=1024, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_proj(x)  # → (batch, seq_len, d_model)

        # Causal mask — position t chỉ attend ≤ t
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))

        x = self.transformer(x, mask=mask)  # (batch, seq_len, d_model)

        # Lấy last position's representation
        last_hidden = x[:, -1, :]  # (batch, d_model)
        return self.output_proj(last_hidden)  # (batch, 1)
```

**Key Design Choices — Time Series vs NLP:**

| Aspect | NLP | Time Series Forecasting |
|---|---|---|
| Input tokens | Discrete words | Continuous feature vectors |
| Embedding | Lookup table | Linear projection |
| Positional encoding | Sinusoidal | Sinusoidal + temporal features |
| Output | Classification | **Regression (price/return)** |
| Model size | Billions | **Thousands-millions** (avoid overfitting) |

---

## 5. Cấp Độ 4: Advanced Techniques

### 5.1 Pretraining Strategy (Chapter 15)

**Transfer learning across coins:**
```
Step 1: Pretrain trên 1000+ coins historical data
        Task: Masked price prediction (như BERT masked token)
        Data: Decades of stock data + years of crypto data

Step 2: Fine-tune trên specific coin (BTC/ETH)
        Data: Recent 1-2 years of hourly candles
        LR: 1e-5 (small — don't destroy pretrained weights)

Step 3: Few-shot adaptation cho new coins
        Data: Vài trăm candles
        LR: 1e-6, freeze most layers
```

### 5.2 Hyperparameter Optimization (Chapter 19)

**Auto-tune:**
```python
search_space = {
    'lookback_window': [48, 96, 168, 336, 720],  # 2h to 30 days
    'hidden_size': [64, 128, 256, 512],
    'num_layers': [2, 3, 4, 6],
    'num_heads': [4, 8, 16],
    'dropout': [0.1, 0.15, 0.2, 0.3],
    'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],
    'weight_decay': [1e-4, 1e-3, 1e-2],
}

# Async random search or ASHA (Asynchronous Successive Halving)
```

### 5.3 GANs for Synthetic Data (Chapter 20)

**Problem:** Financial data limited, especially for new coins.

**Solution:** Use GANs to generate synthetic price sequences that preserve:
- Return distribution (fat tails)
- Volatility clustering
- Autocorrelation structure

**Data augmentation → more robust model training.**

---

## 6. Kiến Trúc Đề Xuất (Production-Ready)

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA PIPELINE                               │
│                                                                       │
│  Raw OHLCV → Feature Engineering → StandardScaler → Sequence Batches  │
│                                                                       │
│  Features (14 per timestep):                                          │
│    • Price: close, open, high, low                                    │
│    • Volume: volume, volume_change                                    │
│    • Returns: return_1h, return_4h, return_24h                        │
│    • Indicators: RSI(14), MACD, Bollinger_width                       │
│    • On-chain: funding_rate, open_interest                            │
│                                                                       │
│  Lookback window: 96 steps (4 days of hourly data)                    │
│  Target: next-step return (regression)                                │
└──────────────────────────────┬────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     MODEL: Transformer Decoder                       │
│                                                                       │
│  Input:          (batch=32, seq_len=96, features=14)                 │
│                                                                       │
│  Embedding:      Linear(14 → 256) + Positional Encoding              │
│                  (sinusoidal + hour-of-day + day-of-week)             │
│                                                                       │
│  Transformer:    4 layers × Causal Multi-Head Self-Attention          │
│                  • 8 heads × 32 dim per head = 256 total              │
│                  • Positionwise FFN: 256 → 1024 → 256                │
│                  • LayerNorm + Residual connections                   │
│                  • Dropout: 0.1                                       │
│                                                                       │
│  Output:         Last token representation → Linear(256, 1)          │
│                  = predicted next-step return                         │
└──────────────────────────────┬────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                          TRAINING                                     │
│                                                                       │
│  Loss:           Huber (delta=1.0) — robust to outliers              │
│  Optimizer:      AdamW (lr=1e-4, weight_decay=1e-2)                  │
│  LR Scheduler:   Cosine annealing + warmup (1000 steps)              │
│  Gradient Clip:  1.0 (prevent explosions)                            │
│  Early Stop:     Patience=20 epochs on validation loss               │
│  Validation:     Walk-forward (NO random split!)                     │
│                                                                       │
│  Walk-forward split:                                                  │
│    Train: Jan 2021 - Dec 2022                                        │
│    Val:   Jan 2023 - Jun 2023                                        │
│    Test:  Jul 2023 - Present                                         │
└──────────────────────────────┬────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     INFERENCE & RISK                                  │
│                                                                       │
│  Single prediction:                                                   │
│    → Predicted return: +2.3%                                          │
│    → Convert to price: current_price × (1 + return)                   │
│                                                                       │
│  Multi-trajectory (Beam Search, K=3):                                 │
│    → Bull case:   +5.2% → $78.2K                                     │
│    → Base case:   +2.3% → $76.1K                                     │
│    → Bear case:   -1.8% → $72.3K                                     │
│                                                                       │
│  Risk metrics:                                                        │
│    → VaR (95%): -3.2%                                                 │
│    → Expected shortfall: -4.1%                                        │
│    → Position sizing: Kelly criterion based on prediction confidence   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Những Cạm Bẫy Phải Tránh

| Vấn Đề | Hậu Quả | Cách Fix |
|---|---|---|
| **Lookahead bias** | Model "nhìn tương lai" → backtest đẹp, live trade thua | Causal masking, time-based split, audit feature pipeline |
| **Overfitting** | Train loss thấp, val loss cao | Dropout 0.1-0.2, weight decay, small model, early stopping |
| **Non-stationarity** | Train bull market, fail bear market | Regime-aware training, continuous fine-tuning, rolling window |
| **Low signal-to-noise** | Crypto noise >> signal | Heavy feature engineering, ensemble models, longer lookback |
| **Data leakage** | Future info in features | Audit every feature — ensure no future data used |
| **Position encoding mismatch** | Train 1h, infer 15m → broken | Keep consistent timeframe across train/inference |
| **Over-optimization** | Curve-fitting to historical data | Out-of-sample testing, walk-forward validation |
| **Ignoring transaction costs** | Profitable on paper, loss in reality | Include fees, slippage, spread in backtest |

---

## 8. Lộ Trình Implement

| Phase | Model | Thời Gian | Metric Target | Dependencies |
|---|---|---|---|---|
| **1** | Linear Regression | 1 ngày | RMSE baseline | Ch 3, 12 |
| **2** | MLP (2 hidden layers) | 2 ngày | Beat Linear 5%+ | Ch 5, 6 |
| **3** | LSTM (2 layers, 128 units) | 3 ngày | Beat MLP 5%+ | Ch 10 |
| **4** | GRU (fast inference) | 2 ngày | ~LSTM, 2x faster | Ch 10 |
| **5** | Transformer Decoder (4 layers) | 5 ngày | Beat LSTM 3-5% | Ch 11, 12 |
| **6** | Multi-coin Pretrain → Fine-tune | 1 tuần | Generalize across assets | Ch 15, 16 |
| **7** | Ensemble (LSTM + Transformer) | 3 ngày | Best robustness | All above |
| **8** | Hyperparameter Tuning | 3 ngày | Optimize all models | Ch 19 |
| **9** | Production Deployment | 1 tuần | Live paper trading | Ch 13 |

### Default Hyperparameters

| Parameter | Linear | MLP | LSTM | Transformer |
|---|---|---|---|---|
| Hidden Size | — | 128 | 128 | 256 (d_model) |
| Layers | 1 | 2 | 2 | 4 |
| Dropout | — | 0.3 | 0.2 | 0.1 |
| Learning Rate | 1e-2 | 1e-3 | 1e-3 | 1e-4 |
| Weight Decay | 1e-3 | 1e-4 | 1e-4 | 1e-2 |
| Batch Size | 256 | 128 | 32 | 32 |
| Loss | Huber | Huber | MSE | Huber |
| Lookback | 24 | 48 | 96 | 96-336 |

---

## 📚 Tài Liệu Tham Khảo

- **Dive into Deep Learning:** https://d2l.ai
- **GitHub Repository:** https://github.com/d2l-ai/d2l-en
- **PyTorch Documentation:** https://pytorch.org/docs/
- **CoinMarketCap API:** https://coinmarketcap.com/api/
- **Original Transformer Paper:** "Attention Is All You Need" (Vaswani et al., 2017)

---

*Nghiên cứu này được tổng hợp từ 23 chapters của D2L + áp dụng thực tế vào crypto price prediction.*
*Ngày tạo: April 14, 2026*
