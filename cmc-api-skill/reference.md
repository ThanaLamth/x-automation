# CMC-API Skill — CoinMarketCap CLI Wrapper

## Setup

### 1. Install CMC CLI
```bash
curl -sSfL https://raw.githubusercontent.com/coinmarketcap-official/CoinMarketCap-CLI/main/install.sh | sh
```

### 2. Set API Key
```bash
# Option A: Environment variable (recommended for automation)
export CMC_API_KEY="your-api-key"

# Option B: Save to file
echo "your-api-key" > ~/.qwen/skills/cmc-api/api-key.txt
```

Get API key: https://pro.coinmarketcap.com/signup

### 3. Make executable
```bash
chmod +x ~/.qwen/skills/cmc-api/cmc-skill.sh
```

## Usage

```bash
# Quick price check
~/.qwen/skills/cmc-api/cmc-skill.sh price 1           # Bitcoin
~/.qwen/skills/cmc-api/cmc-skill.sh price ethereum     # Ethereum by slug

# Search
~/.qwen/skills/cmc-api/cmc-skill.sh search RAVE        # Search RAVE
~/.qwen/skills/cmc-api/cmc-skill.sh search btc         # Search BTC

# Market overview
~/.qwen/skills/cmc-api/cmc-skill.sh trending            # Trending coins
~/.qwen/skills/cmc-api/cmc-skill.sh markets 20          # Top 20 markets
~/.qwen/skills/cmc-api/cmc-skill.sh gainers             # Gainers & losers

# Historical data
~/.qwen/skills/cmc-api/cmc-skill.sh history 1 30        # BTC 30-day history
~/.qwen/skills/cmc-api/cmc-skill.sh history ethereum 7  # ETH 7-day history

# News
~/.qwen/skills/cmc-api/cmc-skill.sh news                # Latest crypto news

# Resolve asset details
~/.qwen/skills/cmc-api/cmc-skill.sh resolve 1           # Get BTC details

# Check connection
~/.qwen/skills/cmc-api/cmc-skill.sh status              # API health check
```

## Available Commands

| Command | Args | Description |
|---------|------|-------------|
| `price` | `<id>` or `<slug>` | Get live price with market data |
| `search` | `<query>` | Search coins by name/symbol |
| `trending` | — | Get trending cryptocurrencies |
| `markets` | `[limit]` | Top markets by volume (default: 10) |
| `history` | `<id|slug>` `[days]` | Price history (default: 7 days) |
| `news` | — | Latest crypto news |
| `gainers` | — | Top gainers and losers (24h) |
| `resolve` | `<id>` or `<slug>` | Get asset metadata |
| `status` | — | Check API connection |

## Raw CMC CLI Commands

The wrapper uses these underlying commands:

```bash
cmc resolve --id 1          # Resolve asset
cmc price --id 1 -o json    # Get price
cmc search "bitcoin" -o json  # Search
cmc trending -o json        # Trending
cmc markets --limit 10 -o json  # Markets
cmc history --id 1 --days 30 -o json  # History
cmc news -o json            # News
cmc top-gainers-losers -o json  # Gainers/losers
cmc status -o json          # API status
```

## Output Format

All commands return formatted output with:
- **Emojis** for visual clarity (🟢🔴📈📊📰)
- **Aligned columns** for readability
- **Filtered results** (top 10-20 items to avoid flooding)

## Integration with Qwen Code

Use in conversation:
```
Check BTC price
What's trending on CMC?
Get news about RAVE token
Show me top gainers today
```
