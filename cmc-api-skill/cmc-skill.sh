#!/usr/bin/env bash
# ============================================================
# CMC-API Skill Wrapper — CoinMarketCap CLI
# Usage: ./cmc-skill.sh <command> [args]
# ============================================================
set -euo pipefail
export PATH="/home/qwen/.local/bin:$PATH"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

log_info()  { echo -e "${GREEN}[+]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[-]${NC} $1"; }
log_cmd()   { echo -e "${BLUE}[>]${NC} $1"; }

usage() {
    echo -e "${GREEN}CMC-API Skill${NC} — CoinMarketCap CLI Wrapper"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  price    <id|slug>    Get price & market data"
    echo "  search   <query>      Search crypto by name/symbol"
    echo "  trending              Get trending coins"
    echo "  markets  [limit]      Get top markets by volume"
    echo "  history  <id> [days]  Get price history (default: 7 days)"
    echo "  news                  Get crypto news"
    echo "  resolve  <id|slug>    Resolve asset details"
    echo "  gainers               Get top gainers & losers"
    echo "  status                Check API connection"
    echo ""
    echo "Examples:"
    echo "  $0 price 1            # Bitcoin price"
    echo "  $0 search RAVE        # Search RAVE"
    echo "  $0 trending           # Trending coins"
    echo "  $0 history 1 30       # BTC 30-day history"
}

load_api_key() {
    local keyfile="/home/qwen/.qwen/skills/cmc-api/api-key.txt"
    if [ -z "${CMC_API_KEY:-}" ] && [ -f "$keyfile" ]; then
        export CMC_API_KEY="$(cat "$keyfile")"
    fi
}

check_api_key() {
    if [ -z "${CMC_API_KEY:-}" ]; then
        log_error "CMC_API_KEY not set!"
        echo "  Export: export CMC_API_KEY='your-key'"
        echo "  Or save: echo 'your-key' > ~/.qwen/skills/cmc-api/api-key.txt"
        exit 1
    fi
}

# Helper: run CMC command with JSON output + retry on rate limit
run_cmc() {
    local max_retries=3
    local retry=0
    local result=""
    while [ $retry -lt $max_retries ]; do
        result=$(cmc "$@" -o json 2>&1)
        if echo "$result" | grep -q "rate_limited"; then
            retry=$((retry + 1))
            local wait_time=$((retry * 10))
            log_warn "Rate limited. Waiting ${wait_time}s... (attempt $retry/$max_retries)"
            sleep $wait_time
        else
            echo "$result"
            return 0
        fi
    done
    echo "$result"
}

# Check if response is an error
check_error() {
    local json="$1"
    if echo "$json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
if 'error' in d:
    print(d['error'])
    sys.exit(1)
" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

cmd_price() {
    local asset="${1:?Usage: $0 price <id|slug>}"
    log_cmd "Getting price for: $asset"
    local json_out
    if [[ "$asset" =~ ^[0-9]+$ ]]; then
        json_out=$(run_cmc price --id "$asset")
    else
        json_out=$(run_cmc price --slug "$asset")
    fi
    echo "$json_out" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    # Response is keyed by coin ID
    for cid, coin in d.items():
        n = coin.get('name','?')
        s = coin.get('symbol','?')
        r = coin.get('cmc_rank','?')
        q = coin.get('quote',{}).get('USD',{})
        p = q.get('price',0)
        c1 = q.get('percent_change_1h',0)
        c24 = q.get('percent_change_24h',0)
        c7 = q.get('percent_change_7d',0)
        mc = q.get('market_cap',0)
        v = q.get('volume_24h',0)
        print(f'  {n} ({s})  Rank: {r}')
        print(f'  Price:  \${p:,.2f}')
        print(f'  1h:     {c1:+.2f}%')
        print(f'  24h:    {c24:+.2f}%')
        print(f'  7d:     {c7:+.2f}%')
        print(f'  MCap:   \${mc:,.0f}')
        print(f'  Vol:    \${v:,.0f}')
except Exception as e:
    out = sys.stdin.read()[:500]
    if out:
        print(out)
    else:
        print(f'Error: {e}')
"
}

cmd_search() {
    local query="${1:?Usage: $0 search <query>}"
    log_cmd "Searching: $query"
    run_cmc search "$query" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if isinstance(data, list):
        print(f'Found {len(data)} results:')
        print()
        for i, c in enumerate(data[:20], 1):
            n = c.get('name','?')
            s = c.get('symbol','?')
            cid = str(c.get('id','?'))
            r = c.get('cmc_rank','?')
            q = c.get('quote',{}).get('USD',{}).get('price',0)
            print(f'  {i:2d}. {n:20s} ({s:6s})  ID:{cid:6s}  Rank:{str(r):5s}  \${q}')
    else:
        print(json.dumps(data, indent=2)[:1000])
except:
    print(sys.stdin.read()[:1000])
"
}

cmd_trending() {
    log_cmd "Getting trending coins..."
    run_cmc trending | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if isinstance(data, list):
        print('Trending Coins:')
        print()
        for i, c in enumerate(data[:20], 1):
            n = c.get('name','?')
            s = c.get('symbol','?')
            r = c.get('cmc_rank','?')
            ch = c.get('quote',{}).get('USD',{}).get('percent_change_24h',0)
            print(f'  {i:2d}. {n:20s} ({s:6s})  Rank:{str(r):5s}  24h:{ch:+.2f}%')
    else:
        print(json.dumps(data, indent=2)[:1000])
except:
    print(sys.stdin.read()[:1000])
"
}

cmd_markets() {
    local limit="${1:-10}"
    log_cmd "Getting top $limit markets..."
    run_cmc markets --limit "$limit" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if isinstance(data, list):
        print(f'Top {min(len(data),int('$limit'))} Markets by Volume:')
        print()
        for i, m in enumerate(data[:20], 1):
            pair = m.get('pair','?')
            ex = m.get('exchange_name','?')
            vol = m.get('volume_usd','?')
            print(f'  {i:2d}. {pair:20s} | {ex:15s} | Vol: \${vol}')
    else:
        print(json.dumps(data, indent=2)[:1000])
except:
    print(sys.stdin.read()[:1000])
"
}

cmd_history() {
    local asset="${1:?Usage: $0 history <id|slug> [days]}"
    local days="${2:-7}"
    log_cmd "Getting ${days}-day history for: $asset"
    local json_out
    if [[ "$asset" =~ ^[0-9]+$ ]]; then
        json_out=$(run_cmc history --id "$asset" --days "$days")
    else
        json_out=$(run_cmc history --slug "$asset" --days "$days")
    fi
    echo "$json_out" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    quotes = d.get('quotes',[])
    if quotes:
        prices = [q.get('quote',{}).get('USD',{}).get('price',0) for q in quotes]
        dates = [q.get('timestamp','')[:10] for q in quotes]
        if prices:
            print(f'Price History ({len(quotes)} data points):')
            print()
            print(f'  Period: {dates[0]} to {dates[-1]}')
            print(f'  High:   \${max(prices):,.2f}')
            print(f'  Low:    \${min(prices):,.2f}')
            print(f'  Latest: \${prices[-1]:,.2f}')
            print()
            print('  Recent data:')
            for dt, pr in list(zip(dates, prices))[-7:]:
                print(f'    {dt}: \${pr:,.2f}')
    else:
        print(json.dumps(d, indent=2)[:1000])
except Exception as e:
    print(f'Error: {e}')
    print(sys.stdin.read()[:500])
"
}

cmd_news() {
    log_cmd "Getting crypto news..."
    run_cmc news | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if isinstance(data, list):
        print('Latest Crypto News:')
        print()
        for i, a in enumerate(data[:15], 1):
            t = a.get('title','?')
            src = a.get('source','?')
            url = a.get('url','')[:80]
            dt = a.get('published_at','')[:10] or a.get('publishedAt','')[:10]
            print(f'  {i:2d}. [{dt}] {t}')
            print(f'      Source: {src} | {url}')
            print()
    else:
        print(json.dumps(data, indent=2)[:2000])
except:
    print(sys.stdin.read()[:2000])
"
}

cmd_gainers() {
    log_cmd "Getting top gainers and losers..."
    run_cmc top-gainers-losers | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if isinstance(data, dict):
        g = data.get('gainers',[])[:10]
        l = data.get('losers',[])[:10]
        if g:
            print('Top Gainers:')
            for x in g:
                nm = x.get('name','?')
                sy = x.get('symbol','?')
                ch = x.get('quote',{}).get('USD',{}).get('percent_change_24h',0)
                print(f'  {nm:20s} ({sy})  {ch:+.2f}%')
        print()
        if l:
            print('Top Losers:')
            for x in l:
                nm = x.get('name','?')
                sy = x.get('symbol','?')
                ch = x.get('quote',{}).get('USD',{}).get('percent_change_24h',0)
                print(f'  {nm:20s} ({sy})  {ch:+.2f}%')
    else:
        print(json.dumps(data, indent=2)[:2000])
except:
    print(sys.stdin.read()[:2000])
"
}

cmd_resolve() {
    local asset="${1:?Usage: $0 resolve <id|slug>}"
    log_cmd "Resolving: $asset"
    local json_out
    if [[ "$asset" =~ ^[0-9]+$ ]]; then
        json_out=$(run_cmc resolve --id "$asset")
    else
        json_out=$(run_cmc resolve --slug "$asset")
    fi
    echo "$json_out" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    nm = d.get('name','?')
    sy = d.get('symbol','?')
    cid = d.get('id','?')
    slug = d.get('slug','?')
    desc = (d.get('description','') or '')[:200]
    print(f'  Name:    {nm}')
    print(f'  Symbol:  {sy}')
    print(f'  ID:      {cid}')
    print(f'  Slug:    {slug}')
    if desc:
        print(f'  Desc:    {desc}...')
except:
    print(sys.stdin.read()[:2000])
"
}

cmd_status() {
    log_cmd "Checking API connection..."
    if cmc status -o json 2>/dev/null; then
        log_info "API connection OK"
    else
        log_error "API connection failed — check your API key"
        exit 1
    fi
}

# ============ MAIN ============
load_api_key

case "${1:-help}" in
    price)     shift; check_api_key; cmd_price "$@" ;;
    search)    shift; check_api_key; cmd_search "$@" ;;
    trending)  shift; check_api_key; cmd_trending ;;
    markets)   shift; check_api_key; cmd_markets "${1:-10}" ;;
    history)   shift; check_api_key; cmd_history "$@" ;;
    news)      shift; check_api_key; cmd_news ;;
    gainers)   shift; check_api_key; cmd_gainers ;;
    resolve)   shift; check_api_key; cmd_resolve "$@" ;;
    status)    shift; cmd_status ;;
    help|--help|-h|*) usage ;;
esac
