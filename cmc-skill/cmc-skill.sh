#!/usr/bin/env bash
# ============================================================
# CMC Browser Skill — Unified wrapper
# Combines browser scraping (no API key needed) + CMC CLI
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRAPE="$SCRIPT_DIR/cmc-browser/cmc-scrape.js"
CLI_SKILL="$SCRIPT_DIR/cmc-api/cmc-skill.sh"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log() { echo -e "${GREEN}[+]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[-]${NC} $1"; }

usage() {
    echo -e "${GREEN}CMC Skill${NC} — CoinMarketCap Data (No API Key Required)"
    echo ""
    echo "Usage: $0 <command> [args]"
    echo ""
    echo "Commands:"
    echo "  price    <slug>     Get coin price (browser scrape)"
    echo "  trending            Top coins by market cap"
    echo "  gainers             Top gainers & losers"
    echo "  news                Latest crypto news"
    echo "  profile  <slug>     Coin profile details"
    echo "  help                Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 price bitcoin"
    echo "  $0 price ravedao"
    echo "  $0 price ethereum"
    echo "  $0 trending"
    echo "  $0 gainers"
    echo "  $0 news"
    echo "  $0 profile bitcoin"
    echo ""
    echo "Uses: Headless browser scraping (no API key needed)"
    echo "Cache: 1 minute TTL stored in ~/.qwen/skills/cmc-browser/cache/"
}

case "${1:-help}" in
    price)
        [ -z "${2:-}" ] && { err "Usage: $0 price <slug>"; exit 1; }
        node "$SCRAPE" price "$2"
        ;;
    trending)
        node "$SCRAPE" trending
        ;;
    gainers)
        node "$SCRAPE" gainers
        ;;
    news)
        node "$SCRAPE" news
        ;;
    profile)
        [ -z "${2:-}" ] && { err "Usage: $0 profile <slug>"; exit 1; }
        node "$SCRAPE" profile "$2"
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        err "Unknown command: $1"
        usage
        exit 1
        ;;
esac
