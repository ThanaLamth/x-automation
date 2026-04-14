#!/usr/bin/env node
/**
 * CMC Browser Scraper — Access CoinMarketCap like a normal browser
 * No API key needed, no rate limits
 *
 * Usage:
 *   node cmc-scrape.js price <id|slug>
 *   node cmc-scrape.js search <query>
 *   node cmc-scrape.js trending
 *   node cmc-scrape.js gainers
 *   node cmc-scrape.js news
 *   node cmc-scrape.js profile <slug>
 */

const puppeteer = require('puppeteer');
const fs = require('fs');

const BASE = 'https://coinmarketcap.com';
const CACHE_DIR = '/home/qwen/.qwen/skills/cmc-browser/cache';
const CACHE_TTL = 60_000; // 1 minute cache

function log(type, msg) {
    const colors = { '>': '\x1b[34m', '+': '\x1b[32m', '!': '\x1b[33m', '-': '\x1b[31m' };
    console.log(`${colors[type] || ''}[${type}]\x1b[0m ${msg}`);
}

function cacheKey(cmd, args) {
    return `${cmd}_${args.join('_')}`.replace(/[^a-zA-Z0-9_-]/g, '_');
}

function cacheRead(key) {
    try {
        const path = `${CACHE_DIR}/${key}.json`;
        const raw = fs.readFileSync(path, 'utf-8');
        const data = JSON.parse(raw);
        if (Date.now() - data.ts < CACHE_TTL) {
            return data.result;
        }
    } catch {}
    return null;
}

function cacheWrite(key, result) {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
    fs.writeFileSync(`${CACHE_DIR}/${key}.json`, JSON.stringify({ ts: Date.now(), result }));
}

async function scrapePrice(slug) {
    const url = `${BASE}/currencies/${slug}/`;
    log('>', `Navigating to: ${url}`);

    await page.goto(url, { waitUntil: 'networkidle2', timeout: 30000 });
    await sleep(5000); // Wait for JS rendering

    return await page.evaluate(() => {
        const info = {};

        // Method 1: Try __NEXT_DATA__ detailRes
        const nextData = document.querySelector('#__NEXT_DATA__');
        if (nextData) {
            try {
                const data = JSON.parse(nextData.textContent);
                const coin = data?.props?.pageProps?.detailRes;
                if (coin && (coin.name || coin.symbol)) {
                    info.name = coin.name || '?';
                    info.symbol = coin.symbol || '?';
                    const usd = coin.quote?.USD || {};
                    info.price = usd.price ? `$${Number(usd.price).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}` : '?';
                    const ch = usd.percent_change_24h;
                    info.change24h = ch != null ? `${ch >= 0 ? '+' : ''}${ch.toFixed(2)}%` : '?';
                    info.marketCap = usd.market_cap ? `$${formatCompact(usd.market_cap)}` : '?';
                    info.volume24h = usd.volume_24h ? `$${formatCompact(usd.volume_24h)}` : '?';
                    info.rank = coin.cmcRank || '?';
                    info.supply = coin.circulatingSupply ? `${formatCompact(coin.circulatingSupply)} ${coin.symbol}` : '?';
                    info.description = coin.description?.substring(0, 500) || '';
                    return info;
                }
            } catch(e) {}
        }

        // Method 2: Extract from body text
        const bodyText = document.body.innerText;
        if (!bodyText || bodyText.length < 100) {
            return { error: 'Page did not load' };
        }

        // Name from title tag
        const title = document.title;
        const titleMatch = title.match(/^([^,]+) price/);
        info.name = titleMatch ? titleMatch[1].trim() : '?';
        info.symbol = (document.querySelector('meta[name="symbol"]')?.content) || '?';

        // Extract patterns from body text
        // Pattern: "#1\n6M\n$74,419.07\n  \n\n4.82% (24h)"
        const priceMatch = bodyText.match(/#\d+\n[^#\n]*?\n\$?([\d,.]+)\n[\s\S]*?([+-]?[\d.]+%?\s*\(24h\))/);
        if (priceMatch) {
            info.price = `$${priceMatch[1]}`;
            info.change24h = priceMatch[2].trim();
        }

        // Market Cap
        const mcapMatch = bodyText.match(/Market cap\s*\n?\s*\$?([\d,.BTKM]+)/i);
        if (mcapMatch) info.marketCap = `$${mcapMatch[1]}`;

        // Volume
        const volMatch = bodyText.match(/Volume\s*\(24h\)\s*\n?\s*\$?([\d,.BTKM]+)/i);
        if (volMatch) info.volume24h = `$${volMatch[1]}`;

        // Rank
        const rankMatch = bodyText.match(/#(\d+)\n/);
        if (rankMatch) info.rank = `#${rankMatch[1]}`;

        // Supply
        const supplyMatch = bodyText.match(/Circulating supply\s*\n?\s*([\d,.BTKM]+)/i);
        const symMatch = bodyText.match(/(\d+\.?\d*\s*[MKB]?)\s*(BTC|ETH|SOL|BNB|XRP|DOGE|ADA|RAVE|RVN|AVAX|MATIC|DOT|SHIB|LTC|LINK|UNI|ATOM|FIL|NEAR|APT|ARB|OP|TRX|PEPE|WLD|RENDER|TAO|STX|IMX|INJ|TIA|ONDO|PENDLE|SUI|SEI|WIF|FET|RUNE|GRT|ALGO|VET|FLOW|SAND|MANA|AXS|ENJ|GALA|CHZ)/);
        if (supplyMatch && symMatch) {
            info.supply = `${supplyMatch[1]} ${symMatch[2]}`;
        } else if (supplyMatch) {
            info.supply = supplyMatch[1];
        }

        return info;
    });
}

function formatCompact(num) {
    if (num >= 1e12) return (num / 1e12).toFixed(2) + 'T';
    if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
    return Number(num).toLocaleString('en-US');
}

async function scrapeSearch(query) {
    log('>', `Searching: ${query}`);

    // Navigate to search page
    const url = `${BASE}/search/?q=${encodeURIComponent(query)}`;
    await page.goto(url, { waitUntil: 'networkidle2', timeout: 20000 });
    await sleep(3000);

    return await page.evaluate(() => {
        const results = [];

        // Try __NEXT_DATA__ first
        const nextData = document.querySelector('#__NEXT_DATA__');
        if (nextData) {
            try {
                const data = JSON.parse(nextData.textContent);
                const searchRes = data?.props?.pageProps?.searchRes;
                const coins = searchRes?.coins || searchRes?.cryptoCurrencies || [];
                coins.forEach(c => {
                    results.push({
                        name: c.name || c.cryptoName || '?',
                        symbol: c.symbol || c.cryptoSymbol || '?',
                        id: c.id || c.coinId || '?',
                        slug: c.slug || '',
                        rank: c.cmcRank || c.rank || '?'
                    });
                });
            } catch {}
        }

        // Fallback: extract from links
        if (results.length === 0) {
            const links = document.querySelectorAll('a[href*="/currencies/"]');
            const seen = new Set();
            links.forEach(a => {
                const href = a.href;
                if (seen.has(href)) return;
                seen.add(href);
                const text = a.textContent.trim();
                if (text && text.length > 2 && text.length < 40) {
                    results.push({ name: text, symbol: '', id: '?', slug: '', rank: '?' });
                }
            });
        }

        return results.slice(0, 20);
    });
}

async function scrapeTrending() {
    const url = BASE;
    log('>', `Fetching trending from homepage: ${url}`);

    await page.goto(url, { waitUntil: 'networkidle2', timeout: 30000 });
    await sleep(5000); // Wait for JS to render

    return await page.evaluate(() => {
        const coins = [];
        // CMC homepage has a table with all coins
        const rows = document.querySelectorAll('table tbody tr, [class*="TableRow"]');

        rows.forEach((row, i) => {
            if (coins.length >= 30) return;
            const cells = row.querySelectorAll('td, [role="cell"]');
            if (cells.length < 5) return;

            // Extract rank, name, price, change
            const nameEl = cells[2]?.querySelector('a p') || cells[2]?.querySelector('a');
            const priceEl = cells[3]?.querySelector('a') || cells[3];
            const changeEl = cells[4]?.querySelector('span') || cells[4];
            const mcapEl = cells[6] || cells[5];

            if (nameEl) {
                coins.push({
                    rank: cells[0]?.textContent.trim() || (i + 1).toString(),
                    name: nameEl.textContent.trim(),
                    price: priceEl?.textContent.trim() || '?',
                    change24h: changeEl?.textContent.trim() || '?',
                    marketCap: mcapEl?.textContent.trim() || '?'
                });
            }
        });

        return coins.length > 0 ? coins : null;
    });
}

async function scrapeGainersLosers() {
    const url = `${BASE}/gainers-losers/`;
    log('>', `Fetching gainers/losers: ${url}`);

    await page.goto(url, { waitUntil: 'networkidle2', timeout: 30000 });
    await sleep(5000);

    // Extract raw body text
    const bodyText = await page.evaluate(() => document.body.innerText);

    // Parse in Node.js
    const gainers = [];
    const losers = [];

    const topGainersIdx = bodyText.search(/Top Gainers/i);
    const topLosersIdx = bodyText.search(/Top Losers/i);
    const endIdx = bodyText.search(/Categories|Leaderboards/i);

    if (topGainersIdx === -1 || topLosersIdx === -1) return { gainers, losers };

    const gainersSection = bodyText.substring(topGainersIdx, topLosersIdx);
    const losersSection = bodyText.substring(topLosersIdx, endIdx !== -1 ? endIdx : bodyText.length);

    function parseCoins(text) {
        const coins = [];
        // Look for patterns like: rank  name  SYMBOL  $price  change%  $volume
        const re = /(\d+)\s{2,}([A-Za-z0-9 .()\-]+)\s{2,}([A-Z]{2,10})\s{2,}\$?([\d,.]+)\s{1,}([+-]?[\d.]+%)\s{1,}\$?([\d,.]+)/g;
        let m;
        while ((m = re.exec(text)) !== null) {
            coins.push({
                rank: m[1],
                name: m[2].trim(),
                symbol: m[3],
                price: `$${m[4]}`,
                change24h: m[5],
                volume: `$${m[6]}`
            });
        }
        return coins;
    }

    const g = parseCoins(gainersSection);
    gainers.push(...g.sort((a, b) => parseFloat(b.change24h) - parseFloat(a.change24h)).slice(0, 15));

    const l = parseCoins(losersSection);
    losers.push(...l.sort((a, b) => parseFloat(a.change24h) - parseFloat(b.change24h)).slice(0, 15));

    return { gainers, losers };
}

async function scrapeNews() {
    const url = `${BASE}/headline-news/`;
    log('>', `Fetching news: ${url}`);

    await page.goto(url, { waitUntil: 'networkidle2', timeout: 30000 });
    await sleep(3000);

    return await page.evaluate(() => {
        const articles = [];
        const items = document.querySelectorAll('a[href*="/news/"], [class*="NewsCard"], article a');

        items.forEach(item => {
            const titleEl = item.querySelector('h2, h3, h4, [class*="title"], p');
            const title = titleEl ? titleEl.textContent.trim() : item.textContent.trim();
            if (title && title.length > 10) {
                articles.push({
                    title: title.substring(0, 120),
                    url: item.href || '#'
                });
            }
        });

        // Deduplicate
        const seen = new Set();
        return articles.filter(a => {
            if (seen.has(a.title)) return false;
            seen.add(a.title);
            return true;
        }).slice(0, 20);
    });
}

async function scrapeProfile(slug) {
    const url = `${BASE}/currencies/${slug}/`;
    log('>', `Fetching profile: ${url}`);

    await page.goto(url, { waitUntil: 'networkidle2', timeout: 30000 });
    await sleep(4000);

    return await page.evaluate(() => {
        const info = {};

        // Description
        const descEl = document.querySelector('[class*="description"] p, .coin-description p');
        if (descEl) info.description = descEl.textContent.trim().substring(0, 300);

        // Links
        const links = {};
        document.querySelectorAll('a[href*="twitter.com"], a[href*="x.com"]').forEach(a => links.twitter = a.href);
        document.querySelectorAll('a[href*="github.com"]').forEach(a => links.github = a.href);
        document.querySelectorAll('a[href*="reddit.com"]').forEach(a => links.reddit = a.href);
        document.querySelectorAll('a[href*="website"]').forEach(a => links.website = a.href);
        info.links = links;

        // Stats
        const stats = {};
        document.querySelectorAll('[class*="statsItem"], [class*="detail"] [class*="item"]').forEach(item => {
            const label = item.querySelector('[class*="label"], [class*="name"]')?.textContent.trim();
            const value = item.querySelector('[class*="value"]')?.textContent.trim();
            if (label && value) stats[label] = value;
        });
        info.stats = stats;

        return info;
    });
}

function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}

function formatOutput(cmd, data) {
    switch (cmd) {
        case 'price':
            console.log(`\n  ${data.name}`);
            console.log(`  Price:    ${data.price}`);
            console.log(`  24h:      ${data.change24h.replace(/\s*\(24h\)/, '')}`);
            console.log(`  MCap:     ${data.marketCap}`);
            console.log(`  Volume:   ${data.volume24h}`);
            console.log(`  Rank:     ${data.rank.replace(/^#+/, '#')}`);
            console.log(`  Supply:   ${data.supply}`);
            if (data.description) console.log(`\n  ${data.description.substring(0, 200)}...`);
            break;

        case 'search':
            console.log(`\n  Found ${data.length} results:`);
            data.forEach((r, i) => {
                const num = (i + 1).toString().padStart(2);
                console.log(`  ${num}. ${r.name}`);
                console.log(`      ${r.url}`);
            });
            break;

        case 'trending':
            if (data && data.length > 0) {
                console.log(`\n  Top Coins on CoinMarketCap:`);
                console.log(`  ${'Rank'.padStart(5)} ${'Name'.padEnd(25)} ${'Price'.padStart(15)} ${'24h'.padStart(10)} ${'MCap'.padStart(18)}`);
                console.log('  ' + '-'.repeat(80));
                data.forEach(c => {
                    console.log(`  ${c.rank.padStart(5)} ${c.name.padEnd(25)} ${c.price.padStart(15)} ${c.change24h.padStart(10)} ${c.marketCap.padStart(18)}`);
                });
            } else {
                console.log('  Could not load trending data — page structure may have changed');
            }
            break;

        case 'gainers':
            console.log('\n  Top Gainers:');
            data.gainers.forEach(g => {
                console.log(`  + ${g.name.padEnd(25)} ${g.change24h}`);
            });
            console.log('\n  Top Losers:');
            data.losers.forEach(l => {
                console.log(`  - ${l.name.padEnd(25)} ${l.change24h}`);
            });
            break;

        case 'news':
            console.log(`\n  Latest News (${data.length} articles):`);
            data.forEach((a, i) => {
                console.log(`  ${(i + 1).toString().padStart(2)}. ${a.title}`);
                console.log(`      ${a.url}`);
            });
            break;

        case 'profile':
            console.log(`\n  Profile Info:`);
            if (data.description) console.log(`  ${data.description}...`);
            if (data.links) {
                console.log('  Links:');
                Object.entries(data.links).forEach(([k, v]) => console.log(`    ${k}: ${v}`));
            }
            if (data.stats) {
                console.log('  Stats:');
                Object.entries(data.stats).forEach(([k, v]) => console.log(`    ${k}: ${v}`));
            }
            break;
    }
}

// ============ MAIN ============
(async () => {
    const cmd = process.argv[2];
    const args = process.argv.slice(3);

    if (!cmd || cmd === 'help' || cmd === '--help') {
        console.log(`
CMC Browser Scraper — No API key needed

Usage: node cmc-scrape.js <command> [args]

Commands:
  price    <slug>       Get coin price (e.g., bitcoin, ethereum)
  search   <query>      Search coins (e.g., RAVE, BTC)
  trending              Get top coins from homepage
  gainers               Get gainers & losers
  news                  Get latest crypto news
  profile  <slug>       Get coin profile details

Examples:
  node cmc-scrape.js price bitcoin
  node cmc-scrape.js search RAVE
  node cmc-scrape.js trending
  node cmc-scrape.js gainers
  node cmc-scrape.js news
  node cmc-scrape.js profile ethereum
`);
        process.exit(0);
    }

    // Check cache
    const key = cacheKey(cmd, args);
    const cached = cacheRead(key);
    if (cached) {
        log('+', 'Cache hit!');
        formatOutput(cmd, cached);
        process.exit(0);
    }

    log('>', `Launching browser...`);
    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
    });

    global.page = await browser.newPage();
    await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36');
    await page.setViewport({ width: 1440, height: 900 });

    try {
        let result;

        switch (cmd) {
            case 'price':
                if (!args[0]) { log('-', 'Usage: price <slug>'); process.exit(1); }
                result = await scrapePrice(args[0]);
                break;

            case 'search':
                if (!args[0]) { log('-', 'Usage: search <query>'); process.exit(1); }
                result = await scrapeSearch(args[0]);
                break;

            case 'trending':
                result = await scrapeTrending();
                break;

            case 'gainers':
                result = await scrapeGainersLosers();
                break;

            case 'news':
                result = await scrapeNews();
                break;

            case 'profile':
                if (!args[0]) { log('-', 'Usage: profile <slug>'); process.exit(1); }
                result = await scrapeProfile(args[0]);
                break;

            default:
                log('-', `Unknown command: ${cmd}`);
                process.exit(1);
        }

        if (result) {
            formatOutput(cmd, result);
            cacheWrite(key, result);
            log('+', 'Done!');
        } else {
            log('!', 'No data found — page structure may have changed');
        }
    } catch (err) {
        log('-', `Error: ${err.message}`);
        await page.screenshot({ path: '/tmp/cmc-error.png', fullPage: true });
        log('!', 'Screenshot saved to /tmp/cmc-error.png');
    } finally {
        await browser.close();
    }
})();
