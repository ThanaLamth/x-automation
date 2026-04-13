const puppeteer = require('puppeteer');
const Parser = require('rss-parser');
const parser = new Parser();

// ==================== CONFIG ====================
const RSS_FEED_URL = "YOUR_RSS_FEED_URL";    // ← Điền RSS feed URL
const X_USERNAME = "your@email.com";         // ← Email/SDT đăng nhập X
const X_PASSWORD = "your_password";          // ← Mật khẩu đăng nhập X

// Proxy config (tùy chọn, bỏ qua nếu không dùng)
const PROXY = {
    enabled: false,
    server: "http://proxy_ip:port",          // ← Proxy server
    username: "proxy_user",                   // ← Proxy user (nếu có)
    password: "proxy_pass"                    // ← Proxy pass (nếu có)
};
// ================================================

async function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms + Math.random() * 1000));
}

async function loginToX(page) {
    console.log('[>] Đang login vào X...');
    await page.goto('https://x.com/i/flow/login', { waitUntil: 'networkidle2' });
    await delay(3000);

    // Step 1: Nhập username/email
    const usernameInput = await page.$('input[autocomplete="username"]');
    if (usernameInput) {
        await usernameInput.type(X_USERNAME, { delay: 50 });
        await delay(1000);
        await page.keyboard.press('Enter');
        await delay(3000);
    }

    // Step 2: Nhập password
    const passwordInput = await page.$('input[type="password"]');
    if (passwordInput) {
        await passwordInput.type(X_PASSWORD, { delay: 50 });
        await delay(1000);
        await page.keyboard.press('Enter');
        await delay(5000);
    }

    // Kiểm tra login thành công
    const currentUrl = page.url();
    if (currentUrl.includes('/login') || currentUrl.includes('/welcome')) {
        console.log('[!] Login thất bại, kiểm tra lại credentials!');
        return false;
    }
    console.log('[+] Login thành công!');
    return true;
}

async function scrollToExplore(page, duration = 5) {
    console.log(`[>] Đang scroll explore trong ${duration} phút...`);
    const endTime = Date.now() + duration * 60 * 1000;

    while (Date.now() < endTime) {
        await page.evaluate(() => {
            window.scrollBy(0, window.innerHeight);
        });
        await delay(2000 + Math.random() * 3000);

        // Thỉnh thoảng like bài viết
        if (Math.random() < 0.3) {
            const likeBtn = await page.$('[data-testid="like"]');
            if (likeBtn) {
                await likeBtn.click();
                await delay(1000);
            }
        }
    }
    console.log('[+] Scroll explore xong!');
}

async function postFromRSS(page) {
    console.log('[>] Đang lấy bài từ RSS feed...');
    const feed = await parser.parseURL(RSS_FEED_URL);
    console.log(`[+] Tìm thấy ${feed.items.length} bài trong feed`);

    for (const item of feed.items) {
        console.log(`[>] Đang đăng: ${item.title}`);

        // Mở compose
        await page.goto('https://x.com/compose/post', { waitUntil: 'networkidle2' });
        await delay(2000);

        // Soạn tweet (tối đa 280 ký tự)
        let tweetText = `${item.title}\n\n${item.link}`;
        if (tweetText.length > 280) {
            tweetText = tweetText.substring(0, 277) + '...';
        }

        // Gõ nội dung
        const textarea = await page.$('[data-testid="tweetTextarea_0"]');
        if (textarea) {
            await textarea.click();
            await page.keyboard.type(tweetText, { delay: 30 });
            await delay(2000);

            // Nhấn Post
            const postBtn = await page.$('[data-testid="tweetButtonInline"]');
            if (postBtn) {
                await postBtn.click();
                console.log(`[+] Đã đăng: ${item.title}`);
                await delay(5000);
            }
        }

        // Delay giữa các bài (5-15 phút)
        const randomDelay = 5 + Math.random() * 10;
        console.log(`[*] Chờ ${randomDelay.toFixed(1)} phút trước bài tiếp theo...`);
        await delay(randomDelay * 60 * 1000);
    }

    console.log('[+] Đăng bài xong!');
}

(async () => {
    console.log('[*] Khởi động GenLogin...');
    const genlogin = new Genlogin(API_KEY);

    // Lấy profile đầu tiên
    const profile = (await genlogin.getProfiles(0, 1)).profiles[0];
    console.log(`[+] Dùng profile: ${profile.id}`);

    // Chạy profile
    const { wsEndpoint } = await genlogin.runProfile(profile.id);
    console.log(`[+] wsEndpoint: ${wsEndpoint}`);

    // Connect Puppeteer
    const browser = await puppeteer.connect({
        browserWSEndpoint: wsEndpoint,
        ignoreHTTPSErrors: true,
        defaultViewport: false
    });

    const page = (await browser.pages())[0];

    // Step 1: Login
    await loginToX(page);

    // Step 2: Scroll explore
    await scrollToExplore(page, 5); // 5 phút

    // Step 3: Đăng bài từ RSS
    await postFromRSS(page);

    // Cleanup
    await browser.disconnect();
    await genlogin.stopProfile(profile.id);
    console.log('[*] Hoàn tất!');
})();
