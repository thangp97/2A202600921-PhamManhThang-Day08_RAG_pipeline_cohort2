"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    crawl4ai-setup   # cài playwright browser
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# 5 bài báo về nghệ sĩ Việt liên quan ma tuý
ARTICLE_URLS = [
    {
        "url": "https://vnexpress.net/nguoi-mau-andrea-aybar-va-ca-si-chi-dan-bi-bat-4814295.html",
        "artist": "Chi Dan",
    },
    {
        "url": "https://tuoitre.vn/bat-nguoi-mau-an-tay-ca-si-chi-dan-co-tien-truc-phuong-do-lien-quan-ma-tuy-20241114114826655.htm",
        "artist": "An Tay",
    },
    {
        "url": "https://vnexpress.net/rapper-binh-gold-tiep-tuc-duong-tinh-voi-ma-tuy-lai-cuop-taxi-4919259.html",
        "artist": "Binh Gold",
    },
    {
        "url": "https://vnexpress.net/ca-si-miu-le-bi-bat-qua-tang-dung-ma-tuy-o-bai-bien-5072657.html",
        "artist": "Miu Le",
    },
    {
        "url": "https://vnexpress.net/ca-si-chau-viet-cuong-nhan-13-nam-tu-vi-nhet-toi-hai-chet-co-gai-3891028.html",
        "artist": "Cuong Toi",
    },
]


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Thu muc san sang: {DATA_DIR}")


# ── Crawl4AI (primary) ──────────────────────────────────────────────────────

async def crawl_article_crawl4ai(url: str) -> dict | None:
    """Crawl bài báo bằng Crawl4AI (dùng headless browser, hỗ trợ JS)."""
    try:
        from crawl4ai import AsyncWebCrawler
    except ImportError:
        return None

    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=url)
            if not result.success:
                return None
            return {
                "url": url,
                "title": result.metadata.get("title", "Unknown"),
                "date_crawled": datetime.now().isoformat(),
                "source": "crawl4ai",
                "content_markdown": result.markdown or "",
            }
    except Exception as e:
        print(f"  [crawl4ai] loi: {e}")
        return None


# ── Requests fallback ───────────────────────────────────────────────────────

def crawl_article_requests(url: str) -> dict | None:
    """Fallback: crawl bằng requests + parse HTML thủ công."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "vi-VN,vi;q=0.9",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code != 200:
            print(f"  [requests] HTTP {resp.status_code}")
            return None

        html = resp.text

        # Trích title từ <title> tag
        title = "Unknown"
        if "<title>" in html:
            start = html.index("<title>") + 7
            end = html.index("</title>", start)
            title = html[start:end].strip()

        # Trích nội dung text thô (loại bỏ HTML tags đơn giản)
        import re
        # Xoá script/style
        clean = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ", html, flags=re.DOTALL)
        # Xoá còn lại các tags
        clean = re.sub(r"<[^>]+>", " ", clean)
        # Chuẩn hoá khoảng trắng
        clean = re.sub(r"\s+", " ", clean).strip()
        # Giữ 5000 ký tự đầu tiên
        content = clean[:5000]

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "source": "requests",
            "content_markdown": content,
        }
    except Exception as e:
        print(f"  [requests] loi: {e}")
        return None


# ── Main ────────────────────────────────────────────────────────────────────

async def crawl_article(entry: dict) -> dict | None:
    """Thử crawl4ai trước, fallback sang requests nếu thất bại."""
    url = entry["url"]
    artist = entry["artist"]

    print(f"\n-> [{artist}] {url}")

    # Thử crawl4ai
    result = await crawl_article_crawl4ai(url)
    if result:
        result["artist"] = artist
        print(f"  [OK] crawl4ai - title: {result['title'][:60]}")
        return result

    # Fallback requests
    print(f"  -> fallback requests...")
    result = crawl_article_requests(url)
    if result:
        result["artist"] = artist
        print(f"  [OK] requests - title: {result['title'][:60]}")
        return result

    print(f"  [FAIL] Khong the crawl: {url}")
    return None


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()
    success = 0

    for i, entry in enumerate(ARTICLE_URLS, 1):
        print(f"\n[{i}/{len(ARTICLE_URLS)}]", end="")
        article = await crawl_article(entry)
        if article is None:
            continue

        filename = f"article_{i:02d}_{entry['artist'].lower().replace(' ', '_')}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved: {filepath.name}")
        success += 1

    print(f"\n{'='*50}")
    print(f"Ket qua: {success}/{len(ARTICLE_URLS)} bai bao da crawl thanh cong")
    if success >= 5:
        print("[PASS] Du 5 bai bao theo yeu cau!")
    else:
        print(f"[WARN] Can them {5 - success} bai bao nua!")


def list_collected_files() -> list[Path]:
    """Liệt kê các file đã crawl trong data/landing/news/."""
    if not DATA_DIR.exists():
        return []
    return [f for f in DATA_DIR.iterdir() if f.suffix == ".json"]


if __name__ == "__main__":
    asyncio.run(crawl_all())

    files = list_collected_files()
    print(f"\nCac file trong {DATA_DIR}:")
    for f in files:
        print(f"  {f.name} ({f.stat().st_size:,} bytes)")
