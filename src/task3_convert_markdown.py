"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import sys
from pathlib import Path

from markitdown import MarkItDown

sys.stdout.reconfigure(encoding="utf-8")

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

SUPPORTED_LEGAL_EXTS = {".pdf", ".docx", ".doc", ".html"}


def convert_legal_docs():
    """Convert PDF/DOCX/HTML files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()
    success, skipped = 0, 0

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in SUPPORTED_LEGAL_EXTS:
            continue

        output_path = output_dir / f"{filepath.stem}.md"
        print(f"  Converting: {filepath.name}")

        try:
            result = md.convert(str(filepath))
            content = result.text_content.strip()
            if not content:
                print(f"    [WARN] Nội dung rỗng, bỏ qua.")
                skipped += 1
                continue

            # Thêm metadata header
            header = f"# {filepath.stem.replace('-', ' ').title()}\n\n"
            header += f"**Nguồn file:** `{filepath.name}`\n\n---\n\n"
            output_path.write_text(header + content, encoding="utf-8")
            print(f"    [OK] {output_path.name} ({len(content):,} chars)")
            success += 1

        except Exception as e:
            print(f"    [FAIL] {e}")
            skipped += 1

    print(f"  Tổng: {success} thành công, {skipped} bỏ qua")
    return success


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    success, skipped = 0, 0

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue

        print(f"  Converting: {filepath.name}")

        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            title = data.get("title", "Unknown")
            url = data.get("url", "N/A")
            date_crawled = data.get("date_crawled", "N/A")
            artist = data.get("artist", "")
            content = data.get("content_markdown", "").strip()

            if not content:
                print(f"    [WARN] Không có nội dung, bỏ qua.")
                skipped += 1
                continue

            header = f"# {title}\n\n"
            if artist:
                header += f"**Nghệ sĩ:** {artist}\n"
            header += f"**Nguồn:** {url}\n"
            header += f"**Ngày crawl:** {date_crawled}\n\n---\n\n"

            output_path.write_text(header + content, encoding="utf-8")
            print(f"    [OK] {output_path.name} ({len(content):,} chars)")
            success += 1

        except Exception as e:
            print(f"    [FAIL] {e}")
            skipped += 1

    print(f"  Tổng: {success} thành công, {skipped} bỏ qua")
    return success


def list_standardized_files() -> list[Path]:
    """Liệt kê toàn bộ file markdown đã convert."""
    if not OUTPUT_DIR.exists():
        return []
    return list(OUTPUT_DIR.rglob("*.md"))


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    legal_count = convert_legal_docs()

    print("\n--- News Articles ---")
    news_count = convert_news_articles()

    total = legal_count + news_count
    print(f"\n{'='*50}")
    print(f"Hoàn thành: {total} files → {OUTPUT_DIR}")

    files = list_standardized_files()
    print(f"\nCác file đã tạo:")
    for f in sorted(files):
        rel = f.relative_to(OUTPUT_DIR)
        print(f"  {rel}  ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    convert_all()
