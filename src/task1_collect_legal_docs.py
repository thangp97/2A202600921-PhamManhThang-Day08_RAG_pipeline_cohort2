"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản pháp luật (PDF/DOCX) từ các nguồn chính thống.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, có năm ban hành.

Gợi ý nguồn:
    - https://thuvienphapluat.vn
    - https://vanban.chinhphu.vn
    - https://luatvietnam.vn

Gợi ý văn bản:
    - Luật Phòng, chống ma tuý 2021 (73/2021/QH15)
    - Nghị định 105/2021/NĐ-CP
    - Bộ luật Hình sự 2015 (sửa đổi 2017) - Chương XX
    - Nghị định 57/2022/NĐ-CP về danh mục chất ma tuý
"""

import json
import sys
import time
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# Danh sách văn bản cần tải — dùng vbpl.vn (Văn bản pháp luật Quốc gia)
# ItemID lấy từ URL trang chi tiết trên vbpl.vn
LEGAL_DOCS = [
    {
        "filename": "luat-phong-chong-ma-tuy-2021.pdf",
        "title": "Luật Phòng, chống ma tuý 2021 (73/2021/QH15)",
        "item_id": "152571",
        "source_url": "https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID=152571",
    },
    {
        "filename": "nghi-dinh-105-2021-huong-dan-luat-phong-chong-ma-tuy.pdf",
        "title": "Nghị định 105/2021/NĐ-CP hướng dẫn thi hành Luật Phòng chống ma tuý",
        "item_id": "154938",
        "source_url": "https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID=154938",
    },
    {
        "filename": "nghi-dinh-57-2022-danh-muc-chat-ma-tuy.pdf",
        "title": "Nghị định 57/2022/NĐ-CP về danh mục chất ma tuý và tiền chất",
        "item_id": "161120",
        "source_url": "https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID=161120",
    },
    {
        "filename": "bo-luat-hinh-su-2015-chuong-XX-toi-pham-ma-tuy.pdf",
        "title": "Bộ luật Hình sự 2015 (sửa đổi 2017) - Chương XX: Các tội phạm về ma tuý",
        "item_id": "112168",
        "source_url": "https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID=112168",
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
}


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def download_pdf_vbpl(item_id: str, dest_path: Path) -> bool:
    """Tải PDF từ vbpl.vn qua endpoint FileDownload."""
    # vbpl.vn cho phép tải file qua endpoint này
    download_url = f"https://vbpl.vn/FileDownload.ashx?vbid={item_id}&type=2"
    try:
        resp = requests.get(download_url, headers=HEADERS, timeout=30, allow_redirects=True)
        content_type = resp.headers.get("Content-Type", "")
        if resp.status_code == 200 and ("pdf" in content_type or len(resp.content) > 10_000):
            dest_path.write_bytes(resp.content)
            print(f"  ✓ Tải PDF thành công: {dest_path.name} ({len(resp.content):,} bytes)")
            return True
        print(f"  ✗ Không nhận được PDF (status={resp.status_code}, type={content_type})")
    except requests.RequestException as e:
        print(f"  ✗ Lỗi khi tải PDF: {e}")
    return False


def download_html_fallback(doc: dict, dest_path: Path) -> bool:
    """Fallback: lưu nội dung HTML của trang văn bản khi không tải được PDF."""
    html_path = dest_path.with_suffix(".html")
    try:
        resp = requests.get(doc["source_url"], headers=HEADERS, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 1_000:
            html_path.write_bytes(resp.content)
            print(f"  ↳ Fallback HTML: {html_path.name} ({len(resp.content):,} bytes)")
            return True
        print(f"  ✗ Fallback HTML thất bại (status={resp.status_code})")
    except requests.RequestException as e:
        print(f"  ✗ Lỗi fallback HTML: {e}")
    return False


def save_metadata(doc: dict, success: bool, file_format: str):
    """Lưu metadata của văn bản vào file JSON."""
    meta_path = DATA_DIR / f"{Path(doc['filename']).stem}.json"
    metadata = {
        "title": doc["title"],
        "filename": doc["filename"],
        "source_url": doc["source_url"],
        "item_id": doc["item_id"],
        "format": file_format,
        "downloaded": success,
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_legal_docs():
    """Tải toàn bộ văn bản pháp luật trong danh sách LEGAL_DOCS."""
    setup_directory()
    results = {"success": 0, "fallback": 0, "failed": 0}

    for doc in LEGAL_DOCS:
        dest = DATA_DIR / doc["filename"]
        print(f"\n→ {doc['title']}")

        if dest.exists():
            print(f"  ✓ Đã tồn tại, bỏ qua: {dest.name}")
            results["success"] += 1
            continue

        # Thử tải PDF trực tiếp
        if download_pdf_vbpl(doc["item_id"], dest):
            save_metadata(doc, success=True, file_format="pdf")
            results["success"] += 1
        # Fallback: lưu HTML
        elif download_html_fallback(doc, dest):
            save_metadata(doc, success=True, file_format="html")
            results["fallback"] += 1
        else:
            save_metadata(doc, success=False, file_format="none")
            results["failed"] += 1

        time.sleep(1)  # Tránh bị block

    print(f"\n{'='*50}")
    print(f"Kết quả: {results['success']} PDF | {results['fallback']} HTML fallback | {results['failed']} thất bại")
    total = results["success"] + results["fallback"]
    print(f"Tổng file hợp lệ: {total}/3 (yêu cầu tối thiểu 3)")
    return results


def list_collected_files() -> list[Path]:
    """Liệt kê các file đã tải trong data/landing/legal/."""
    if not DATA_DIR.exists():
        return []
    return [f for f in DATA_DIR.iterdir() if f.suffix in {".pdf", ".docx", ".html"}]


if __name__ == "__main__":
    collect_legal_docs()

    files = list_collected_files()
    print(f"\nCác file trong {DATA_DIR}:")
    for f in files:
        print(f"  {f.name} ({f.stat().st_size:,} bytes)")
