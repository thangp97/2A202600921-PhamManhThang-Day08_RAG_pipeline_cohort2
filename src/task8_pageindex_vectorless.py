"""
Task 8 — PageIndex Vectorless RAG.

PageIndex hiểu cấu trúc document (heading, section, table) thay vì dùng embedding.
Không cần chunking, không cần vector store. Chỉ hỗ trợ PDF.

Dùng làm fallback khi hybrid search (Task 5+6+7) trả về score thấp.

Workflow:
    1. submit_document(pdf_path) → doc_id  (upload 1 lần, lưu cache)
    2. submit_query(doc_id, query) → retrieval_id
    3. get_retrieval(retrieval_id) → kết quả (poll đến khi xong)
"""

import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
PDF_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DOC_IDS_CACHE = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"

# Fallback threshold — dùng PageIndex khi top-1 score < ngưỡng này
FALLBACK_SCORE_THRESHOLD = 0.3


def _get_client():
    from pageindex.client import PageIndexClient
    if not PAGEINDEX_API_KEY:
        raise ValueError("Thiếu PAGEINDEX_API_KEY trong .env")
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def upload_documents() -> dict:
    """
    Upload toàn bộ PDF documents lên PageIndex.
    Lưu doc_ids vào cache để tránh upload lại.

    Returns:
        dict filename → doc_id
    """
    if DOC_IDS_CACHE.exists():
        cached = json.loads(DOC_IDS_CACHE.read_text(encoding="utf-8"))
        print(f"  Loaded {len(cached)} doc_ids từ cache")
        return cached

    client = _get_client()
    doc_ids = {}

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"  Uploading {len(pdf_files)} PDF files...")

    for pdf_file in pdf_files:
        print(f"  Uploading: {pdf_file.name}...")
        result = client.submit_document(file_path=str(pdf_file))
        doc_id = result["doc_id"]
        doc_ids[pdf_file.name] = doc_id
        print(f"    doc_id: {doc_id}")

    # Chờ tất cả documents xử lý xong
    print("\n  Waiting for documents to be processed...")
    for filename, doc_id in doc_ids.items():
        for _ in range(30):  # tối đa 5 phút
            if client.is_retrieval_ready(doc_id):
                print(f"    {filename}: ready")
                break
            time.sleep(10)
        else:
            print(f"    {filename}: timeout — tiếp tục anyway")

    DOC_IDS_CACHE.write_text(
        json.dumps(doc_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Saved doc_ids to {DOC_IDS_CACHE.name}")
    return doc_ids


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content', 'score', 'metadata', 'source': 'pageindex'}
        Sorted by score descending.
    """
    if not DOC_IDS_CACHE.exists():
        raise RuntimeError("Chưa upload documents. Chạy upload_documents() trước.")

    doc_ids = json.loads(DOC_IDS_CACHE.read_text(encoding="utf-8"))
    client = _get_client()

    # Submit query cho tất cả docs (thinking=True cho kết quả tốt hơn)
    retrieval_ids = {}
    for filename, doc_id in doc_ids.items():
        try:
            result = client.submit_query(doc_id=doc_id, query=query, thinking=True)
            retrieval_ids[filename] = result["retrieval_id"]
        except Exception as e:
            print(f"  [PageIndex] submit_query failed for {filename}: {e}")
            return []

    # Poll kết quả từng doc
    all_results = []
    for filename, retrieval_id in retrieval_ids.items():
        for _ in range(20):  # tối đa ~40 giây
            result = client.get_retrieval(retrieval_id)
            status = result.get("status", "")

            if status == "completed":
                # Cấu trúc: retrieved_nodes[].relevant_contents[][].relevant_content
                nodes = result.get("retrieved_nodes", [])
                for rank, node in enumerate(nodes):
                    title = node.get("title", "")
                    for content_group in node.get("relevant_contents", []):
                        for item in content_group:
                            content = item.get("relevant_content", "")
                            if content:
                                all_results.append({
                                    "content": content,
                                    # PageIndex không trả score → dùng rank nghịch đảo
                                    "score": 1.0 / (1 + rank),
                                    "metadata": {
                                        "source": filename,
                                        "doc_type": "legal",
                                        "section": title,
                                        "section_title": item.get("section_title", title),
                                    },
                                    "source": "pageindex",
                                })
                break
            elif status == "failed":
                print(f"  Retrieval failed for {filename}")
                break
            time.sleep(3)

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]


def search_with_fallback(
    query: str,
    primary_results: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Dùng PageIndex làm fallback nếu primary results có score thấp.

    Args:
        query: Câu truy vấn
        primary_results: Kết quả từ hybrid search (Task 5+6+7)
        top_k: Số lượng kết quả

    Returns:
        primary_results nếu đủ tốt, ngược lại fallback sang PageIndex
    """
    if primary_results and primary_results[0]["score"] >= FALLBACK_SCORE_THRESHOLD:
        return primary_results[:top_k]

    print(f"  [Fallback] Primary score={primary_results[0]['score']:.3f} < {FALLBACK_SCORE_THRESHOLD} → dùng PageIndex")
    return pageindex_search(query, top_k=top_k)


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("Hãy set PAGEINDEX_API_KEY trong file .env")
        sys.exit(1)

    print("=" * 50)
    print("Task 8: PageIndex Vectorless RAG")
    print("=" * 50)

    print("\n[1/2] Uploading documents...")
    doc_ids = upload_documents()
    print(f"  Documents ready: {list(doc_ids.keys())}")

    print("\n[2/2] Test queries:")
    queries = [
        "hình phạt cho tội tàng trữ ma tuý",
        "cai nghiện tự nguyện tại gia đình",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        results = pageindex_search(q, top_k=3)
        if not results:
            print("  (không có kết quả)")
        for i, r in enumerate(results, 1):
            print(f"  [{i}] score={r['score']:.4f} | {r['metadata']['source']} | {r['metadata'].get('section_title','')}")
            print(f"       {r['content'][:150]}...")
