"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả bằng RRF
    3. Rerank bằng cross-encoder
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from task5_semantic_search import semantic_search
from task6_lexical_search import lexical_search, build_bm25_index
from task7_reranking import rerank_cross_encoder, rerank_rrf
from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 15
RETRIEVAL_POOL = 20     # Số candidates lấy từ mỗi ranker trước khi merge
MAX_CHUNKS_PER_SOURCE = 2  # Giới hạn chunks per document để tăng diversity


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank (cross-encoder) → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Song song lấy kết quả từ semantic + lexical
    dense_results  = semantic_search(query, top_k=RETRIEVAL_POOL)
    sparse_results = lexical_search(query, top_k=RETRIEVAL_POOL)

    # Step 2: Merge bằng RRF (gộp unique candidates)
    merged = rerank_rrf([dense_results, sparse_results], top_k=RETRIEVAL_POOL * 2)
    for item in merged:
        item["source"] = "hybrid"

    # Giới hạn max chunks per source để tránh 1 document chiếm hết context
    seen_sources: dict[str, int] = {}
    diverse_merged = []
    for item in merged:
        src = item.get("metadata", {}).get("source", "")
        if seen_sources.get(src, 0) < MAX_CHUNKS_PER_SOURCE:
            diverse_merged.append(item)
            seen_sources[src] = seen_sources.get(src, 0) + 1
    merged = diverse_merged

    if not merged:
        print("  [!] Không có kết quả từ hybrid search → fallback PageIndex")
        return _pageindex_fallback(query, top_k)

    # Step 3: Rerank
    if use_reranking:
        final_results = rerank_cross_encoder(query, merged, top_k=top_k)
    else:
        final_results = merged[:top_k]

    # Step 4: Kiểm tra threshold → fallback
    best_score = final_results[0]["score"] if final_results else 0.0
    if best_score < score_threshold:
        print(f"  [Fallback] Hybrid score={best_score:.3f} < {score_threshold} → PageIndex")
        fallback = _pageindex_fallback(query, top_k)
        if fallback:
            return fallback
        # PageIndex không khả dụng → trả hybrid results
        return final_results

    return final_results


def _pageindex_fallback(query: str, top_k: int) -> list[dict]:
    """Gọi PageIndex và đánh dấu source='pageindex'."""
    results = pageindex_search(query, top_k=top_k)
    for r in results:
        r["source"] = "pageindex"
    return results


if __name__ == "__main__":
    print("Building BM25 index...")
    build_bm25_index()

    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['metadata'].get('doc_type','')} | {r['metadata'].get('source','')}")
            print(f"     {r['content'][:100]}...")
