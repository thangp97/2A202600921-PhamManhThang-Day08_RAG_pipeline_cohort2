"""
Task 7 — Reranking Module.

Method chính: Cross-encoder qua Jina Reranker v2 API (multilingual).
Bonus: RRF và MMR cũng được implement.

Cơ chế cross-encoder:
    - Bi-encoder (Task 5): encode query và document RIÊNG LẼ → nhanh nhưng kém chính xác
    - Cross-encoder: encode cặp (query, document) CÙNG LÚC → mô hình thấy tương tác
      giữa từng token → chính xác hơn nhiều nhưng chậm hơn (O(n) API calls)
    - Dùng làm reranker: retrieval nhanh trước (top-20), cross-encoder chọn top-5 cuối
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")


# =============================================================================
# Cross-encoder (Jina Reranker v2 — method chính)
# =============================================================================

def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates dùng Jina Reranker v2 (multilingual cross-encoder).

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    import requests
    from dotenv import load_dotenv
    load_dotenv()

    jina_api_key = os.environ.get("JINA_API_KEY", "").strip()
    if not jina_api_key:
        raise ValueError("Thiếu JINA_API_KEY trong .env")

    if not candidates:
        return []

    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={
            "Authorization": f"Bearer {jina_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": [c["content"] for c in candidates],
            "top_n": min(top_k, len(candidates)),
        },
        timeout=30,
    )
    response.raise_for_status()

    reranked = response.json()["results"]
    return [
        {**candidates[r["index"]], "score": r["relevance_score"]}
        for r in reranked
    ]


# =============================================================================
# RRF — Reciprocal Rank Fusion
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))
    k=60 là hằng số từ paper Cormack et al. 2009 — giảm ảnh hưởng của rank 1
    so với rank 2, tránh một ranker độc quyền quyết định kết quả.

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60)
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)

    return results


# =============================================================================
# MMR — Maximal Marginal Relevance
# =============================================================================

def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR(d) = λ * sim(query, d) - (1-λ) * max_s∈S sim(d, s)
    λ=0.7: ưu tiên relevance hơn diversity (0.0=diversity, 1.0=relevance thuần)

    Args:
        query_embedding: Vector embedding của query (đã normalize)
        candidates: List of {'content', 'score', 'embedding', 'metadata'}
        top_k: Số lượng kết quả
        lambda_param: Trade-off relevance vs diversity
    """
    import numpy as np

    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    selected_indices: list[int] = []
    remaining_indices = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining_indices:
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])

            max_sim_to_selected = 0.0
            for sel_idx in selected_indices:
                sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    return [
        {**candidates[i], "score": float(cosine_sim(query_embedding, candidates[i]["embedding"]))}
        for i in selected_indices
    ]


# =============================================================================
# Unified interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: "cross_encoder" | "rrf" | "mmr"
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "rrf":
        raise ValueError("RRF cần nhiều ranked lists — gọi rerank_rrf(ranked_lists, top_k) trực tiếp")
    elif method == "mmr":
        raise ValueError("MMR cần query_embedding — gọi rerank_mmr(query_embedding, candidates, top_k) trực tiếp")
    else:
        raise ValueError(f"Unknown method: {method}. Chọn: cross_encoder | rrf | mmr")


# =============================================================================
# Demo: kết hợp Task 5 + Task 6 + reranking
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search, build_bm25_index

    print("Building BM25 index...")
    build_bm25_index()

    queries = [
        "hình phạt cho tội tàng trữ ma tuý",
        "cai nghiện tự nguyện tại gia đình cộng đồng",
        "Bình Gold bị bắt vì ma túy",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print("=" * 60)

        # Lấy top-10 từ mỗi ranker
        sem_results = semantic_search(q, top_k=10)
        lex_results = lexical_search(q, top_k=10)

        # Gộp unique candidates (ưu tiên semantic, bổ sung từ lexical)
        seen = set()
        candidates = []
        for r in sem_results + lex_results:
            key = r["content"][:80]
            if key not in seen:
                seen.add(key)
                candidates.append(r)

        print(f"  Candidates: {len(sem_results)} semantic + {len(lex_results)} lexical = {len(candidates)} unique")

        # Cross-encoder rerank top-5
        reranked = rerank_cross_encoder(q, candidates, top_k=5)
        print(f"  After rerank (cross-encoder):")
        for i, r in enumerate(reranked, 1):
            print(f"  [{i}] score={r['score']:.4f} | {r['metadata'].get('doc_type','')} | {r['metadata'].get('source','')}")
            print(f"       {r['content'][:100]}...")

        # RRF comparison
        rrf_results = rerank_rrf([sem_results, lex_results], top_k=5)
        print(f"\n  RRF top-3 (để so sánh):")
        for i, r in enumerate(rrf_results[:3], 1):
            print(f"  [{i}] rrf={r['score']:.5f} | {r['metadata'].get('source','')}")
