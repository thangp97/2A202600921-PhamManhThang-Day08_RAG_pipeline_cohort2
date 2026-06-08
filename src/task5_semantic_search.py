"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import os
import sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"
sys.stdout.reconfigure(encoding="utf-8")

# Reuse config từ Task 4
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
WEAVIATE_COLLECTION = "DrugLawDocs"

# Singleton để tránh load lại model/client mỗi lần gọi
_model = None
_client = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_client():
    global _client
    if _client is None or not _client.is_connected():
        import weaviate
        from weaviate.auth import Auth
        from dotenv import load_dotenv
        load_dotenv()

        url = os.environ.get("WEAVIATE_URL", "").strip()
        key = os.environ.get("WEAVIATE_API_KEY", "").strip()
        if not url.startswith("http"):
            url = "https://" + url

        _client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(key),
        )
    return _client


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    from weaviate.classes.query import MetadataQuery

    # Bước 1: Embed query — E5 cần prefix "query: " cho query text
    model = _get_model()
    query_embedding = model.encode(
        f"query: {query}",
        normalize_embeddings=True,
    ).tolist()

    # Bước 2: Query Weaviate với near_vector (cosine similarity)
    client = _get_client()
    collection = client.collections.get(WEAVIATE_COLLECTION)

    results = collection.query.near_vector(
        near_vector=query_embedding,
        limit=top_k,
        return_metadata=MetadataQuery(distance=True),
    )

    # Bước 3: Chuyển kết quả — distance = 1 - cosine_similarity
    output = []
    for obj in results.objects:
        props = obj.properties
        output.append({
            "content": props.get("content", ""),
            "score": 1.0 - (obj.metadata.distance or 0.0),
            "metadata": {
                "source":      props.get("source", ""),
                "source_path": props.get("source_path", ""),
                "doc_type":    props.get("doc_type", ""),
                "heading_1":   props.get("heading_1", ""),
                "heading_2":   props.get("heading_2", ""),
                "chunk_index": props.get("chunk_index", 0),
            },
        })

    # Đảm bảo sorted descending theo score
    output.sort(key=lambda x: x["score"], reverse=True)
    return output


if __name__ == "__main__":
    queries = [
        "hình phạt cho tội tàng trữ ma tuý",
        "cai nghiện ma túy tự nguyện",
        "Bình Gold bị bắt vì ma túy",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print("=" * 60)
        results = semantic_search(q, top_k=5)
        for i, r in enumerate(results, 1):
            print(f"[{i}] score={r['score']:.4f} | {r['metadata']['doc_type']} | {r['metadata']['source']}")
            print(f"     {r['content'][:120]}...")
