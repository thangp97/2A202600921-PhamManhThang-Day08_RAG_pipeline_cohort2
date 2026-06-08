"""
Task 6 — Lexical Search Module (BM25).

Dùng BM25Okapi (rank-bm25) với corpus load từ data/standardized/.

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)

Cài đặt:
    pip install rank-bm25
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Singleton — chỉ build index 1 lần
_corpus: list[dict] = []
_bm25 = None


def _load_and_chunk() -> list[dict]:
    """Load markdown files và chunk giống Task 4."""
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    headers_to_split_on = [
        ("#", "heading_1"),
        ("##", "heading_2"),
        ("###", "heading_3"),
    ]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        base_meta = {
            "source": md_file.name,
            "source_path": str(md_file.relative_to(STANDARDIZED_DIR)),
            "doc_type": doc_type,
        }

        for split in header_splitter.split_text(content):
            text = split.page_content if hasattr(split, "page_content") else split
            split_meta = split.metadata if hasattr(split, "metadata") else {}

            if len(text) <= CHUNK_SIZE:
                if text.strip():
                    chunks.append({
                        "content": text.strip(),
                        "metadata": {**base_meta, **split_meta, "chunk_index": len(chunks)},
                    })
            else:
                for i, sub_text in enumerate(recursive_splitter.split_text(text)):
                    if sub_text.strip():
                        chunks.append({
                            "content": sub_text.strip(),
                            "metadata": {**base_meta, **split_meta, "chunk_index": len(chunks), "sub_chunk": i},
                        })

    return chunks


def _tokenize(text: str) -> list[str]:
    """Tokenize đơn giản: lowercase + split theo khoảng trắng."""
    return text.lower().split()


def build_bm25_index(corpus: list[dict] = None):
    """
    Xây dựng BM25 index từ corpus.
    Nếu corpus=None, tự load từ data/standardized/.
    """
    global _corpus, _bm25
    from rank_bm25 import BM25Okapi

    _corpus = corpus if corpus is not None else _load_and_chunk()
    tokenized_corpus = [_tokenize(doc["content"]) for doc in _corpus]
    _bm25 = BM25Okapi(tokenized_corpus)
    return _bm25


def _ensure_index():
    """Lazy-build index nếu chưa có."""
    if _bm25 is None:
        build_bm25_index()


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    import numpy as np

    _ensure_index()

    tokenized_query = _tokenize(query)
    scores = _bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": _corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": _corpus[idx]["metadata"],
            })

    return results


if __name__ == "__main__":
    print("Building BM25 index...")
    build_bm25_index()
    print(f"Index built: {len(_corpus)} chunks\n")

    queries = [
        "Điều 248 tàng trữ trái phép chất ma tuý",
        "cai nghiện tự nguyện tại gia đình",
        "Bình Gold dương tính ma túy",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print("=" * 60)
        results = lexical_search(q, top_k=5)
        if not results:
            print("  (không tìm thấy kết quả)")
        for i, r in enumerate(results, 1):
            print(f"[{i}] score={r['score']:.4f} | {r['metadata']['doc_type']} | {r['metadata']['source']}")
            print(f"     {r['content'][:120]}...")
