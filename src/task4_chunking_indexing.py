"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB — Weaviate embedded không hỗ trợ Windows)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in, nhưng embedded mode không hỗ trợ Windows)
    - ChromaDB (đơn giản, local, hỗ trợ Windows — được dùng ở đây)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb
"""

import os
import sys
from pathlib import Path

# Phải set TRƯỚC khi import huggingface_hub / sentence_transformers
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

sys.stdout.reconfigure(encoding="utf-8")

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Chunking ---
# Chọn HYBRID: MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter
#   Lý do:
#   - Văn bản pháp luật có cấu trúc Chương/Điều/Khoản → MarkdownHeader tận
#     dụng heading để giữ nguyên context theo đơn vị pháp lý có nghĩa.
#   - Bài báo cũng có heading (## Tiêu đề, ### Nội dung).
#   - Fallback RecursiveCharacter cho các section quá dài (ví dụ danh mục
#     chất ma tuý liệt kê hàng trăm dòng).
#   - chunk_size=800: đủ lớn để chứa 1 điều luật trung bình (~200-300 từ),
#     không quá lớn để mất precision khi retrieval.
#   - overlap=100: giữ context giữa các chunk, tránh mất câu quan trọng ở biên.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Embedding ---
# Chọn intfloat/multilingual-e5-small (384 dim, multilingual)
#   Lý do:
#   - Dataset 100% tiếng Việt → all-MiniLM-L6-v2 (English-only) sẽ kém.
#   - multilingual-e5-small hỗ trợ 100+ ngôn ngữ bao gồm tiếng Việt, MTEB tốt.
#   - Chỉ 120MB, tải nhanh, chạy local không cần API key.
#   - Khi mạng ổn định có thể nâng cấp lên BAAI/bge-m3 (1024 dim) để tốt hơn.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384

# --- Weaviate Cloud ---
# Weaviate embedded không hỗ trợ Windows → dùng Weaviate Cloud (free sandbox).
# Set trong .env: WEAVIATE_URL và WEAVIATE_API_KEY
WEAVIATE_COLLECTION = "DrugLawDocs"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "source_path": str(md_file.relative_to(STANDARDIZED_DIR)),
                "type": doc_type,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo hybrid strategy:
      1. MarkdownHeaderTextSplitter: tách theo heading (Chương/Điều/##/###)
      2. RecursiveCharacterTextSplitter: split tiếp nếu section vẫn quá dài

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    # Headers cần tách — pháp luật VN dùng # ## ### và cả Chương/Điều
    headers_to_split_on = [
        ("#", "heading_1"),
        ("##", "heading_2"),
        ("###", "heading_3"),
    ]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,  # giữ header trong chunk để có context
    )

    # Fallback splitter cho section quá dài
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "，", " ", ""],
    )

    chunks = []
    for doc in documents:
        # Bước 1: tách theo header
        header_splits = header_splitter.split_text(doc["content"])

        for split in header_splits:
            text = split.page_content if hasattr(split, "page_content") else split
            # Metadata từ header splitter (heading_1, heading_2, ...)
            split_meta = split.metadata if hasattr(split, "metadata") else {}

            if len(text) <= CHUNK_SIZE:
                # Chunk đủ nhỏ, giữ nguyên
                chunks.append({
                    "content": text.strip(),
                    "metadata": {
                        **doc["metadata"],
                        **split_meta,
                        "chunk_index": len(chunks),
                    },
                })
            else:
                # Bước 2: recursive split nếu section quá dài
                sub_splits = recursive_splitter.split_text(text)
                for i, sub_text in enumerate(sub_splits):
                    chunks.append({
                        "content": sub_text.strip(),
                        "metadata": {
                            **doc["metadata"],
                            **split_meta,
                            "chunk_index": len(chunks),
                            "sub_chunk": i,
                        },
                    })

    # Loại bỏ chunk rỗng
    chunks = [c for c in chunks if c["content"]]
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng BAAI/bge-m3.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    print(f"  Loading model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["content"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks (batch processing)...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity → normalize trước
    )

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    return chunks


def index_to_weaviate(chunks: list[dict]):
    """
    Index chunks vào Weaviate Cloud.
    Đọc WEAVIATE_URL và WEAVIATE_API_KEY từ .env
    """
    import weaviate
    from weaviate.auth import Auth
    from weaviate.classes.config import Configure, DataType, Property
    from dotenv import load_dotenv

    load_dotenv()

    weaviate_url = os.environ.get("WEAVIATE_URL", "").strip()
    weaviate_api_key = os.environ.get("WEAVIATE_API_KEY", "").strip()

    if not weaviate_url or not weaviate_api_key:
        raise ValueError(
            "Thiếu WEAVIATE_URL hoặc WEAVIATE_API_KEY trong file .env\n"
            "Tạo file .env với nội dung:\n"
            "  WEAVIATE_URL=https://your-cluster.weaviate.network\n"
            "  WEAVIATE_API_KEY=your-api-key"
        )

    if not weaviate_url.startswith("http"):
        weaviate_url = "https://" + weaviate_url

    print(f"  Connecting to Weaviate Cloud: {weaviate_url}...")
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )

    try:
        # Xoá collection cũ nếu tồn tại (để re-index)
        if client.collections.exists(WEAVIATE_COLLECTION):
            client.collections.delete(WEAVIATE_COLLECTION)
            print(f"  Deleted existing collection: {WEAVIATE_COLLECTION}")

        # Tạo collection với schema
        collection = client.collections.create(
            name=WEAVIATE_COLLECTION,
            # Vectorizer.none() vì ta tự cung cấp vector từ bge-m3
            vectorizer_config=Configure.Vectorizer.none(),
            # BM25 built-in cho lexical search (Task 6)
            properties=[
                Property(name="content",     data_type=DataType.TEXT),
                Property(name="source",      data_type=DataType.TEXT),
                Property(name="source_path", data_type=DataType.TEXT),
                Property(name="doc_type",    data_type=DataType.TEXT),
                Property(name="heading_1",   data_type=DataType.TEXT),
                Property(name="heading_2",   data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ],
        )
        print(f"  Created collection: {WEAVIATE_COLLECTION}")

        # Batch insert
        print(f"  Inserting {len(chunks)} chunks...")
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                meta = chunk["metadata"]
                batch.add_object(
                    properties={
                        "content":      chunk["content"],
                        "source":       meta.get("source", ""),
                        "source_path":  meta.get("source_path", ""),
                        "doc_type":     meta.get("type", ""),
                        "heading_1":    meta.get("heading_1", ""),
                        "heading_2":    meta.get("heading_2", ""),
                        "chunk_index":  int(meta.get("chunk_index", 0)),
                    },
                    vector=chunk["embedding"],
                )

        # Verify
        count = collection.aggregate.over_all(total_count=True).total_count
        print(f"  [OK] Indexed {count} objects in Weaviate")

    finally:
        client.close()


def index_to_vectorstore(chunks: list[dict]):
    """Wrapper gọi Weaviate indexing."""
    index_to_weaviate(chunks)


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking : hybrid (MarkdownHeader + Recursive)")
    print(f"             chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Store    : Weaviate Cloud — {WEAVIATE_COLLECTION}")
    print("=" * 50)

    print("\n[1/4] Loading documents...")
    docs = load_documents()
    print(f"  Loaded {len(docs)} documents")
    for d in docs:
        print(f"    - {d['metadata']['source_path']} ({len(d['content']):,} chars)")

    print("\n[2/4] Chunking...")
    chunks = chunk_documents(docs)
    print(f"  Created {len(chunks)} chunks")
    legal = sum(1 for c in chunks if c["metadata"]["type"] == "legal")
    news  = sum(1 for c in chunks if c["metadata"]["type"] == "news")
    print(f"    legal: {legal} chunks | news: {news} chunks")

    print("\n[3/4] Embedding...")
    chunks = embed_chunks(chunks)
    print(f"  Embedded {len(chunks)} chunks (dim={EMBEDDING_DIM})")

    print("\n[4/4] Indexing to Weaviate...")
    index_to_vectorstore(chunks)

    print("\n" + "=" * 50)
    print(f"[DONE] Pipeline complete. {len(chunks)} chunks indexed.")


if __name__ == "__main__":
    run_pipeline()
