"""
RAG Evaluation Pipeline — RAGAS 0.1.x framework.

Chạy:
    python group_project/evaluation/eval_pipeline.py

Yêu cầu:
    pip install "ragas==0.1.21" datasets langchain-openai
    OPENROUTER_API_KEY phải có trong .env
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

# Redirect RAGAS sang OpenRouter bằng cách set env vars trước khi import ragas
os.environ["OPENAI_API_KEY"]  = os.getenv("OPENROUTER_API_KEY", "")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH        = Path(__file__).parent / "results.md"


# =============================================================================
# LOAD DATA
# =============================================================================

def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# RAG PIPELINE WRAPPERS
# =============================================================================

_bm25_built = False

def _ensure_index():
    global _bm25_built
    if not _bm25_built:
        from task9_retrieval_pipeline import build_bm25_index
        build_bm25_index()
        _bm25_built = True


def run_rag(question: str) -> dict:
    """Config A: Hybrid search + Reranking (default)."""
    _ensure_index()
    from task10_generation import generate_with_citation
    result = generate_with_citation(query=question, top_k=5)
    return {
        "answer":   result["answer"],
        "contexts": [c["content"] for c in result["sources"]],
    }


def run_rag_no_rerank(question: str) -> dict:
    """Config B: Hybrid search, không reranking (ablation)."""
    _ensure_index()
    import os as _os
    from openai import OpenAI
    from task9_retrieval_pipeline import retrieve
    from task10_generation import reorder_for_llm, format_context, SYSTEM_PROMPT, MODEL, OPENROUTER_BASE_URL, TEMPERATURE, TOP_P

    chunks    = retrieve(question, top_k=5, use_reranking=False)
    reordered = reorder_for_llm(chunks)
    context   = format_context(reordered)
    user_msg  = f"Context:\n\n{context}\n\n---\n\nQuestion: {question}"

    client = OpenAI(
        api_key=_os.getenv("OPENROUTER_API_KEY"),
        base_url=OPENROUTER_BASE_URL,
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return {
        "answer":   response.choices[0].message.content,
        "contexts": [c["content"] for c in chunks],
    }


# =============================================================================
# RAGAS EVALUATION
# =============================================================================

def evaluate_config(golden_dataset: list[dict], config_name: str, rag_fn) -> dict:
    """
    Chạy RAGAS evaluation trên golden dataset với một config.

    Returns:
        Dict với scores: faithfulness, answer_relevancy, context_recall, context_precision
        và 'dataframe' chứa per-question scores.
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    from datasets import Dataset

    print(f"\n{'='*60}")
    print(f"Evaluating: {config_name}  ({len(golden_dataset)} questions)")
    print("=" * 60)

    # Cấu hình LLM cho RAGAS dùng OpenRouter
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
    )
    ragas_llm = LangchainLLMWrapper(llm)
    for metric in [faithfulness, answer_relevancy, context_recall, context_precision]:
        metric.llm = ragas_llm

    # Thu thập kết quả từ RAG pipeline
    eval_data: dict[str, list] = {
        "question":     [],
        "answer":       [],
        "contexts":     [],
        "ground_truth": [],
    }

    for i, item in enumerate(golden_dataset, 1):
        print(f"  [{i:2d}/{len(golden_dataset)}] {item['question'][:65]}...")
        try:
            result = rag_fn(item["question"])
            eval_data["question"].append(item["question"])
            eval_data["answer"].append(result["answer"])
            eval_data["contexts"].append(result["contexts"])
            eval_data["ground_truth"].append(item["expected_answer"])
        except Exception as e:
            print(f"    SKIP (error): {e}")

    dataset = Dataset.from_dict(eval_data)
    result  = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    df = result.to_pandas()

    scores = {
        "faithfulness":      float(df["faithfulness"].mean()),
        "answer_relevancy":  float(df["answer_relevancy"].mean()),
        "context_recall":    float(df["context_recall"].mean()),
        "context_precision": float(df["context_precision"].mean()),
        "dataframe":         df,
    }

    print(f"\n  [{config_name}] scores:")
    for k, v in scores.items():
        if k != "dataframe":
            status = "✓" if v >= 0.7 else "✗"
            print(f"    {status} {k:22s}: {v:.4f}")

    return scores


# =============================================================================
# EXPORT RESULTS
# =============================================================================

def export_results(scores_a: dict, scores_b: dict, golden_dataset: list[dict]):
    """Ghi kết quả ra results.md."""
    df_a = scores_a.pop("dataframe")
    df_b = scores_b.pop("dataframe")

    metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    avg_a   = sum(scores_a[m] for m in metrics) / len(metrics)
    avg_b   = sum(scores_b[m] for m in metrics) / len(metrics)
    winner  = "Config A (Hybrid + Rerank)" if avg_a >= avg_b else "Config B (Hybrid, No Rerank)"

    # Worst performers theo faithfulness
    df_a["question"] = [item["question"] for item in golden_dataset[:len(df_a)]]
    worst = df_a.nsmallest(3, "faithfulness")[["question", "faithfulness", "answer_relevancy"]]

    lines = [
        "# RAG Evaluation Results\n",
        "## Cấu hình đánh giá\n",
        "- **Framework**: RAGAS 0.1.21",
        "- **LLM Judge**: GPT-4o-mini (OpenRouter)",
        f"- **Golden Dataset**: {len(golden_dataset)} cặp Q&A — pháp luật ma tuý Việt Nam",
        "- **Threshold mục tiêu**: 0.7 cho tất cả metrics\n",
        "---\n",
        "## So sánh A/B\n",
        "| Metric | Config A — Hybrid + Rerank | Config B — Hybrid, No Rerank | Δ (A−B) |",
        "|--------|---------------------------|------------------------------|---------|",
    ]

    for m in metrics:
        a, b  = scores_a[m], scores_b[m]
        delta = a - b
        mark  = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "≈")
        lines.append(f"| {m} | {a:.4f} | {b:.4f} | {mark} {delta:+.4f} |")

    lines += [
        f"| **Average** | **{avg_a:.4f}** | **{avg_b:.4f}** | **{avg_a - avg_b:+.4f}** |",
        "",
        "---\n",
        "## Nhận xét A/B\n",
        f"- **Config tốt hơn**: {winner}",
        f"- Reranking {'**cải thiện**' if avg_a >= avg_b else '**không cải thiện**'} chất lượng tổng thể ({abs(avg_a - avg_b):.4f} điểm)",
        f"- `context_recall` A={scores_a['context_recall']:.4f} vs B={scores_b['context_recall']:.4f} — "
        f"{'Reranking giúp chọn đúng chunks hơn' if scores_a['context_recall'] >= scores_b['context_recall'] else 'Reranking không ảnh hưởng đến recall'}\n",
        "---\n",
        "## Worst Performers (faithfulness thấp nhất — Config A)\n",
        "| # | Câu hỏi | Faithfulness | Answer Relevancy |",
        "|---|---------|-------------|-----------------|",
    ]

    for rank, (_, row) in enumerate(worst.iterrows(), 1):
        q = row["question"][:72] + "..." if len(row["question"]) > 72 else row["question"]
        lines.append(f"| {rank} | {q} | {row['faithfulness']:.4f} | {row['answer_relevancy']:.4f} |")

    lines += [
        "",
        "---\n",
        "## Đề xuất cải tiến\n",
        "1. **Tăng TOP_K retrieval**: Thử top_k=7–10 để cải thiện `context_recall`",
        "2. **Chunk overlap lớn hơn**: Tránh cắt đứt điều khoản pháp lý giữa chừng",
        "3. **Query expansion**: Với câu hỏi có số điều (Điều 249), expand thêm tên luật đầy đủ",
        "4. **Vietnamese cross-encoder**: Fine-tune reranker trên corpus pháp lý tiếng Việt",
        "5. **Golden dataset 50+ câu**: 16 câu là tối thiểu — cần nhiều hơn để kết quả đáng tin cậy\n",
    ]

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults saved → {RESULTS_PATH}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    print("\n[Step 1/3] Config A — Hybrid + Reranking")
    scores_a = evaluate_config(golden_dataset, "Config A — Hybrid + Rerank", run_rag)

    print("\n[Step 2/3] Config B — Hybrid, No Reranking")
    scores_b = evaluate_config(golden_dataset, "Config B — Hybrid, No Rerank", run_rag_no_rerank)

    print("\n[Step 3/3] Exporting results...")
    export_results(scores_a, scores_b, golden_dataset)

    print("\nDone! Xem kết quả tại group_project/evaluation/results.md")
