"""
RAG Chatbot — Pháp luật Ma tuý & Tin tức.

Chạy: streamlit run group_project/app.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="DrugLaw RAG Chatbot",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Chatbot Pháp luật Ma tuý")
st.caption("Trả lời câu hỏi về pháp luật ma tuý và tin tức liên quan · Có citation · Hỗ trợ hội thoại")

# =============================================================================
# LAZY IMPORT (tránh load nặng khi reload UI)
# =============================================================================

@st.cache_resource(show_spinner="Đang khởi tạo pipeline RAG...")
def load_pipeline():
    from task9_retrieval_pipeline import build_bm25_index
    from task10_generation import generate_with_citation
    build_bm25_index()
    return generate_with_citation


generate_with_citation = load_pipeline()


# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []   # [{"role": "user"|"assistant", "content": str, "sources": list}]

if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = []  # Lịch sử cho LLM


# =============================================================================
# HELPERS
# =============================================================================

def build_contextual_query(user_query: str, history: list[dict]) -> str:
    """Kết hợp câu hỏi mới với lịch sử hội thoại để LLM hiểu follow-up."""
    if not history:
        return user_query

    # Lấy tối đa 3 lượt hội thoại gần nhất
    recent = history[-6:]  # 3 user + 3 assistant turns
    history_text = "\n".join(
        f"{'Người dùng' if m['role'] == 'user' else 'Trợ lý'}: {m['content']}"
        for m in recent
    )
    return (
        f"[Lịch sử hội thoại]\n{history_text}\n\n"
        f"[Câu hỏi hiện tại] {user_query}"
    )


def render_sources(sources: list[dict]):
    """Render source documents trong expander."""
    if not sources:
        return

    with st.expander(f"📚 Nguồn tài liệu ({len(sources)} chunks)", expanded=False):
        for i, chunk in enumerate(sources, 1):
            meta = chunk.get("metadata", {})
            source_name = meta.get("source", f"Nguồn {i}")
            doc_type = meta.get("doc_type") or meta.get("type", "unknown")
            score = chunk.get("score", 0.0)
            retrieval_src = chunk.get("source", "hybrid")

            st.markdown(
                f"**[{i}] {source_name}** · `{doc_type}` · "
                f"score: `{score:.3f}` · via `{retrieval_src}`"
            )
            st.caption(chunk.get("content", "")[:300] + "...")
            if i < len(sources):
                st.divider()


# =============================================================================
# DISPLAY CHAT HISTORY
# =============================================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])


# =============================================================================
# CHAT INPUT
# =============================================================================

user_input = st.chat_input("Nhập câu hỏi về pháp luật ma tuý...")

if user_input:
    # Hiển thị tin nhắn người dùng
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    # Tạo query có context lịch sử
    contextual_query = build_contextual_query(
        user_input,
        st.session_state.conversation_context,
    )

    # Gọi RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm và tổng hợp..."):
            try:
                result = generate_with_citation(contextual_query)
                answer = result["answer"]
                sources = result["sources"]
            except Exception as e:
                answer = f"❌ Lỗi khi xử lý câu hỏi: {e}"
                sources = []

        st.markdown(answer)
        render_sources(sources)

    # Lưu vào session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

    # Cập nhật conversation context (chỉ lưu content, không lưu sources)
    st.session_state.conversation_context.append({"role": "user", "content": user_input})
    st.session_state.conversation_context.append({"role": "assistant", "content": answer})


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("⚙️ Cài đặt")

    if st.button("🗑️ Xoá lịch sử hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_context = []
        st.rerun()

    st.divider()
    st.subheader("📊 Thống kê phiên")
    turns = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.metric("Số lượt hỏi", turns)

    st.divider()
    st.subheader("💡 Câu hỏi gợi ý")
    sample_questions = [
        "Hình phạt tàng trữ ma tuý theo pháp luật Việt Nam?",
        "Luật Phòng chống ma tuý 2021 quy định gì về cai nghiện?",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý gần đây?",
        "Thủ tục xử lý người nghiện ma tuý tự nguyện?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True, key=q):
            st.session_state["_prefill"] = q
            st.rerun()

    st.divider()
    st.caption("Stack: Streamlit · Task 9 (Hybrid Retrieval) · Task 10 (GPT-4o-mini)")
