"""Streamlit chatbot kết nối Retrieval Task 9 và Generation Task 10.

Chạy bằng lệnh: streamlit run app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def _format_score(value) -> str:
    """Format score an toàn khi retrieval không trả về số."""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def render_sources(sources: list[dict]) -> None:
    """Hiển thị danh sách chunk tham khảo theo một format thống nhất."""
    if not sources:
        return

    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)"):
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata") or {}
            source_name = metadata.get("source", "Không rõ nguồn")
            doc_type = metadata.get("type", metadata.get("doc_type", "unknown"))
            retrieval_type = source.get("source", "hybrid")
            st.markdown(
                f"**[{index}] {source_name}**  \n"
                f"`{doc_type}` · `{retrieval_type}` · "
                f"score `{_format_score(source.get('score'))}`"
            )

            content = str(source.get("content", "")).strip()
            st.caption(content[:800] + ("…" if len(content) > 800 else ""))
            if index < len(sources):
                st.divider()


st.set_page_config(
    page_title="E-commerce Support RAG Chatbot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


with st.sidebar:
    st.title("🛒 E-commerce Support RAG")
    st.caption(
        "Trợ lý hỏi đáp về đổi trả, thanh toán, vận chuyển, bảo mật "
        "và chính sách người bán."
    )

    st.divider()
    st.subheader("📡 Trạng thái hệ thống")
    index_ready = (PROJECT_ROOT / "chroma_db" / "chroma.sqlite3").exists()
    api_ready = any(
        os.getenv(name)
        for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")
    )
    st.caption(f"{'✅' if index_ready else '⚠️'} Vector index")
    st.caption(f"{'✅' if api_ready else '⚠️'} LLM API key")

    st.divider()
    st.subheader("💡 Câu hỏi gợi ý")
    suggestions = [
        "Thời hạn yêu cầu trả hàng hoặc hoàn tiền là bao lâu?",
        "Những trường hợp nào người mua được hoàn tiền?",
        "Chính sách vận chuyển quy định như thế nào?",
        "Shopee bảo vệ dữ liệu cá nhân ra sao?",
        "Các hành vi gian lận nào bị nghiêm cấm?",
    ]
    for suggestion in suggestions:
        if st.button(
            suggestion,
            use_container_width=True,
            key=f"suggestion_{suggestion}",
        ):
            st.session_state.pending_query = suggestion

    st.divider()
    st.subheader("⚙️ Thiết lập")
    top_k = st.slider(
        "Số chunks retrieval (top_k)",
        min_value=2,
        max_value=10,
        value=5,
        help="Số đoạn tài liệu được đưa vào bước sinh câu trả lời.",
    )

    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    st.divider()
    st.caption("**Kiến trúc:**")
    st.caption(
        "Semantic + BM25 → RRF rerank → PageIndex fallback → "
        "LLM generation có citation"
    )


st.title("🛒 E-commerce Support RAG Chatbot")
st.caption("Hỏi đáp dựa trên bộ tài liệu chính sách đã thu thập của nhóm")

if not index_ready:
    st.info(
        "Vector index chưa sẵn sàng. Hãy chạy "
        "`python -m src.task4_chunking_indexing` trước khi đặt câu hỏi."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            retrieval_source = message.get("retrieval_source")
            generation_mode = message.get("generation_mode")
            if retrieval_source or generation_mode:
                st.caption(
                    f"Retrieval: `{retrieval_source or 'unknown'}` · "
                    f"Generation: `{generation_mode or 'unknown'}`"
                )
            render_sources(message.get("sources", []))


user_input = st.chat_input(
    "Nhập câu hỏi về chính sách thương mại điện tử…",
    disabled=not index_ready,
)
query = user_input or st.session_state.pending_query

if query and index_ready:
    st.session_state.pending_query = None
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    answer = ""
    sources: list[dict] = []
    retrieval_source = "none"
    generation_mode = "error"
    warning = None

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm tài liệu và tổng hợp câu trả lời…"):
            try:
                from src.task10_generation import generate_with_citation

                response = generate_with_citation(query, top_k=top_k)
                answer = response.get("answer", "Chưa thể trả lời.")
                sources = response.get("sources", [])
                retrieval_source = response.get("retrieval_source", "none")
                generation_mode = response.get("generation_mode", "unknown")
                warning = response.get("warning")
            except Exception as exc:
                answer = (
                    "❌ **Không thể xử lý câu hỏi lúc này.** "
                    "Vui lòng kiểm tra vector index và cấu hình API."
                )
                warning = str(exc)

        st.markdown(answer)
        st.caption(
            f"Retrieval: `{retrieval_source}` · Generation: `{generation_mode}`"
        )
        if warning:
            with st.expander("ℹ️ Chi tiết trạng thái"):
                st.code(warning)
        render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_source": retrieval_source,
            "generation_mode": generation_mode,
        }
    )
