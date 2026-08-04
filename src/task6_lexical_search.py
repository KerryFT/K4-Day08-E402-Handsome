"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}


def load_corpus() -> list[dict]:
    """
    Load corpus chunks.
    1. Lấy dữ liệu từ ChromaDB nếu đã được index từ Task 4 (qua get_collection() hoặc PersistentClient).
    2. Fallback: Tự động đọc và chunk các file markdown từ data/standardized/.
    """
    global CORPUS
    if CORPUS:
        return CORPUS

    chunks = []

    # 1. Thử load trực tiếp qua get_collection() của Task 4 hoặc ChromaDB persistent client
    try:
        from src.task4_chunking_indexing import get_collection
        collection = get_collection()
        results = collection.get()
        if results and results.get("documents"):
            for doc, meta in zip(results["documents"], results["metadatas"]):
                chunks.append({"content": doc, "metadata": meta or {}})
            if chunks:
                CORPUS = chunks
                return CORPUS
    except Exception:
        pass

    if CHROMA_DIR.exists():
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            collection_names = [c.name for c in client.list_collections()]
            if "ecommerce_support_docs" in collection_names:
                collection = client.get_collection("ecommerce_support_docs")
                results = collection.get()
                if results and results.get("documents"):
                    for doc, meta in zip(results["documents"], results["metadatas"]):
                        chunks.append({"content": doc, "metadata": meta or {}})
                    if chunks:
                        CORPUS = chunks
                        return CORPUS
        except Exception:
            pass

    # 2. Fallback: Đọc trực tiếp từ data/standardized/ và chunk dữ liệu
    if STANDARDIZED_DIR.exists():
        md_files = list(STANDARDIZED_DIR.rglob("*.md"))
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            use_splitter = True
        except ImportError:
            use_splitter = False

        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8")
            doc_type = "legal" if "legal" in str(md_file) else "news"
            metadata = {"source": md_file.name, "type": doc_type}

            if use_splitter:
                splits = splitter.split_text(content)
                for i, split_text in enumerate(splits):
                    chunks.append({
                        "content": split_text,
                        "metadata": {**metadata, "chunk_index": i}
                    })
            else:
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                for i, p in enumerate(paragraphs):
                    chunks.append({
                        "content": p,
                        "metadata": {**metadata, "chunk_index": i}
                    })

    CORPUS = chunks
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def lexical_search(query: str, top_k: int = 10, corpus: list[dict] = None) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa
        corpus: Corpus tùy chọn (nếu None sẽ tự load qua load_corpus())

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if corpus is None:
        corpus = load_corpus()

    if not corpus:
        return []

    bm25 = build_bm25_index(corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Lấy thứ tự các index có điểm giảm dần
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in sorted_indices:
        if scores[idx] > 0:
            results.append({
                "content": corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": corpus[idx]["metadata"]
            })

    return results


if __name__ == "__main__":
    # Test thử lexical search
    print("=" * 50)
    print("Task 6: Lexical Search (BM25)")
    print("=" * 50)

    test_query = "phương thức thanh toán shopee"
    print(f"\nQuery: '{test_query}'")
    results = lexical_search(test_query, top_k=5)

    if not results:
        print("  Không tìm thấy kết quả phù hợp nào.")
    else:
        for r in results:
            print(f"  [{r['score']:.3f}] {r['content'][:100]}...")

