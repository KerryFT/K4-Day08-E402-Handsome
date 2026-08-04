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

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Corpus được load lười (lazy) từ ChromaDB lần đầu gọi
CORPUS: list[dict] = []
_bm25_index = None


def _load_corpus():
    """Load corpus từ ChromaDB nếu chưa có."""
    global CORPUS
    if CORPUS:
        return

    try:
        from task4_chunking_indexing import get_collection
    except ImportError:
        from src.task4_chunking_indexing import get_collection

    collection = get_collection()
    data = collection.get(include=["documents", "metadatas"])

    CORPUS = []
    for doc, meta in zip(data["documents"], data["metadatas"]):
        CORPUS.append({"content": doc, "metadata": meta})


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    # Tokenize đơn giản bằng split() (phù hợp cho tiếng Việt đã có dấu cách giữa các từ)
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


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

    global _bm25_index

    _load_corpus()
    if not CORPUS:
        return []

    if _bm25_index is None:
        _bm25_index = build_bm25_index(CORPUS)

    tokenized_query = query.lower().split()
    scores = _bm25_index.get_scores(tokenized_query)

    # Get top_k indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(round(scores[idx], 4)),
                "metadata": CORPUS[idx]["metadata"]
            })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
