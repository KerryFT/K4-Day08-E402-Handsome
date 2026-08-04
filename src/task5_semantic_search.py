"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _is_duplicate_chunk(cand1: dict, cand2: dict) -> bool:
    """Kiểm tra 2 đoạn chunk có bị trùng lặp hoặc overlap quá lớn hay không."""
    c1, c2 = cand1["content"].strip(), cand2["content"].strip()
    meta1, meta2 = cand1.get("metadata", {}), cand2.get("metadata", {})

    if c1 == c2:
        return True

    # Nếu thuộc cùng 1 file văn bản và vị trí chunk gần kề (<= 3), hạ ngưỡng overlap xuống 35%
    same_source = (meta1.get("source") == meta2.get("source") and meta1.get("source") is not None)
    idx_diff = abs(meta1.get("chunk_index", -99) - meta2.get("chunk_index", -99))
    threshold = 0.35 if (same_source and idx_diff <= 3) else 0.60

    # Nếu một chunk nằm trọn trong chunk kia (do overlap)
    if len(c1) > 40 and len(c2) > 40:
        if c1 in c2 or c2 in c1:
            return True

    # Word set overlap ratio đối với chunk nhỏ hơn
    words1 = set(c1.lower().split())
    words2 = set(c2.lower().split())
    if not words1 or not words2:
        return False

    intersection = words1.intersection(words2)
    smaller_size = min(len(words1), len(words2))

    if smaller_size > 0 and (len(intersection) / smaller_size) >= threshold:
        return True

    return False


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity với cơ chế khử trùng lặp.

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
    try:
        from task4_chunking_indexing import get_collection, get_embedding_model
    except ImportError:
        from src.task4_chunking_indexing import get_collection, get_embedding_model

    model = get_embedding_model()
    query_vector = model.encode(query).tolist()

    collection = get_collection()
    # Lấy n_results gấp 3 lần top_k để có không gian khử trùng lặp
    fetch_k = max(top_k * 3, 30)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )

    candidates = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        score = max(0.0, 1.0 - dist)  # cosine distance → similarity
        candidates.append({"content": doc, "score": round(score, 4), "metadata": meta})

    # Sort giảm dần theo score
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Khử trùng lặp
    unique_output = []
    for cand in candidates:
        is_dup = False
        for accepted in unique_output:
            if _is_duplicate_chunk(cand, accepted):
                is_dup = True
                break
        if not is_dup:
            unique_output.append(cand)
            if len(unique_output) == top_k:
                break

    return unique_output
    

if __name__ == "__main__":
    # Test
    results = semantic_search("quy định trả hàng hoàn tiền shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
