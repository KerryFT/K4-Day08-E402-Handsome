"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

import sys
from typing import Optional
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _cosine_sim(v1: list[float], v2: list[float]) -> float:
    """Tính Cosine Similarity giữa 2 vector."""
    if not v1 or not v2:
        return 0.0
    a = np.array(v1)
    b = np.array(v2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not candidates:
        return []

    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[query, c["content"]] for c in candidates]
        scores = model.predict(pairs)

        reranked = []
        for cand, score in zip(candidates, scores):
            item = cand.copy()
            item["score"] = float(round(score, 4))
            reranked.append(item)

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:top_k]
    except Exception:
        # Fallback sắp xếp theo score sẵn có
        sorted_cands = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
        return sorted_cands[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            cand = candidates[idx]
            # Relevance to query
            if "embedding" in cand and cand["embedding"]:
                relevance = _cosine_sim(query_embedding, cand["embedding"])
            else:
                relevance = cand.get("score", 0.0)

            # Max similarity to already selected candidates
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sel_cand = candidates[sel_idx]
                if (
                    "embedding" in cand
                    and "embedding" in sel_cand
                    and cand["embedding"]
                    and sel_cand["embedding"]
                ):
                    sim = _cosine_sim(cand["embedding"], sel_cand["embedding"])
                else:
                    sim = 1.0 if cand["content"] == sel_cand["content"] else 0.0
                max_sim_to_selected = max(max_sim_to_selected, sim)

            # MMR Score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    output = []
    for i in selected:
        item = candidates[i].copy()
        output.append(item)
    return output


def rerank_rrf(
    ranked_lists: list[list[dict]] | list[dict], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker) hoặc 1 danh sách candidates.
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    if not ranked_lists:
        return []

    # Xử lý trường hợp nhận vào 1 danh sách candidate duy nhất hoặc danh sách các danh sách
    if isinstance(ranked_lists, list) and len(ranked_lists) > 0 and isinstance(ranked_lists[0], dict):
        lists_to_fuse = [ranked_lists]
    else:
        lists_to_fuse = ranked_lists

    rrf_scores = {}   # content -> score
    content_map = {}  # content -> full dict

    for ranked_list in lists_to_fuse:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank))
            if key not in content_map:
                content_map[key] = item

    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = round(score, 6)
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict] | list[list[dict]],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        try:
            from task4_chunking_indexing import get_embedding_model
        except ImportError:
            from src.task4_chunking_indexing import get_embedding_model

        model = get_embedding_model()
        query_emb = model.encode(query).tolist()
        return rerank_mmr(query_emb, candidates, top_k)
    elif method == "rrf":
        return rerank_rrf(candidates, top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Chính sách trả hàng và hoàn tiền Shopee trong 15 ngày", "score": 0.8, "metadata": {}},
        {"content": "Các phương thức thanh toán hỗ trợ trên Shopee Vietnam", "score": 0.6, "metadata": {}},
        {"content": "Quy định đăng bán sản phẩm dành cho người bán", "score": 0.5, "metadata": {}},
    ]
    results = rerank("chính sách trả hàng shopee", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
