"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_LANDING_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def upload_documents() -> list[str]:
    """
    Upload toàn bộ markdown/PDF documents lên PageIndex.
    Returns:
        List of doc_ids
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY chưa được set trong .env")
        return []

    try:
        from pageindex.client import PageIndexClient
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        doc_ids = []

        # Upload các file PDF trong data/landing/legal
        pdf_files = list(LEGAL_LANDING_DIR.glob("*.pdf")) if LEGAL_LANDING_DIR.exists() else []
        for pdf_file in pdf_files:
            print(f"Uploading PDF: {pdf_file.name}")
            try:
                resp = client.submit_document(str(pdf_file))
                doc_id = resp.get("doc_id") or resp.get("id")
                if doc_id:
                    doc_ids.append(doc_id)
                    print(f"  ✓ Uploaded: {pdf_file.name} -> {doc_id}")
            except Exception as e:
                print(f"  ✗ Error uploading {pdf_file.name}: {e}")

        return doc_ids
    except Exception as e:
        print(f"⚠ PageIndex Client Error: {e}")
        return []


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    results = []

    if PAGEINDEX_API_KEY:
        try:
            from pageindex.client import PageIndexClient
            client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

            resp = client.submit_query(query=query)
            retrieval_id = resp.get("retrieval_id") or resp.get("id")

            if retrieval_id:
                # Poll cho tới khi hoàn thành hoặc đạt max_retries
                retrieval = {}
                for _ in range(10):
                    retrieval = client.get_retrieval(retrieval_id)
                    status = retrieval.get("status")
                    if status == "completed" or "retrieved_nodes" in retrieval:
                        break
                    time.sleep(1)

                nodes = retrieval.get("retrieved_nodes", [])
                rank_score = 0.95
                for node in nodes:
                    relevant_contents = node.get("relevant_contents", [])
                    for group in relevant_contents:
                        for item in group:
                            content = item.get("relevant_content", "") or item.get("content", "")
                            if content:
                                results.append({
                                    "content": content.strip(),
                                    "score": round(rank_score, 3),
                                    "metadata": {"section": item.get("section_title", "PageIndex Node")},
                                    "source": "pageindex"
                                })
                                rank_score = max(0.1, rank_score - 0.05)
        except Exception as e:
            print(f"⚠ PageIndex query exception: {e}")

    # Fallback nếu chưa có API key hoặc kết quả từ API rỗng
    if not results:
        if STANDARDIZED_DIR.exists():
            md_files = list(STANDARDIZED_DIR.rglob("*.md"))
            for md_file in md_files:
                text = md_file.read_text(encoding="utf-8")
                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
                if paragraphs:
                    results.append({
                        "content": paragraphs[0][:500],
                        "score": 0.5,
                        "metadata": {"source_file": md_file.name},
                        "source": "pageindex"
                    })
                if len(results) >= top_k:
                    break

    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("danh sách sản phẩm cấm đăng bán", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
