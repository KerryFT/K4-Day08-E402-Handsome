# PHÂN CÔNG THÀNH VIÊN THEO CHECKPOINT

Thành Viên: Hoàng Vũ Trung Nguyên - 2A202601076
            Hoàng Trung Hải - 2A202601054
            Vũ Hữu Trường - 2A202601694

> `Checkpoint 0` là bước setup môi trường chung. Năm bảng bên dưới tương ứng với
> năm checkpoint công việc chính từ `Checkpoint 1` đến `Checkpoint 5`.

## Checkpoint 1 — Thu thập và chuẩn hóa dữ liệu

**Thời gian:** 0:10–0:35 (25 phút)  
**Phạm vi:** Task 1–3

| Role | Vai trò | Mô tả công việc cơ bản | Thành viên |
|---|---|---|---|
| Role 1 | Team Leader & RAG Architect | Phân công nguồn dữ liệu và kiểm tra để các thành viên không thu thập trùng tài liệu. | Hoàng Vũ Trung Nguyên |
| Role 2 | Data & Pipeline Specialist / Data Dev | Thực hiện Task 1: tải tối thiểu 3 tài liệu chính sách gốc vào `data/landing/legal/`. | Vũ Hữu Trường |
| Role 3 | Frontend & Chatbot Dev | Thực hiện Task 2: crawl tối thiểu 5 bài viết hoặc thông báo vào `data/landing/news/`. | Vũ Hữu Trường |
| Role 4 / 5 / 6 | Evaluation & QA Engineer | Thực hiện Task 3: chuyển toàn bộ tài liệu thành Markdown trong `data/standardized/`. | Hoàng Trung Hải |

**Kết quả cần đạt:** Có ít nhất 3 file legal, 5 file news và các file Markdown
tương ứng trong `data/standardized/`.

---

## Checkpoint 2 — Chunking, indexing và search cơ bản

**Thời gian:** 0:35–1:00 (25 phút)  
**Phạm vi:** Task 4–6

| Role | Vai trò | Mô tả công việc cơ bản | Thành viên |
|---|---|---|---|
| Role 1 | Team Leader & RAG Architect | Kiểm tra `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100` và xác nhận model embedding `BAAI/bge-m3`. | Hoàng Vũ Trung Nguyên |
| Role 2 | Data & Dense Search Dev | Thực hiện Task 4: chunk tài liệu, tạo embedding và index dữ liệu vào ChromaDB (`chroma_db/`). | Vũ Hữu Trường |
| Role 3 | Sparse Search Dev / UI Dev | Thực hiện Task 5: hoàn thiện `semantic_search()` sử dụng cosine similarity và HyDE. | Hoàng Trung Hải |
| Role 4 / 5 / 6 | Evaluation & QA Engineer | Thực hiện Task 6: hoàn thiện `lexical_search()` sử dụng BM25 hoặc TF-IDF. | Hoàng Trung Hải |

**Kết quả cần đạt:** Tạo được `chroma_db/`; các kiểm thử Task 4, 5 và 6 pass;
demo được sự khác nhau giữa Semantic Search và BM25.

---

## Checkpoint 3 — Reranking và vectorless fallback

**Thời gian:** 1:00–1:20 (20 phút)  
**Phạm vi:** Task 7–8

| Role | Vai trò | Mô tả công việc cơ bản | Thành viên |
|---|---|---|---|
| Role 1 | Team Leader & RAG Architect | Kiểm tra công thức Reciprocal Rank Fusion với `k=60`, bảo đảm cân bằng kết quả Semantic và BM25. | Hoàng Vũ Trung Nguyên |
| Role 2 | Pipeline Specialist / Sparse Dev | Thực hiện Task 7: hoàn thiện `rerank_rrf()` trong `src/task7_reranking.py`. | Vũ Hữu Trường |
| Role 3 | Frontend & Chatbot Dev | Thực hiện Task 8: tích hợp PageIndex để truy vấn tài liệu theo cấu trúc mà không cần vector. | Hoàng Trung Hải |
| Role 4 / 5 / 6 | Evaluation & QA Engineer | Dùng câu hỏi ngoài domain để kiểm tra hệ thống có kích hoạt PageIndex fallback đúng hay không. | Hoàng Vũ Trung Nguyên |

**Kết quả cần đạt:** RRF gộp được hai danh sách kết quả; PageIndex trả kết quả
phù hợp; nhóm giải thích được fallback dựa trên cosine score thay vì RRF score.

---

## Checkpoint 4 — Pipeline hoàn chỉnh và generation

**Thời gian:** 1:20–1:45 (25 phút)  
**Phạm vi:** Task 9–10

| Role | Vai trò | Mô tả công việc cơ bản | Thành viên |
|---|---|---|---|
| Role 1 | Team Leader & RAG Architect | Rà soát toàn bộ mã nguồn và chạy `pytest tests/test_individual.py` để nghiệm thu các task cá nhân. | Hoàng Vũ Trung Nguyên |
| Role 2 | Data & Pipeline Specialist | Thực hiện Task 9: nối Semantic Search, BM25, RRF và PageIndex fallback thành một retrieval pipeline. | Vũ Hữu Trường |
| Role 3 | Frontend & Chatbot Dev | Thực hiện Task 10: reorder tài liệu, gọi LLM và sinh câu trả lời có citation. | Hoàng Trung Hải |
| Role 4 / 5 / 6 | Evaluation & QA Engineer | Rà soát định dạng citation và kiểm tra các khẳng định trong câu trả lời có nguồn phù hợp. | Hoàng Vũ Trung Nguyên |

**Kết quả cần đạt:** Bộ kiểm thử cá nhân đạt 35/35; pipeline trả kết quả đúng cấu
trúc; câu trả lời có trích dẫn nguồn.

---

## Checkpoint 5 — Chatbot UI và đánh giá RAGAS

**Thời gian:** 1:45–2:15 (30 phút)  
**Phạm vi:** Bài tập nhóm

| Role | Vai trò | Mô tả công việc cơ bản | Thành viên |
|---|---|---|---|
| Role 1 | Team Leader & RAG Architect | Tổng hợp code tốt nhất của nhóm vào `app.py`, điều phối tích hợp và theo dõi tiến độ báo cáo. | Hoàng Vũ Trung Nguyên |
| Role 2 | Data & Pipeline Specialist | Kết nối `generate_with_citation()` từ Task 10 vào luồng xử lý câu hỏi của `app.py`. | Hoàng Vũ Trung Nguyên |
| Role 3 | Frontend & Chatbot Dev | Hoàn thiện Streamlit UI: giao diện chat, thanh `top_k`, câu hỏi gợi ý và vùng hiển thị nguồn tham khảo. | Hoàng Vũ Trung Nguyên |
| Role 4 / 5 / 6 | Evaluation & QA Engineer | Tạo golden dataset 15–20 câu, chạy RAGAS và hoàn thiện báo cáo A/B trong `group_project/evaluation/results.md`. | Hoàng Trung Hải |

**Kết quả cần đạt:** Chatbot trả lời kèm danh sách nguồn; golden dataset đủ số
lượng; báo cáo có đầy đủ chỉ số và so sánh A/B.

