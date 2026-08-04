"""
Task 1 — Thu thập văn bản chính sách thương mại điện tử / hỗ trợ khách hàng.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản chính sách (PDF/DOCX) từ trang chính thức của một sàn TMĐT.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, mô tả đúng nội dung.

Gợi ý nguồn (ví dụ trang công khai Shopee Vietnam — help.shopee.vn):
    - https://help.shopee.vn/portal/4/article/77251 (Chính sách trả hàng và hoàn tiền)
    - https://help.shopee.vn/portal/4/article/79198 (Phương thức thanh toán)
    - https://help.shopee.vn/portal/4/article/77244 (Chính sách bảo mật)

Gợi ý văn bản (chủ đề chính sách thương mại điện tử):
    - Chính sách đổi trả/hoàn tiền (Returns/Refund Policy)
    - Phương thức thanh toán (Payment Methods)
    - Chính sách bảo mật (Privacy Policy)
    - Quy định đăng bán sản phẩm cho người bán (Seller Listing Regulations)

Nhớ gắn metadata `customer_role` (`buyer`/`seller`/`both`) cho từng tài liệu — yêu cầu riêng
của K4 Variant (kế thừa từ Lab 07), cần thiết để viết benchmark query dùng metadata_filter.

Lưu ý: một số trang help center dùng JavaScript render nội dung (SPA) — crawl về chỉ thấy
tiêu đề mà không có nội dung thật. Đổi sang bài viết khác cùng domain thay vì cố xử lý,
và chỉ dùng nguồn công khai/được phép chia sẻ.
"""

from pathlib import Path
import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def download_file(url: str, filename: str) -> Path:
    """Tải file từ URL và lưu vào DATA_DIR (nếu có direct link)."""
    response = requests.get(url)
    response.raise_for_status()
    filepath = DATA_DIR / filename
    filepath.write_bytes(response.content)
    print(f"✓ Đã tải: {filepath}")
    return filepath


def collect_legal_docs() -> list[Path]:
    """
    Thu thập và xác nhận các file chính sách (PDF/DOCX) trong thư mục DATA_DIR.
    Nếu chưa đủ 3 file, có thể tải về từ nguồn hỗ trợ.
    """
    setup_directory()

    valid_exts = {".pdf", ".docx", ".doc"}
    existing_files = [
        f for f in DATA_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in valid_exts
    ]

    print(f"--- Đã tìm thấy {len(existing_files)} file chính sách trong {DATA_DIR} ---")
    for f in existing_files:
        size_kb = f.stat().st_size / 1024
        print(f"  • {f.name} ({size_kb:.1f} KB)")

    if len(existing_files) >= 3:
        print(f"✓ Đã có đủ {len(existing_files)} file chính sách hợp lệ (yêu cầu ≥ 3).")
    else:
        print(f"⚠️ Cần thêm {3 - len(existing_files)} file chính sách nữa vào {DATA_DIR}.")

    return existing_files


if __name__ == "__main__":
    collect_legal_docs()

