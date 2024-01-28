# Vietnamese Extractive QA for High School History
Chúng tôi xây dựng một quá trình Extractive QA nhằm giải quyết những câu hỏi lịch sử trong chương trình THPT. Quá trình Extractive QA có thể được mô tả với 3 thành phần chính như sau: Corpus (kho văn bản, nơi chứa các văn bản chứa thông tin), Document Retriever (trình tìm kiếm thông tin dựa trên câu hỏi / từ khóa) và Reader (mô hình hỏi đáp, trích ra câu hỏi khi được cung cấp context và question).


# Xây dựng kho văn bản
Dữ liệu của chúng tôi bao gồm:
- Toàn bộ phần văn bản trong sách giáo khoa lịch sử lớp 10, 11, 12, Nhà xuất bản Giáo dục Việt Nam.
- Các phần đề cương lịch sử được crawl từ các trang web giáo dục uy tín như hocmai...
- Được xử lý theo định dạng Tiêu đề - Văn bản nhằm giúp phần tìm kiếm thông tin dễ dàng hơn. 

Dữ liệu được chúng tôi thu thập bằng nhiều phương pháp khác nhau (crawl văn bản trên website, OCR bằng Google Lens...).

# Tìm kiếm thông tin
Trong phần này, chúng tôi sử dụng BM25 và Semantic Search để tìm kiếm đoạn văn bản phù hợp dựa trên câu hỏi được cung cấp. Chúng tôi cũng đề xuất một phương pháp kết hợp nhằm xử lý những trường hợp tìm được đúng đoạn văn bản nhưng lại không tìm được câu trả lời (có thể do độ dài của đoạn văn bản tìm được).
- BM25 mang lại kết quả cao, được chúng tôi ưu tiên sử dụng.
- Semantic Search: chúng tôi sử dụng mô hình SimeCSE_Vietnamese, xem thêm [tại đây](https://github.com/vovanphuc/SimeCSE_Vietnamese)

# Mô hình QA
Chúng tôi sử dụng pretrained-model Extractive QA của Nguyễn Vũ bình Lê, chi tiết xem thêm [tại đây](https://huggingface.co/nguyenvulebinh/vi-mrc-base)

Vì mục đích sử dụng, chúng tôi đổi tên thư mục gốc thành extractive_qa_mrc, chúng tôi hoàn toàn tôn trọng và đảm bảo quyền tác giả.

# Demo
Vào terminal và sử dụng lệnh:
```python
streamlit run demo.py

```
