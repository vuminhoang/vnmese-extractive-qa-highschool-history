# Vietnamese Extractive QA for High School History
Chúng tôi xây dựng một quá trình Extractive QA nhằm giải quyết những câu hỏi lịch sử trong chương trình THPT. Quá trình Extractive QA có thể được mô tả với 3 thành phần chính như sau: Corpus (kho văn bản, nơi chứa các văn bản chứa thông tin), Document Retriever (trình tìm kiếm thông tin dựa trên câu hỏi / từ khóa) và Reader (mô hình hỏi đáp, trích ra câu hỏi khi được cung cấp context và question).


# Xây dựng kho văn bản
Dữ liệu của chúng tôi bao gồm:
- Toàn bộ phần văn bản trong sách giáo khoa lịch sử lớp 10, 11, 12, Nhà xuất bản Giáo dục Việt Nam.
- Các phần đề cương lịch sử được crawl từ các trang web giáo dục uy tín như hocmai...
- Được xử lý theo định dạng Tiêu đề - Văn bản nhằm giúp phần tìm kiếm thông tin dễ dàng hơn. 

# Tìm kiếm thông tin

# Mô hình QA

# Demo
Vào terminal và sử dụng lệnh:
```python
streamlit run demo.py

```
