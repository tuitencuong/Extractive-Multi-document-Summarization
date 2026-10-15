# Web trực quan hóa tóm tắt đa văn bản

Web local này cho phép nhập nhiều văn bản, thêm ô nhập, upload file `.txt`, `.docx`, `.pdf`, chạy từng bước và xem câu nào trong văn bản gốc được chọn vào bản tóm tắt.

## Cài thư viện

Từ folder gốc của repo:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\web_visualizer\requirements.txt
```

## Chạy web

```powershell
.\.venv\Scripts\python.exe .\web_visualizer\app.py
```

Mở trình duyệt tại:

```text
http://127.0.0.1:5000
```

## Các bước trên web

1. `Tiền xử lý`: tách câu, chuẩn hóa, token hóa tiếng Việt, bỏ câu có tối đa 4 token. Mở từng câu để xem token.
2. `KMeans`: Word2Vec 256 chiều, cộng vector từ thành vector câu, gom cụm và chọn câu gần tâm. Mỗi thẻ hiển thị số thành viên thực tế, mở rộng để đọc các câu trong cụm.
3. `Centroid`: chấm điểm TF-IDF theo chủ đề chung và lọc câu quá giống nhau (ngưỡng 0,95). Thanh điểm và nhãn cho biết câu được giữ.
4. `Điều chỉnh λ`: chọn λ từ 0–1 (mặc định 0,6), nhấn Áp dụng để tính lại và mở kết quả MMR.
5. `MMR`: hiển thị thứ tự chọn thực tế, điểm liên quan và trùng lặp với công thức λ = 0,6. Câu đầu là câu mốc; cả train/test chọn `ceil(0,7 × C)` câu với C là số ứng viên sau Centroid (đã tính câu mốc).
6. `Câu đã chọn`: giữ câu MMR, không bổ sung lại câu đã loại; hiển thị tập câu trước Sort.
7. `Sort & kết quả`: sắp theo thứ tự tài liệu nhập vào và vị trí câu trong tài liệu; đối chiếu trước–sau, tô vàng câu được chọn và sao chép bản đã Sort.

## Số câu và bố cục

- Để trống **Số câu tóm tắt** để dùng quy tắc trong `Kmeans_CentroidBase_MMR_SentencePosition.py`: dưới 20 câu hợp lệ → 3 cụm; 20–49 → 5; 50–99 → 8; từ 100 → 10. K chỉ quyết định số đại diện. Số câu tự động bằng 70% số ứng viên sau Centroid, làm tròn lên; không bổ sung trở lại cho đủ K.
- Nhập số nguyên 1–16 để giới hạn thêm đầu ra; số câu cuối không vượt số câu MMR đã chọn.
- Kéo thanh có dấu ⋮ giữa hai vùng để đổi chiều rộng. Có thể dùng phím ←/→ khi thanh được focus; nhấp đúp để đặt lại. Chiều rộng được lưu trong trình duyệt. Màn hình nhỏ chuyển thành bố cục dọc.

## Ghi chú

- `.txt` được đọc trực tiếp với nhiều encoding phổ biến.
- `.docx` cần `python-docx`.
- `.pdf` cần `pypdf`; chất lượng trích xuất phụ thuộc vào PDF có text thật hay chỉ là ảnh scan.
- Điểm và cụm trong giao diện dùng để trực quan hóa pipeline, không ghi đè dữ liệu trong `Data/ViMs`.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m unittest web_visualizer.test_app web_visualizer.test_pipeline_demo web_visualizer.test_source_order -v
```

## Demo tương tác tích hợp

Chọn **Demo tương tác** ở đầu vùng kết quả. Phần này chạy trực tiếp `run_pipeline` và dùng lại các cảnh mô phỏng của `methods/main_method/interactive_pipeline_demo.py`, trong cùng web Flask; không cần chạy thêm server cổng 8765.

- Nhập văn bản hoặc tải file như chế độ tổng quan. Có nút **Nạp mẫu 3 tài liệu** để thử nhanh.
- Tham số demo gồm K (lấy mặc định trực tiếp từ file demo; hiện là 6; xóa để tự động), λ MMR, chế độ train/test và ngưỡng cosine. Centroid demo dùng cosine + lọc trùng, khác với chế độ tổng quan dùng CentroidBow gốc.
- Ô số câu tóm tắt dùng chung: để trống giữ toàn bộ `ceil(0,7 × C)` câu MMR; nhập 1–16 để giới hạn thêm đầu ra. Train/test đều dùng tỷ lệ 70%; bước cuối chỉ sắp thứ tự, không bổ sung câu.
- Sáu giai đoạn: Đầu vào & biểu diễn → K-means → Câu đại diện → Centroid → MMR → Kết quả. Dùng Trước/Tiếp, phím ←/→ ngoài ô nhập, hoặc Tự chạy và chỉnh tốc độ. Các cảnh K-means có điều khiển riêng để xem gán từng câu và dịch chuyển tâm.
- Chuyển chế độ sẽ dừng mô phỏng và giữ kết quả của mỗi chế độ. Nhấn **Chạy** sau khi đổi nội dung/tham số để tính lại.
- Demo giới hạn 30 tài liệu, 100.000 ký tự, 5–300 câu hợp lệ, K tối đa 30; ngưỡng cosine quá cao có thể không giữ được câu nào và sẽ báo cần giảm ngưỡng.

`web_visualizer/pipeline_demo.py` chuyển renderer từ file demo thành asset có phạm vi riêng để giữ giao diện chung. Khởi động lại Flask sau khi sửa file demo để cập nhật renderer.

Bước cuối ở cả hai chế độ là **Sort theo thứ tự input**. Sort giữ nguyên tập câu đã chọn, chỉ sắp theo chỉ số xuất hiện gốc; nội dung sao chép cũng dùng thứ tự này. Demo giữ các cảnh, đồ thị và màu gốc của `interactive_pipeline_demo.py`; chỉ nối thêm một bước Sort cuối. Phần tích hợp dùng phạm vi CSS riêng để không ảnh hưởng giao diện tóm tắt.

## Đối chiếu số bước demo

Web dùng trực tiếp `flowBuild` trong file demo và chỉ nối thêm 1 bước Sort. Số bước phụ thuộc K, số vòng K-means, ứng viên Centroid và các lượt MMR; không có giới hạn cứng 40/61. Cùng một kết quả pipeline, số bước trên web luôn bằng số bước file gốc + 1. Dòng trạng thái hiển thị rõ số bước gốc và bước Sort.

Nạp mẫu sẽ khôi phục cả văn bản và tham số gốc (hiện K = 6). Renderer và backend cùng đọc phiên bản mới nhất của file demo; tải lại trang sau khi sửa file. Hai chế độ dùng chung màu xanh ngọc của web; màu riêng của các cụm và cảnh báo vẫn giữ ý nghĩa.
