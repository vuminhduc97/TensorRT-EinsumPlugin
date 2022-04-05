## lời tựa

Vì TensorRT không triển khai plugin Einsum và mạng thường xuyên sử dụng tùy chọn này trong quá trình chuyển đổi GCN, do đó, plugin phải được viết bằng tay và nhân tiện, tôi cũng đã học cách viết và đăng ký plugin vào. Kho lưu trữ này cũng nhằm trình bày cách tùy chỉnh plugin đã viết. 4 5 Trong file plugin, chức năng của từng chức năng thành viên được chú thích đơn giản (có thể còn nhiều chỗ chưa rõ ràng, nên tìm hiểu thêm trên Baidu)

Hướng dẫn này rất được khuyến khích [triển khai quyền tự do của các plugin tùy chỉnh TensorRT] (https://zhuanlan.zhihu.com/p/297002406). Theo hướng dẫn này, bạn chắc chắn có thể tạo các plugin có thể sử dụng được. Hiện tại, tất cả các bạn có thể tìm thấy trực tuyến là Biên dịch trực tiếp và tạo một `libnvinfer_plugin.so` mới để thay thế thư viện gốc chính thức. Tuy nhiên, không thể sử dụng phương pháp này khi phiên bản TensorRT-OSS không khớp với TensorRT, vì vậy đây là một phương pháp khác linh hoạt hơn: ** trực tiếp Biên dịch EinsumPlugin thành dự án dự án của riêng bạn **.
Kho này ** chỉ thực hiện thao tác `nctkv, kvw-> nctw` **, các cấu trúc khác tương tự, bạn có thể tự mình sửa đổi plug-in và tạo phiên bản phù hợp cho riêng mình.

## Môi trường

> TensorRT8.0

Sự khác biệt lớn nhất giữa TensorRT8.0 và phiên bản trước là plugin của phiên bản này phải thêm một `throw()` sau mỗi hàm thành viên. Bạn có thể hiểu điều đó bằng cách xem chương trình `Einsum.cpp`. TensorRT7.0 cũng có thể được sử dụng mà không cần sửa đổi mã nào.

Nếu bạn muốn sử dụng TensorRT7, hãy sửa đổi đường dẫn TensorRT trực tiếp trong `einsum/CMakeLists.txt` và chuyển sang` einsum_common7 '

## thủ công

Tham khảo `CMakeLists.txt` trong thư mục gốc để thêm einsum vào dự án của riêng bạn.

 Hãy nhớ sử dụng `REGISTER_TENSORRT_PLUGIN (EinsumCreator) 'để đăng ký plugin Einsum trong tệp nguồn của phân tích cú pháp mô hình (chẳng hạn như onnx2trt_gcn.cpp tại đây), để có thể tìm thấy nó trong quá trình phân tích cú pháp
 Cách sử dụng các trường hợp kiểm thử 

1. Chạy `create_onnx.py` để tạo tệp onnx cho các test
2. Biên dịch kho lưu trữ và sau đó chạy `onnnx2trt_gcn`, nếu không có lỗi nào được báo cáo, điều đó có nghĩa là không có vấn đề gì với plugin einsum

## quá trình

1. Từ tệp plugin của TensorRT-OSS, sao chép một quy trình chính thức, sau đó viết trực tiếp vào đó và thay thế nội dung của chức năng tương ứng.
2. Về thư viện phụ thuộc của Plugin tùy chỉnh, nó chủ yếu dựa vào các thư viện trong `TensorRT-OSS/plugin/common` và` TensorRT-(số phiên bản)/sample/common`, vì vậy các phiên bản tương ứng của `TensorRT-OSS và TensorRT được yêu cầu `Sao chép tất cả các tệp trong đường dẫn này vào một thư mục (chẳng hạn như thư mục` einsum/einsum_common * `trong repo này), sau đó sử dụng cmake để biên dịch và tạo thư viện tương ứng, đồng thời liên kết thư viện với` einsum.cpp` Just

## chi tiết
Vì ví dụ của kho lưu trữ này giống với thư viện phụ thuộc plugin của EinsumPlugin, nên `target_link_libraries (PUBLIC)` được sử dụng để kết nối thư viện với chương trình ví dụ `onnx2trt_gcn`

Chỉ `onnx2trt_gcn.cpp` được sử dụng trong thư mục gốc, các tệp cpp khác không cần thiết và có thể bị xóa.
