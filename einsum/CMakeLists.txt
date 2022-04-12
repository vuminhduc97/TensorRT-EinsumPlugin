cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 11)

include_directories(/home/xzy/Data2/xzyLinuxInstallPackage/5TensorRT/TensorRT-7.2.1.6/include/)
file(GLOB_RECURSE trtlib /home/xzy/Data2/xzyLinuxInstallPackage/5TensorRT/TensorRT-7.2.1.6/lib/*.so)
# set(trthead /home/xzy/Data2/xzyLinuxInstallPackage/5TensorRT/TensorRT-8.0.0.3/include/)
# file(GLOB_RECURSE trtlib /home/xzy/Data2/xzyLinuxInstallPackage/5TensorRT/TensorRT-8.0.0.3/lib/*.so)
message("tensort lib: in " ${trtlib})

find_package(CUDA REQUIRED)
file(GLOB_RECURSE CUDA_LIBRARIES /usr/local/cuda/lib64/*.so) # Máy mình không hiểu sao file so không được nhập hoàn chỉnh nên mình tự nhập toàn bộ, nếu môi trường của bạn ổn thì không cần thêm câu này
message("cuda include : " ${CUDA_INCLUDE_DIRS})
message("cuda lib : " ${CUDA_LIBRARIES})

include_directories(${trthead} ${CUDA_INCLUDE_DIRS}) # Liên kết cuda riêng, thư viện cuda có thể được gọi cho einsum_common
link_libraries(${trtlib} ${CUDA_LIBRARIES})

set(einsum_common_dir einsum_common7) # Đặt phiên bản của einsum_common, tức là phiên bản của tensorrt. Hiện tại, chỉ có các phiên bản 8 và 7 của tensorrt được hỗ trợ.
add_subdirectory(${einsum_common_dir} einsum_common)

add_library(einsumlib Einsum.cpp)
set (Einsum_Heads
    ${trthead}
    ${CUDA_INCLUDE_DIRS}
    ${einsum_common_dir}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set (Einsum_Libs
    ${trtlib}
    ${CUDA_LIBRARIES}
    einsum_common_lib
)
target_include_directories(einsumlib PUBLIC ${Einsum_Heads}) # Sử dụng PUBLIC để nó có thể được chuyển lên và các thư viện được liên kết có thể được sử dụng bởi các thư viện sử dụng einsumlib
target_link_libraries(einsumlib PUBLIC ${Einsum_Libs})
