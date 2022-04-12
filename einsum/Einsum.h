/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef EINSUM_H
#define EINSUM_H
#include "plugin.h"
#include "serialize.hpp"
#include <cudnn.h>
#include <iostream>
#include <string>
#include <vector>

typedef unsigned short half_type;

namespace nvinfer1
{
namespace plugin
{
//! Với cuối cùng, lớp (Einsum) không thể được kế thừa trong tương lai
class Einsum final : public nvinfer1::IPluginV2DynamicExt
{

public:
    /** 03
     * @brief Einsum được sử dụng trong giai đoạn parse. Nó được sử dụng để tạo phương thức khởi tạo được gọi khi plugin (op) được tạo. Nó cần chuyển các trọng số và tham số.
     */
    Einsum(std::string equation);
    /** 04
     * @brief Einsum Được sử dụng trong giai đoạn sao chép, hàm tạo được sử dụng khi sao chép plugin này
     */
    Einsum(std::string equation, int N, int K, int C, int T, int V, int W);
    /**
     * @brief Einsum Đối với giai đoạn deserialize, chuyển các thông số và trọng số được serialize vào plugin và tạo op
     * @param serialData
     * @param serialLength
     */
    Einsum(void const* serialData, size_t serialLength);

    Einsum() = delete;  //! Hàm tạo mặc định phải bị xóa

    ~Einsum() override; //! Call terminate to release video memory and complete destructuring


    //!
    //! \brief clone Clone this plugin object to TensorRT's builder, network or engine
    //!     Hàm tạo thứ hai ở trên sẽ được gọi để hoàn thành bản sao
    //!     It is mainly used to pass constant weights and parameters, and to copy the plugin in multiple copies, so that it can be used by different engines, builders or networks.
    //! \return
    //!
    nvinfer1::IPluginV2DynamicExt* clone() const throw() override;


    /**
     * @brief getNbOutputs How many Tensors (that is, several outputs) are returned by the op, generally return 1 (one output) directly
     * @return
     */
    int getNbOutputs() const throw() override;

    /**
     * @brief getOutputDataType Trả về kiểu dữ liệu của đầu ra（bao gồm float, half[float16], int 8, int32, bool)
     *      Nói chung, kiểu dữ liệu của dữ liệu đầu vào được trả về trực tiếp, tức là kiểu dữ liệu đầu vào và đầu ra giống nhau
     * @param index
     * @param inputTypes
     * @param nbInputs
     * @return
     */
    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const throw() override;


    //!
    //! \brief getOutputDimensions nhận kích thước của batch
    //!     When TensorRT supports Dynamic-shape, the batch dimension must be explicit, that is to say, the dimension processed by TensorRT has changed from the previous three-dimensional [3,-1,-1] to [1,3,-1,-1] ].
    //! Hàm là trả về thứ nguyên của batch, ví dụ trên là trả về 1.
    //!     Những gì chúng ta cần làm trong hàm thành viên này là suy ra thứ nguyên đầu ra của op dựa trên thứ nguyên đầu vào 
    //      (nói chung, thứ nguyên đầu vào có thể được sử dụng trực tiếp làm thứ nguyên đầu ra)
    //!     Lưu ý: Mặc dù thứ nguyên đầu ra được xác định bởi thứ nguyên đầu vào, nhưng thứ nguyên đầu ra này thực sự là một mặc định (nghĩa là nó đã được tính toán trước khi tính toán)。
    //! Nếu kích thước đầu ra của op cần được tính toán theo hoạt động thực tế thì không thể。
    //! \param outputIndex
    //! \param inputs
    //! \param nbInputs
    //! \param exprBuilder
    //! \return
    //!
    // DynamicExt plugins returns DimsExprs class instead of Dims
    DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) throw() override;


    /**
     * @brief initialize Hàm khởi tạo, được gọi khi engine được tạo (nghĩa là trước khi op sẵn sàng bắt đầu chạy)
     *      Chủ yếu khởi tạo trước một số tham số mở ra không gian, nói chung là các tham số cần thiết cho hoạt động cuda
     （For example, conv needs to open up the memory of weight and bias in advance），
     * Nếu người vận hành cần những thông số này, nó phải mở trước bộ nhớ video ở đây。
     *      Lưu ý: Nếu nhà điều hành op cần mở không gian bộ nhớ video tương đối lớn, hãy cố gắng không tự đăng ký dung lượng bộ nhớ video. 
     Bạn có thể sử dụng con trỏ vùng làm việc được chuyển từ giao diện TensorRT chính thức để lấy dung lượng bộ nhớ video。
     * Vì nếu op được gọi nhiều lần bởi một mạng và op này cần mở nhiều dung lượng bộ nhớ video, thì TensorRT sẽ mở rất nhiều bộ nhớ video theo số lần plugin này được gọi khi xây dựng mạng，
     * dẫn đến tràn bộ nhớ. (--self: Sử dụng con trỏ workspace đảm bảo rằng cùng một địa chỉ được sử dụng bởi mỗi op sẽ không gây tràn bộ nhớ video. 
     Tất cả các op hoạt động trong cùng một không gian và một op được hoàn thành sau khi hoạt động.，
     * Đặt các tham số của lần chọn tiếp theo vào không gian này, thực hiện lần chọn tiếp theo, dữ liệu của mỗi lần chọn không cần được giữ lại, 
     vì vậy hãy chạy lần chọn tiếp theo để xóa trực tiếp dữ liệu của lần chọn trước đó)
     * @return
     */
    int initialize() throw() override;


    /**
     * @brief terminate Release some video memory space opened up by the op - used for destructors, called when the engine is destroyed
     */
    void terminate() throw() override;


    /**
     * @brief getWorkspaceSize Trả về kích thước dữ liệu thực tế của biến bộ nhớ video trung gian theo yêu cầu của op (kích thước bộ nhớ video tạm thời)
     (byte size)——Nói chung, hàm chính thức được sử dụng để lấy nó, được tiêu chuẩn hóa nhiều hơn.
     *      Tại đây, hãy xác định dung lượng bộ nhớ video mà op này cần để chạy, để bạn có thể trực tiếp sử dụng TensorRT để mở dung lượng trong quá trình hoạt động thực tế thay vì 
     tự đăng ký dung lượng bộ nhớ video, để tránh vấn đề tràn bộ nhớ video ở trên.。
     * @param inputs
     * @param nbInputs
     * @param outputs
     * @param nbOutputs
     * @return
     */
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                            const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const throw() override;


    /**
     * @brief enqueue Hàm chạy khi op thực sự được thực thi
     *      Đặt quá trình tính toán do chính bạn thực hiện trong hàm này, hàm này thường được thực hiện bởi cuda 
     (hoạt động op được thực hiện bởi C ++ cũng có thể được đưa vào, nhưng do CPU được thực thi nên tốc độ tương đối chậm)
     *      Tính toán outputs theo inputs và chuyển nó đến con trỏ tương ứng
     *      Lưu ý: Nếu op cần lưu trữ tạm thời một số biến trung gian trong bộ nhớ video, nó có thể được lấy thông qua workspace tham số con trỏ đến
     *      .Cu được viết theo mặc định là độ chính xác FP32. Khi TensorRT chạy ở chế độ hoạt động FP16, khi chạy đến op plug-in không hỗ trợ FP16, 
     nó sẽ tự động chuyển sang chế độ FP32 và sau đó chuyển trở lại FP16 sau khi op chạy.，
     * Do đó, việc chuyển đổi dữ liệu thường xuyên như vậy cũng sẽ làm tăng thời gian tiêu thụ
     * @param inputDesc
     * @param outputDesc
     * @param inputs
     * @param outputs
     * @param workspace Địa chỉ của không gian làm việc được cấp phát, 
        qua đó các biến trung gian cần thiết cho phép tính op được lưu trữ tạm thời để tránh phát triển không gian lặp lại
     * @param stream
     * @return
     */
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs,
                void* workspace,
                cudaStream_t stream) throw() override;


    //!
    //! \brief setPluginNamespace Đặt tên namespace cho plugin này, mặc định là "". (nói chung là không đặt)
    //!     Lưu ý: các plugin trong cùng một namespace sẽ xung đột nếu chúng có cùng tên
    //!     sửa đổi mPluginNamespace
    //! \param pluginNamespace
    //!
    void setPluginNamespace(const char* pluginNamespace) throw() override;
    const char* getPluginNamespace() const throw() override;    //! 获取该plugin的namespace名字


    //!
    //! \brief configurePlugin Xác định xem các kiểu dữ liệu đầu vào và đầu ra có chính xác hay không. 
    // Bạn cũng có thể sử dụng thông tin cấu hình này để yêu cầu TensorRT chọn thuật toán thích hợp để điều chỉnh mô hình nó dường như cũng chịu trách nhiệm tính toán các biến trung gian có liên quan
    //!     Mã thực thi plugin mà chúng tôi thường viết là cố định và không cần phải điều chỉnh, vì vậy điều này chủ yếu dành cho hoạt động chính thức
    //! \param in
    //! \param nbInputs
    //! \param out
    //! \param nbOutputs
    //!
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                         const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) throw() override;


    //!
    //! \brief getSerializationSize Returns how many bytes need to be written to the buffer when serializing the op
    //!     Nói chung, trọng số + tổng số byte của tham số
    //! \return
    //!
    size_t getSerializationSize() const throw() override;


    //!
    //! \brief serialize According to the serialization size getSerializationSize(), serialize the data to be used into the buffer in order
    //!     Có nghĩa là trọng số + tham số + không gian bộ nhớ phụ (nó phải là một biến trung gian) được tuần tự hóa vào buffer
    //! \param buffer
    //!
    void serialize(void* buffer) const throw() override;


    // DynamicExt plugin supportsFormat update.
    //!
    //! \brief supportsFormatCombination TensorRT Call this method to determine whether the (input/output) corresponding to the pos index supports the format and data type specified by inOut[pos].format and inOut[pos].type
    //!   Hiện tại, hãy cân nhắc xem định dạng và kiểu dữ liệu của đầu vào / đầu ra có đáp ứng yêu cầu hay không
    //!     kiểu dữ liệu đề cập đến DataType：float, half[float16], int 8, int32, bool
    //!     định dạng đề cập đến TensorFormat：kLINEAR，kNCHW，kNCHW2，
    //! \param pos
    //! \param inOut
    //! \param nbInputs
    //! \param nbOutputs
    //! \return
    //!
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) throw() override;


    //!
    //! \brief attachToContext If this op uses some other things, such as cublas handle, you can directly use the cublas handle provided by TensorRT
    //!     Không chắc chắn về cách sử dụng nó
    //! \param cudnn
    //! \param cublas
    //! \param allocator
    //!
    void attachToContext(cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) throw() override;

    //!
    //! \brief getPluginType Tự đặt tên và số phiên bản của plugin (chẳng hạn như leakyrelu 1)
    //!     Lưu ý: Vì PluginNamespace thường mặc định là "" nên plugin ở đây không được lặp lại, nếu không quá trình biên dịch sẽ báo lỗi (???? để được kiểm tra)
    //! \return
    //!
    const char* getPluginType() const throw() override;
    const char* getPluginVersion() const throw() override;


    void destroy() throw() override;

    void detachFromContext() throw() override;

private:
    std::string equation;
    int N,K,C,T,V,W;
    std::string mNamespace;
//    const char* mPluginNamespace;   //! namepace of plugin，Nói chung là không được đặt, nó là ""
    std::string mPluginNamespace;
};

class EinsumCreator : public BaseCreator
{
public:
    //!
    //! \brief EinsumCreator
    //! 01
    EinsumCreator();

    ~EinsumCreator() override = default;


    //!
    //! \brief getPluginName Tương ứng với getPluginType và getPluginVersion trong lớp plugin. giống hệt nhau
    //! \return
    //!
    const char* getPluginName() const throw() override;
    const char* getPluginVersion() const throw() override;


    //!
    //! \brief getFieldNames
    //!     Trả về dữ liệu cấu trúc PluginFieldCollection, bao gồm tên tham số và loại của plugin đã thêm
    //! \param PluginFieldCollection mFC: Đây là một biến thành viên
    //!     Chức năng chính là chuyển các trọng số và tham số theo yêu cầu của op này, sẽ không được sử dụng trong engine infer, 
    // nhưng được sử dụng trong parse (chẳng hạn như caffe2trt, onnx2trt) để xác định xem parse có thể thành công hay không.。
    //! Khi sử dụng pparse để parse op này, trọng số và thông số của op này sẽ chuyển qua Models-->TensorRT engine --> quá trình của TensorRT runtime。
    //!     Để biết quy trình cụ thể, vui lòng tham khảo liên kết
    //! \return
    //!
    const PluginFieldCollection* getFieldNames() throw() override;


    //!
    //! \brief createPlugin Tạo một plugin thông qua PluginFieldCollection thu được, lấy ra các trọng số và thông số theo yêu cầu của op, sau đó gọi hàm tạo đầu tiên của class plugin để tạo plugin
    //!
    //! another understanding（refer tohttps://github.com/LitLeo/TensorRT_Tutorial/blob/master/blogs/TensorRT%20Plugin%E4%BD%BF%E7%94%A8%E6%96%B9%E5%BC%8F%E7%AE%80%E4%BB%8B-%E4%BB%A5leaky%20relu%E5%B1%82%E4%B8%BA%E4%BE%8B.md）
    //!     Theo dữ liệu được tuần tự hóa, nó được giải mã hóa thành một lớp plugin.
    //! Các thông số bắt buộc là：
    //!     Tên của plugin, thông số này rất quan trọng, nó là chứng chỉ duy nhất cho plugin nào cần giải mã
    //!     dữ liệu tuần tự hóa
    //!     độ dài của dữ liệu được tuần tự hóa
    //! \param name tên của plugin
    //! \param fc   Dữ liệu cần thiết để tuần tự hóa
    //! \return
    //! 02
    IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) throw() override;


    //!
    //! \brief deserializePlugin Deserialize the data of the onnx model read by the op into the network. Call the third constructor of the plugin class to create the plugin
    //! \param name
    //! \param serialData
    //! \param serialLength
    //! \return
    //!
    IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) throw() override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_INSTANCE_NORMALIZATION_PLUGIN_H
