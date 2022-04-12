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
    //! \brief configurePlugin 判断输入和输出的数据类型是否正确。也可以通过这个配置信息告诉TensorRT去选择合适的算法来调优这个模型——貌似也负责计算相关的中间变量
    //!     我们一般写的plugin执行代码都是固定的，不要调优，所以这个主要是针对官方的op
    //! \param in
    //! \param nbInputs
    //! \param out
    //! \param nbOutputs
    //!
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                         const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) throw() override;


    //!
    //! \brief getSerializationSize 返回序列化(serialize)该op时需要写多少字节到buffer中
    //!     一般为权重+参数的总的字节数
    //! \return
    //!
    size_t getSerializationSize() const throw() override;


    //!
    //! \brief serialize 根据序列化大小getSerializationSize()，把需要用到的数据按照顺序序列化到buffer中
    //!     就是指权重+参数+额外的内存空间(应该是中间变量吧)序列化到buffer中
    //! \param buffer
    //!
    void serialize(void* buffer) const throw() override;


    // DynamicExt plugin supportsFormat update.
    //!
    //! \brief supportsFormatCombination TensorRT调用该方法来判断pos索引对应的(输入/输出)是否支持inOut[pos].format和inOut[pos].type指定的格式和数据类型
    //!     知乎有说，但是每读懂
    //! 暂且认为判断输入/输出的格式和数据类型是否满足要求
    //!     数据类型指DataType：float, half[float16], int 8, int32, bool
    //!     格式指TensorFormat：kLINEAR，kNCHW，kNCHW2，等（暂时不懂具体啥区别）
    //! \param pos
    //! \param inOut
    //! \param nbInputs
    //! \param nbOutputs
    //! \return
    //!
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) throw() override;


    //!
    //! \brief attachToContext 如果这个op使用到了一些其他东西，例如cublas handle，可以直接借助TensorRT内部提供的cublas handle
    //!     暂时不清楚如何使用
    //! \param cudnn
    //! \param cublas
    //! \param allocator
    //!
    void attachToContext(cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) throw() override;

    //!
    //! \brief getPluginType 自己设置该plugin的名字和版本号（比如leakyrelu 1)
    //!     注意：由于PluginNamespace一般默认为"",所以这里的plugin必须不能重复，否则编译报错(????待测试）
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
//    const char* mPluginNamespace;   //! 该plugin的namepace名字，一般不设置，为""即可
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
    //! \brief getPluginName 对应Plugin插件类中的getPluginType，getPluginVersion。 一模一样
    //! \return
    //!
    const char* getPluginName() const throw() override;
    const char* getPluginVersion() const throw() override;


    //!
    //! \brief getFieldNames
    //!     返回PluginFieldCollection结构数据，包含添加插件的参数名和类型
    //! \param PluginFieldCollection mFC: 这是成员变量
    //!     主要作用是传递这个op所需要的权重和参数，在engine推理时不会使用，而在parse中使用（比如caffe2trt,onnx2trt），决定了是否可以解析成功。
    //! 当使用parse解析这个op时，这个op的权重和参数会经历Models-->TensorRT engine --> TensorRT runtime的过程。
    //!     具体过程参考知乎链接
    //! \return
    //!
    const PluginFieldCollection* getFieldNames() throw() override;


    //!
    //! \brief createPlugin 通过得到的PluginFieldCollection去创建一个plugin，将op需要的权重和参数取出来，然后调用插件类的第一个构造函数，来创建plugin
    //!
    //! 另一种理解（参考https://github.com/LitLeo/TensorRT_Tutorial/blob/master/blogs/TensorRT%20Plugin%E4%BD%BF%E7%94%A8%E6%96%B9%E5%BC%8F%E7%AE%80%E4%BB%8B-%E4%BB%A5leaky%20relu%E5%B1%82%E4%B8%BA%E4%BE%8B.md）
    //!     根据序列化数据，反序列化为plugin类。
    //! 需要的参数有：
    //!     plugin的名字，该参数非常重要，是反序列化为哪种plugin的唯一凭证
    //!     序列化数据
    //!     序列化后的数据的长度
    //! \param name 该plugin的名字
    //! \param fc   序列化所需的数据
    //! \return
    //! 02
    IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) throw() override;


    //!
    //! \brief deserializePlugin 将op读取到的onnx模型的data数据反序列化到network中。调用插件类的第三个构造函数，来创建plugin
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
