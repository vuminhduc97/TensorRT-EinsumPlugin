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
#include "Einsum.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

using namespace nvinfer1;
using nvinfer1::plugin::Einsum;
using nvinfer1::plugin::EinsumCreator;


PluginFieldCollection EinsumCreator::mFC{};
std::vector<PluginField> EinsumCreator::mPluginAttributes;
REGISTER_TENSORRT_PLUGIN(EinsumCreator);
using namespace nvinfer1::plugin;
// int main(int argc, char** argv)
// {
//     return 0;
// }

namespace
{
constexpr const char* INSTANCE_PLUGIN_VERSION{"1"};
constexpr const char* INSTANCE_PLUGIN_NAME{"Einsum"}; //! Lúc này, plugin tương ứng được gọi là CustomEinsum_TRT
} // namespace
#define CHECK_CUDA(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

#define CHECK_CUDNN(call)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        cudnnStatus_t status = call;                                                                                   \
        if (status != CUDNN_STATUS_SUCCESS)                                                                            \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

inline bool is_CHW(nvinfer1::Dims const& dims)
{
    return true;
//    (dims.nbDims == 3 && dims.type[0] == nvinfer1::DimensionType::kCHANNEL
//        && dims.type[1] == nvinfer1::DimensionType::kSPATIAL && dims.type[2] == nvinfer1::DimensionType::kSPATIAL);
}


/**
 * @brief Einsum::~Einsum Giải phóng bộ nhớ video bị chiếm bởi op
 */
Einsum::~Einsum(){
//    std::cout<< "析构plugin类\t IN ~Plugin" << std::endl;
    terminate();
}
Einsum::Einsum(std::string equation)
    :equation(equation)
{
    std::cout << "Construct Plugin \ t TRONG hàm tạo đầu tiên, được sử dụng trong giai đoạn parse" << std::endl;
}

Einsum::Einsum(std::string equation, int N, int K, int C, int T, int V, int W)
    :equation(equation),N(N),K(K),C(C),T(T),V(V),W(W)
{
//    std::cout << "Construct Plugin\t IN hàm tạo thứ hai, cho clone" << std::endl;
}
//! Đọc dữ liệu khi deserialization
Einsum::Einsum(void const* serialData, size_t serialLength)
{
//    std::cout << "Construct Plugin \ t IN constructor thứ ba để sử dụng deserialization" << std::endl;
    const char *d = reinterpret_cast<const char*>(serialData), *a = d;
    equation = read<std::string>(d);
    N = read<int>(d);
    K = read<int>(d);
    C = read<int>(d);
    T = read<int>(d);
    V = read<int>(d);
    W = read<int>(d);
}


// Einsum returns one output.
int Einsum::getNbOutputs() const throw()
{
    std::cout << "nhận được số lượng đầu ra\t IN getNbOutpus" << std::endl;
    return 1;
}

DimsExprs Einsum::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) throw()
{
    std::cout << "Tính toán kích thước đầu ra\t IN Plugin::getOutputDimensions" << std::endl;
    //! Đây là kết quả trả về trực tiếp của batch đầu vào
    nvinfer1::DimsExprs output;
    if(equation == "nctkv,kvw->nctw"){
        output.nbDims = 4; //! D của DimsExprs là một mảng 8 chiều cố định, vì vậy nbDims phải được sử dụng để đánh dấu một số kích thước
        output.d[0] = inputs[0].d[0];
        output.d[1] = inputs[0].d[1];
        output.d[2] = inputs[0].d[2];
        output.d[3] = inputs[1].d[2];
    }
    return output;
}

int Einsum::initialize() throw()
{
    std::cout << "Khởi tạo lớp plugin\t IN initialize" << std::endl;
    return 0;
}

void Einsum::terminate() throw()
{
}

size_t Einsum::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const throw()
{   //! Kích thước sử dụng bộ nhớ video ở đây do chính bạn ước tính
    //! Tính dung lượng bộ nhớ video trung gian mà bạn nghĩ op này cần trong quá trình chuyển tiếp (do chính bạn thiết lập)
    std::cout << "Nhận kích thước không gian làm việc\t IN getWorkspaceSize" << std::endl;
    size_t need_num = 0;
    size_t res = need_num * sizeof(float); //! Tính toán không gian bộ nhớ bị chiếm bởi số lượng tham số này
    return res;
}


int Einsum::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,const nvinfer1::PluginTensorDesc* outputDesc,
                            const void* const* inputs, void* const* outputs,
                            void* workspace,
                            cudaStream_t stream) throw()
{
    // printf("error code enter plugin::enqueue %d\n", (int)cudaGetLastError());
    std::cout << "bắt đầu infer\t IN enqueue" << endl;
    const float* x = reinterpret_cast<const float*>(inputs[0]); //! Ở đây inputs[0] không bằng inputs[1]-45
    const float* A = reinterpret_cast<const float*>(inputs[1]);

    if(false){ // debug Khi được sử dụng, nó được sử dụng để in thông tin đầu vào và đầu ra của infer plugin hiện tại
        float* x1 = (float*)malloc(sizeof(float)*N*C*T*K*V);
        float* A1 = (float*)malloc(sizeof(float)*K*V*W);
        cudaMemcpy(x1,x,sizeof(float)*N*C*T*K*V,cudaMemcpyDeviceToHost);
        cudaMemcpy(A1,A,sizeof(float)*K*V*W,cudaMemcpyDeviceToHost);
        float A_sum = 0,x_sum=0;
        std::cout << std::endl << "ma trận kề A";
        for(int i=0; i<K*V*W; i++){
    //        printf("%10.3f",A1[i]);
            A_sum += A1[i];
        }
        std::cout << std::endl << "Nhập X";
        for(int i=0; i<N*C*T*K*V; i++){
            printf("%10.3f",x1[i]);
            x_sum += x1[i];
        }
        std::cout << std::endl;
        printf("x_sum: %10.3f\tA_sum: %10.3f\n", x_sum,A_sum);
        // in đầu vào
        printf("Kích thước đầu vào x là：\t");
        for(int i = 0; i < inputDesc[0].dims.nbDims; ++i){
            std::cout << inputDesc[0].dims.d[i] << ' ';
        }
        printf("Số chiều của ma trận kề của A là：\t");
        for(int i = 0; i < inputDesc[1].dims.nbDims; ++i){
            std::cout << inputDesc[1].dims.d[i] << ' ';
        }
        std::cout << std::endl;
    }
//    float* output0 = reinterpret_cast<float*>(outputs[0]);
    cublasHandle_t mCublas; //! tạo cublas
    cublasCreate(&mCublas); //! Hai dòng này là cố định, không cần quan tâm
    float onef{ 1.0f }, zerof{ 0.0f };
    cublasSetStream(mCublas, stream);
    if(equation == "nctkv,kvw->nctw"){
        //! nct,kv * kv,w --> nctw
        //! Phép nhân ma trận X*A=(AT*XT)T
        cublasSgemm(mCublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    W, N*C*T, K*V,
                    &onef,
                    reinterpret_cast<const float*>(inputs[1]), W,
                    reinterpret_cast<const float*>(inputs[0]), K*V,
                    &zerof,
                    reinterpret_cast<float*>(outputs[0]), W
                    );
    }

    cublasDestroy(mCublas);
    // printf("error code leave plugin::enqueue %d\n", (int)cudaGetLastError());
    return 0;
}

size_t Einsum::getSerializationSize() const throw()
{
    std::cout << "Nhận kích thước dữ liệu được serialized\t IN getSerializationSize" << std::endl;
//    return (serialized_size(equation) +
//            serialized_size(N) * 6
//            );
    return sizeof(equation) + sizeof(N) * 6; //! Câu trên không dùng được sẽ báo lỗi, hiện tại chưa rõ lý do.
}

//! Lưu trữ các biến trung gian, tương ứng với deserialize
void Einsum::serialize(void* buffer) const throw()
{
    std::cout << "dữ liệu serialize \t IN serialize" << std::endl;
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, equation);
    write(d, N);
    write(d, K);
    write(d, C);
    write(d, T);
    write(d, V);
    write(d, W);
}

//! Phát hiện xem kiểu dữ liệu và định dạng plugin có đáp ứng yêu cầu hay không (tự thiết kế)
/*  Ở đây hoàn toàn không thể trả về true trực tiếp vì đầu vào của hàm này không chỉ là đầu vào của lớp này mà có thể là đầu vào khác, 
nếu trả về true mà không có phán đoán thì lỗi tiếp theo phải là
Được khám phá bằng thực nghiệm:
    .Định dạng có thể phân biệt liệu dữ liệu đầu vào có phải là dữ liệu đầu vào của lớp do chính nó chỉ định hay không, 
    để trả về giá trị false cho dữ liệu không được lớp đầu vào
*/
bool Einsum::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) throw()
{
    // std::cout << "Xác định xem đầu vào và đầu ra có đáp ứng các yêu cầu hay không \t IN supportsFormatCombination" << std::endl;
    ASSERT(inOut && pos < (nbInputs + nbOutputs));
    bool res = ((inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF)
                && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR && inOut[pos].type == inOut[0].type);
    std::cout << "mục lục" << pos <<"Xác định xem đầu vào và đầu ra có đáp ứng các yêu cầu hay không\t IN supportsFormatCombination\t" << res << std::endl;
    // printf("%d\n", inOut[pos].format == nvinfer1::PluginFormat::kLINEAR);
    return res;
    // return true;
}

const char* Einsum::getPluginType() const throw()
{
    std::cout << "Nhận tên plugin\t IN getPluginType" << std::endl;
    return INSTANCE_PLUGIN_NAME;
}

const char* Einsum::getPluginVersion() const throw()
{
    std::cout << "Tải phiên bản plugin \t IN getPluginVersion" << std::endl;
    return INSTANCE_PLUGIN_VERSION;
}

void Einsum::destroy() throw()
{
    std::cout << "delete plugin class\t IN destroy" << std::endl;
    delete this;
}

IPluginV2DynamicExt* Einsum::clone() const throw()
{
    auto* plugin = new Einsum(equation, N, K, C, T, V, W);
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    return plugin;
}

// Set plugin namespace
void Einsum::setPluginNamespace(const char* pluginNamespace) throw()
{
    mPluginNamespace = pluginNamespace;
}

const char* Einsum::getPluginNamespace() const throw()
{
    return mPluginNamespace.c_str();
}

//! Trả về kiểu dữ liệu đầu ra giống với kiểu dữ liệu đầu vào
nvinfer1::DataType Einsum::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const throw()
{
    std::cout << "return output data type\t IN getOutputDataType" << endl;
//    ASSERT(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
}

void Einsum::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                         const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) throw()
{
    std::cout << "Tính toán thông tin biến trung gian\t IN Plugin::configurePlugin" << std::endl;
    N = in[0].desc.dims.d[0];
    C = in[0].desc.dims.d[1];
    T = in[0].desc.dims.d[2];
    K = in[0].desc.dims.d[3];
    V = in[0].desc.dims.d[4];
    W = in[1].desc.dims.d[2];
    cout << "N==>" << N << "\tC==>" << C << "\tT==>" << T << "\tK==>" << K << "\tV==>" << V << "\tW==>" << W << std::endl;
    cout << "K==>" << in[1].desc.dims.d[0] << "\tV==>" << in[1].desc.dims.d[1] << "\tW==>" << in[1].desc.dims.d[2] << std::endl;
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void Einsum::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) throw()
{
}

// Detach the plugin object from its execution context.
void Einsum::detachFromContext() throw()
{
}

// EinsumCreator methods
//! Tương ứng với thao tác trong createPlugin
//! khởi tạo plugin field meta data
//!     Còn hiện tại, nó được hiểu là dữ liệu mà plugin sử dụng, được lưu trữ dưới dạng key + dâta（tức là PluginField)
//! PluginField: Stored in the form of variable name + value, the data of the plugin
EinsumCreator::EinsumCreator()
{
    std::cout << "khởi tạo class plugin Creator\t IN PluginCreator" << std::endl;
    //! hiểu biết cá nhân---
    //! Lý do sử dụng "phương trình" là các ATTRIBUTES tương ứng với Einsum trong mô hình ONNX là phương trình, có thể thu được bằng cách xem thông tin của nút thông qua netron
    //! Plugin ONNX có một số ATTRIBUTES, chỉ cần thêm một số ở đây, các tên phải giống nhau và các loại dữ liệu phải tương ứng
    mPluginAttributes.emplace_back(PluginField("equation", nullptr, PluginFieldType::kCHAR, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* EinsumCreator::getPluginName() const throw()
{
    return INSTANCE_PLUGIN_NAME;
}

const char* EinsumCreator::getPluginVersion() const throw()
{
    return INSTANCE_PLUGIN_VERSION;
}

const PluginFieldCollection* EinsumCreator::getFieldNames() throw()
{
    std::cout << "Nhận thông tin plugin\t IN getFieldNames" << std::endl;
    return &mFC;
}

//!
//! Có thể coi đây là phần mở đầu của toàn bộ phân tích Tạo một lớp plugin lớp trong hàm này, truyền tham số phương trình lớp cho nó -> rồi thực hiện các thao tác khác nhau trong lớp plugin
//! \brief EinsumCreator::createPlugin Từ dữ liệu được parse(dữ liệu thuộc loại PluginFieldCollection), tạo một class plugin thêm (Einsum)
//! \param name
//! \param fc
//! \return
//!
IPluginV2DynamicExt* EinsumCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) throw()
{
    std::cout << "parse dữ liệu ONNX và xây dựng lớp plugin\t IN PluginCreator::createPlugin" << std::endl;
    const char* mequation;
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i) {
        const char* attrName = fields[i].name; //! read data name
        if (!strcmp(attrName, "equation")) {
            // assert(fields[i].type == PluginFieldType::kCHAR);
            // strcpy(mequation,*(static_cast<const std::string*>(fields[i].data)));
            mequation = static_cast<const char*>(fields[i].data); //! read data
        }
    }
    std::cout << "equation is " << mequation << std::endl;
    //! create plugin
    Einsum* obj = new Einsum(mequation);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2DynamicExt* EinsumCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) throw()
{
    Einsum* obj = new Einsum(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
