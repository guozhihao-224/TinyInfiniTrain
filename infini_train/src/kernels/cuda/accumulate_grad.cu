#include "cuda_runtime_api.h"
#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess) {                                                                                   \
            LOG(FATAL) << "CUDA Error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__;       \
        }                                                                                                              \
    } while (0)

__global__ void AccumulateGradKernel(const float *grad_ptr, float rate, float *tensor_ptr, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor_ptr[idx] += rate * grad_ptr[idx];
    }
}

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    size_t num_elements = gradient->NumElements();

    const float *grad_ptr = static_cast<const float *>(gradient->DataPtr());
    float *tensor_ptr = static_cast<float *>(tensor->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AccumulateGradKernel<<<num_blocks, threads_per_block>>>(grad_ptr, rate, tensor_ptr, num_elements);
}

__global__ void AdamAccumulateGradKernel(const float *grad, float *param, float *m, float *v,
                                         float learning_rate, float beta1, float beta2, float eps,
                                         float m_correction, float v_correction, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        float g = grad[idx];
        
        // 更新一阶矩估计
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        
        // 更新二阶矩估计
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        // 偏差修正
        float m_hat = m[idx] / m_correction;
        float v_hat = v[idx] / v_correction;
        
        // 更新参数
        param[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + eps);
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    size_t num_elements = grad->NumElements();
    
    // 获取数据指针
    const float *grad_ptr = static_cast<const float *>(grad->DataPtr());
    float *param_ptr = static_cast<float *>(param->DataPtr());
    float *m_ptr = static_cast<float *>(m->DataPtr());
    float *v_ptr = static_cast<float *>(v->DataPtr());
    
    // 计算偏差修正因子
    float m_correction = 1.0f - powf(beta1, t);
    float v_correction = 1.0f - powf(beta2, t);
    
    // 配置 CUDA 启动参数
    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    
    // 启动 kernel
    AdamAccumulateGradKernel<<<num_blocks, threads_per_block>>>(
        grad_ptr, param_ptr, m_ptr, v_ptr,
        learning_rate, beta1, beta2, eps,
        m_correction, v_correction, num_elements
    );
    
    // 同步确保完成
    CUDA_CHECK(cudaDeviceSynchronize());
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL
