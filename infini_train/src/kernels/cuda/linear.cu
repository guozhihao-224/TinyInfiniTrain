#include "cublas_v2.h"
#include "glog/logging.h"
#include <cub/block/block_reduce.cuh>

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

#define CUBLAS_CHECK(call)                                                                                             \
    do {                                                                                                               \
        cublasStatus_t status = call;                                                                                  \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                                         \
            LOG(FATAL) << "CUBLAS Error: " << cublasGetStatusString(status) << " at " << __FILE__ << ":" << __LINE__;  \
        }                                                                                                              \
    } while (0)

// 朴素矩阵乘法 CUDA kernel
__global__ void MatmulForwardKernel(const float *A, const float *B, float *C, 
                                     int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();

    // 获取维度
    int64_t M = input_dims[input_dims.size() - 2];
    int64_t K = input_dims[input_dims.size() - 1];
    int64_t N = other_dims[other_dims.size() - 1];

    // 计算 batch 维度
    std::vector<int64_t> batch_dims;
    int64_t batch_size = 1;

    int input_batch_rank = input_dims.size() - 2;
    int other_batch_rank = other_dims.size() - 2;
    int max_batch_rank = std::max(input_batch_rank, other_batch_rank);

    for (int i = 0; i < max_batch_rank; ++i) {
        int64_t dim = 1;
        if (i < input_batch_rank) {
            dim = input_dims[i];
        }
        if (i < other_batch_rank) {
            if (other_dims[i] != 1 && dim != 1 && other_dims[i] != dim) {
                LOG(FATAL) << "Incompatible batch dimensions for matmul";
            }
            dim = std::max(dim, other_dims[i]);
        }
        batch_dims.push_back(dim);
        batch_size *= dim;
    }

    // 创建输出张量
    std::vector<int64_t> output_dims = batch_dims;
    output_dims.push_back(M);
    output_dims.push_back(N);
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32, input->GetDevice());
    output->Fill<float>(0.0f);

    // 使用朴素 CUDA kernel 进行矩阵乘法
    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    const float *other_ptr = static_cast<const float *>(other->DataPtr());
    float *output_ptr = static_cast<float *>(output->DataPtr());

    int64_t input_stride = M * K;
    int64_t other_stride = K * N;
    int64_t output_stride = M * N;

    // 配置 CUDA kernel
    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y);

    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t input_offset = 0;
        int64_t other_offset = 0;

        if (input_batch_rank > 0) {
            input_offset = (b % batch_size) * input_stride;
        }
        if (other_batch_rank > 0) {
            other_offset = (b % batch_size) * other_stride;
        }

        // 启动 kernel
        MatmulForwardKernel<<<grid_size, block_size>>>(
            input_ptr + input_offset,
            other_ptr + other_offset,
            output_ptr + b * output_stride,
            M, N, K
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    return output;
}

// 梯度计算 CUDA kernel
__global__ void GradInputKernel(const float *dy, const float *w, float *dx, 
                                 int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += dy[row * N + k] * w[col * N + k];  // w^T
        }
        dx[row * K + col] = sum;
    }
}

__global__ void GradOtherKernel(const float *x, const float *dy, float *dw, 
                                 int K, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < K && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum += x[i * K + row] * dy[i * N + col];
        }
        dw[row * N + col] = sum;
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();

    int64_t M = input_dims[input_dims.size() - 2];
    int64_t K = input_dims[input_dims.size() - 1];
    int64_t N = other_dims[other_dims.size() - 1];

    // 计算 batch 维度
    std::vector<int64_t> batch_dims;
    int64_t batch_size = 1;

    int input_batch_rank = input_dims.size() - 2;
    int other_batch_rank = other_dims.size() - 2;
    int max_batch_rank = std::max(input_batch_rank, other_batch_rank);

    for (int i = 0; i < max_batch_rank; ++i) {
        int64_t dim = 1;
        if (i < input_batch_rank) {
            dim = input_dims[i];
        }
        if (i < other_batch_rank) {
            dim = std::max(dim, other_dims[i]);
        }
        batch_dims.push_back(dim);
        batch_size *= dim;
    }

    // 创建梯度张量
    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, grad_output->GetDevice());
    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32, grad_output->GetDevice());
    grad_input->Fill<float>(0.0f);
    grad_other->Fill<float>(0.0f);

    const float *grad_output_ptr = static_cast<const float *>(grad_output->DataPtr());
    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    const float *other_ptr = static_cast<const float *>(other->DataPtr());
    float *grad_input_ptr = static_cast<float *>(grad_input->DataPtr());
    float *grad_other_ptr = static_cast<float *>(grad_other->DataPtr());

    int64_t input_stride = M * K;
    int64_t other_stride = K * N;
    int64_t output_stride = M * N;

    // 配置 CUDA kernel
    dim3 block_size(16, 16);
    dim3 grid_size_grad_input((K + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y);
    dim3 grid_size_grad_other((N + block_size.x - 1) / block_size.x, (K + block_size.y - 1) / block_size.y);

    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t input_offset = 0;
        int64_t other_offset = 0;

        if (input_batch_rank > 0) {
            input_offset = (b % batch_size) * input_stride;
        }
        if (other_batch_rank > 0) {
            other_offset = (b % batch_size) * other_stride;
        }

        // grad_input = grad_output @ other^T
        GradInputKernel<<<grid_size_grad_input, block_size>>>(
            grad_output_ptr + b * output_stride,
            other_ptr + other_offset,
            grad_input_ptr + input_offset,
            M, K, N
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // grad_other = input^T @ grad_output
        GradOtherKernel<<<grid_size_grad_other, block_size>>>(
            input_ptr + input_offset,
            grad_output_ptr + b * output_stride,
            grad_other_ptr + other_offset,
            K, N, M
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    return {grad_input, grad_other};
}

__global__ void BiasCopyKernel(float *output, const float *bias, int bs, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bs * out_features) {
        return;
    }
    int j = idx % out_features;
    output[idx] = bias[j];
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {

    /*
        !transpose: output = input * weight + bias
        output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]

        transpose:  output = input * weight^T + bias
        output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);

    // As for cublas:
    // C = alpha * op(B) * op(A) + beta * C
    // Dimensions:
    //   input:  (bs, in_features)
    //   weight: (in_features, out_features) or (out_features, in_features) if transposed
    //   output: (bs, out_features)
    const int64_t out_features = weight_dims[transpose ? 0 : 1];

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32, input->GetDevice());

    if (bias) {
        CHECK_EQ(bias->Dims().size(), 1);
        CHECK_EQ(bias->Dims()[0], out_features);
        int threads_per_block = 256;
        int num_blocks = (bs * out_features + threads_per_block - 1) / threads_per_block;
        BiasCopyKernel<<<num_blocks, threads_per_block>>>(
            static_cast<float *>(output->DataPtr()), static_cast<const float *>(bias->DataPtr()), bs, out_features);
    } else {
        output->Fill<float>(0.0f);
    }

    const float alpha = 1.0f;
    const float beta = 1.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    if (transpose) {
        // weight is [out_features, in_features] here

        // output = input * weight.T --> output.T = weight * input.T
        // C = output.T[out_features, bs]
        // A = weight.T[in_features, out_features]
        // B = input.T[in_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, bs, in_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), in_features,
                                 static_cast<const float *>(input->DataPtr()), in_features, &beta,
                                 static_cast<float *>(output->DataPtr()), out_features));
    } else {
        // output = input * weight --> output.T =  weight.T * input.T
        // C = output.T[out_features, bs]
        // A = weight.T[out_features, in_features]
        // B = input.T[in_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, out_features, bs, in_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), out_features,
                                 static_cast<const float *>(input->DataPtr()), in_features, &beta,
                                 static_cast<float *>(output->DataPtr()), out_features));
    }
    CUBLAS_CHECK(cublasDestroy(handle));
    return output;
}

template <int BLOCK_SIZE>
__global__ void ReduceColumnsKernel(const float *__restrict__ input, float *__restrict__ output, int num_rows,
                                    int num_cols) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int row = blockIdx.x;
    float sum = 0.0f;

    for (int col = threadIdx.x; col < num_cols; col += blockDim.x) { sum += input[row * num_cols + col]; }

    float reduced = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
        output[row] = reduced;
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, grad_output->GetDevice());
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32, grad_output->GetDevice());
    grad_input->Fill<float>(0.0f);
    grad_weight->Fill<float>(0.0f);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32,
                                             grad_output->GetDevice());
        grad_bias->Fill<float>(0.0f);
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    if (transpose) {
        // weight is [out_features, in_features] here

        // d_input = d_output * weight --> d_input.T = weight.T * d_output.T
        // C = d_input.T[in_features, bs]
        // A = weight.T[in_features, out_features]
        // B = d_output.T[out_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in_features, bs, out_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), in_features,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features, &beta,
                                 static_cast<float *>(grad_input->DataPtr()), in_features));

        // d_weight = d_output.T * input --> d_weight.T = input.T * d_output
        // C = d_weight.T[in_features, out_features]
        // A = input.T[in_features, bs]
        // B = d_output.T[out_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, in_features, out_features, bs, &alpha,
                                 static_cast<const float *>(input->DataPtr()), in_features,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features, &beta,
                                 static_cast<float *>(grad_weight->DataPtr()), in_features));
    } else {
        // weight is [in_features, out_features] here

        // d_input = d_output * weight.T --> d_input.T = weight * d_output.T
        // C = d_input.T[in_features, bs]
        // A = weight.T[out_features, in_features]
        // B = d_output.T[out_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, in_features, bs, out_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), out_features,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features, &beta,
                                 static_cast<float *>(grad_input->DataPtr()), in_features));

        // d_weight = input.T * d_output --> d_weight.T = d_output.T * input
        // C = d_weight.T[out_features, in_features]
        // A = d_output.T[out_features, bs]
        // B = input.T[in_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, out_features, in_features, bs, &alpha,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features,
                                 static_cast<const float *>(input->DataPtr()), in_features, &beta,
                                 static_cast<float *>(grad_weight->DataPtr()), out_features));
    }

    // d_bias = \sum_i(i=0, bs-1) d_output[i]
    if (bias) {
        constexpr int BLOCK_SIZE = 256;
        int threads_per_block = BLOCK_SIZE;
        int num_blocks = out_features;
        ReduceColumnsKernel<BLOCK_SIZE>
            <<<num_blocks, threads_per_block>>>(static_cast<const float *>(grad_output->DataPtr()),
                                                static_cast<float *>(grad_bias->DataPtr()), out_features, bs);
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_LINEAR_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_LINEAR_KERNEL(MatmulForward)
REGISTER_CUDA_LINEAR_KERNEL(MatmulBackward)
REGISTER_CUDA_LINEAR_KERNEL(LinearForward)
REGISTER_CUDA_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CUDA_LINEAR_KERNEL
