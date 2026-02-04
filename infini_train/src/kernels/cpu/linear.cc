#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <numeric>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
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
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);
    output->Fill<float>(0.0f);

    // 执行矩阵乘法
    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    const float *other_ptr = static_cast<const float *>(other->DataPtr());
    float *output_ptr = static_cast<float *>(output->DataPtr());

    int64_t input_stride = M * K;
    int64_t other_stride = K * N;
    int64_t output_stride = M * N;

    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t input_offset = 0;
        int64_t other_offset = 0;

        if (input_batch_rank > 0) {
            input_offset = (b % batch_size) * input_stride;
        }
        if (other_batch_rank > 0) {
            other_offset = (b % batch_size) * other_stride;
        }

        const float *a = input_ptr + input_offset;
        const float *b_mat = other_ptr + other_offset;
        float *c = output_ptr + b * output_stride;

        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    sum += a[i * K + k] * b_mat[k * N + j];
                }
                c[i * N + j] = sum;
            }
        }
    }

    return output;
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
    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32);
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

    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t input_offset = 0;
        int64_t other_offset = 0;

        if (input_batch_rank > 0) {
            input_offset = (b % batch_size) * input_stride;
        }
        if (other_batch_rank > 0) {
            other_offset = (b % batch_size) * other_stride;
        }

        const float *dy = grad_output_ptr + b * output_stride;
        const float *x = input_ptr + input_offset;
        const float *w = other_ptr + other_offset;
        float *dx = grad_input_ptr + input_offset;
        float *dw = grad_other_ptr + other_offset;

        // grad_input = grad_output @ other^T
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t k = 0; k < K; ++k) {
                float sum = 0.0f;
                for (int64_t j = 0; j < N; ++j) {
                    sum += dy[i * N + j] * w[k * N + j];
                }
                dx[i * K + k] += sum;
            }
        }

        // grad_other = input^T @ grad_output
        for (int64_t k = 0; k < K; ++k) {
            for (int64_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int64_t i = 0; i < M; ++i) {
                    sum += x[i * K + k] * dy[i * N + j];
                }
                dw[k * N + j] += sum;
            }
        }
    }

    return {grad_input, grad_other};
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {
    /*
    transpose:  output = input * weight^T + bias
    output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]

    !transpose: output = input * weight + bias
    output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    const int out_features = weight_dims[transpose ? 0 : 1];

    if (bias) {
        const auto &bias_dims = bias->Dims();
        CHECK_EQ(bias_dims.size(), 1);
        CHECK_EQ(bias_dims[0], out_features);
    }

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    if (transpose) {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix().transpose();
    } else {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix();
    }

    if (bias) {
        output->EigenMatrix().rowwise() += bias->EigenVector();
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    /*
    transpose: grad_input = grad_output * weight
    grad_input[*, in_features] = grad_output[*, out_features] * weight[out_features, in_features]
    grad_weight[out_features, in_features] = grad_output[*, out_features]^T * input[*, in_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)

    !transpose: grad_input = grad_output * weight^T
    grad_input[*, in_features] = grad_output[_, out_features] * weight[in_features, out_features]^T
    grad_weight[in_features, out_features] = input[*, in_features]^T * grad_output[*, out_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32);
    }

    if (transpose) {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix();
        grad_weight->EigenMatrix() = grad_output->EigenMatrix().transpose() * input->EigenMatrix();
    } else {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix().transpose();
        grad_weight->EigenMatrix() = input->EigenMatrix().transpose() * grad_output->EigenMatrix();
    }
    if (bias) {
        grad_bias->EigenVector() = grad_output->EigenMatrix().colwise().sum();
    }

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_LINEAR_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_LINEAR_KERNEL(MatmulForward)
REGISTER_CPU_LINEAR_KERNEL(MatmulBackward)
REGISTER_CPU_LINEAR_KERNEL(LinearForward)
REGISTER_CPU_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CPU_LINEAR_KERNEL
