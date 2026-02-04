#include <cstddef>
#include <cmath>
#include <memory>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
        static_cast<float *>(tensor->DataPtr())[idx] += rate * static_cast<const float *>(gradient->DataPtr())[idx];
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    int64_t num_elements = grad->NumElements();

    const float *grad_ptr = static_cast<const float *>(grad->DataPtr());
    float *param_ptr = static_cast<float *>(param->DataPtr());
    float *m_ptr = static_cast<float *>(m->DataPtr());
    float *v_ptr = static_cast<float *>(v->DataPtr());

    // 计算偏差修正因子
    float m_correction = 1.0f - std::pow(beta1, t);
    float v_correction = 1.0f - std::pow(beta2, t);

    for (int64_t i = 0; i < num_elements; ++i) {
        float g = grad_ptr[i];

        // 更新一阶矩估计 (动量)
        m_ptr[i] = beta1 * m_ptr[i] + (1.0f - beta1) * g;

        // 更新二阶矩估计 (RMS)
        v_ptr[i] = beta2 * v_ptr[i] + (1.0f - beta2) * g * g;

        // 偏差修正
        float m_hat = m_ptr[i] / m_correction;
        float v_hat = v_ptr[i] / v_correction;

        // 更新参数
        param_ptr[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
