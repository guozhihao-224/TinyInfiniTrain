# TinyInfiniTrain 作业报告

## 一、test 通过截图

## 二、作业步骤

> 将代码填入下面代码块中指定位置，并详细描述完成该作业的解决思路和遇到的问题。

### 作业一：autograd机制调用Neg kernel的实现

难度：⭐

对应测例：`TEST(ElementwiseTest, NegForward)`，`TEST(ElementwiseTest, NegBackward)`

需要实现的代码块位置：`infini_train/src/autograd/elementwise.cc`

```c++
std::vector<std::shared_ptr<Tensor>> Neg::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    // =================================== 作业 ===================================
    // TODO：通过Dispatcher获取设备专属kernel，对输入张量进行取反操作
    // HINT: 依赖test_dispatcher，kernel实现已给出
    // =================================== 作业 ===================================
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "NegForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input)};
}

std::vector<std::shared_ptr<Tensor>> Neg::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    // =================================== 作业 ===================================
    // TODO：通过Dispatcher获取设备专属的反向传播kernel，计算梯度
    // HINT: 依赖test_dispatcher，kernel实现已给出
    // =================================== 作业 ===================================
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "NegBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output)};
}
```

#### 解决思路



#### 遇到问题



### 作业二：实现矩阵乘法

难度：⭐⭐

#### CPU实现

对应测例：`TEST(MatmulTest, BasicMatrixMultiply)`，`TEST(MatmulTest, BatchedMatrixMultiply)`, `TEST(MatmulTest, BackwardPass)`

需要实现的代码块位置：`infini_train/src/kernels/cpu/linear.cc`

```c++
    std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
        // =================================== 作业 ===================================
        // TODO：实现CPU上的矩阵乘法前向计算
        // REF:
        // =================================== 作业 ===================================
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
        // =================================== 作业 ===================================
        // TODO：实现CPU上的矩阵乘法反向传播
        // REF:
        // =================================== 作业 ===================================
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
```

#### CUDA实现

对应测例：`TEST(MatmulTest, BasicMatrixMultiplyCuda)`,`TEST(MatmulTest, BatchedMatrixMultiplyCuda)`,`TEST(MatmulTest, BackwardPassCuda)`

需要实现的代码块位置：`infini_train/src/kernels/cuda/linear.cu`

```c++
    std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
        // =================================== 作业 ===================================
        // TODO：实现CUDA上的矩阵乘法前向计算
        // REF:
        // =================================== 作业 ===================================
    }

    std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
        MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
                    const std::shared_ptr<Tensor> &grad_output) {
        // =================================== 作业 ===================================
        // TODO：实现CUDA上的矩阵乘法反向传播
        // REF:
        // =================================== 作业 ===================================
    }
```

#### 解决思路



#### 遇到问题



### 作业三：实现Adam优化器

难度：⭐

#### CPU实现

对应测例：`TEST(AdamOptimizerTest, BasicParameterUpdate)`,`TEST(AdamOptimizerTest, MomentumAccumulation)`

代码位置：infini_train/src/kernels/cpu/accumulate_grad.cc

```c++
void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF: 
    // =================================== 作业 ===================================
}
```

#### CUDA实现

对应测例：`TEST(AdamOptimizerTest, BasicParameterUpdateCuda)`,`TEST(AdamOptimizerTest, MomentumAccumulationCuda)`

代码位置：infini_train/src/kernels/cuda/accumulate_grad.cu

```c++
void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF: 
    // =================================== 作业 ===================================
}
```

#### 解决思路



#### 遇到问题



### 作业四：实现Tensor基础操作

#### 实现Tensor的Flatten操作

难度：⭐

对应测例：`TEST(TensorTransformTest, Flatten2DTo1D)`,`TEST(TensorTransformTest, FlattenWithRange) `,`TEST(TensorTransformTest, FlattenNonContiguous)`

代码位置：infini_train/src/tensor.cc

```c++
std::shared_ptr<Tensor> Tensor::Flatten(int64_t start, int64_t end) {
    // =================================== 作业 ===================================
    // TODO：实现张量扁平化操作，将指定维度范围[start, end]内的所有维度合并为一个维度
    // HINT: 
    // =================================== 作业 ===================================
     // 处理负索引
    int64_t ndim = dims_.size();
    if (start < 0) {
        start += ndim;
    }
    if (end < 0) {
        end += ndim;
    }

    // 边界检查
    CHECK_GE(start, 0);
    CHECK_LT(start, ndim);
    CHECK_GE(end, 0);
    CHECK_LT(end, ndim);
    CHECK_LE(start, end);

    // 计算新的 shape
    std::vector<int64_t> new_shape;

    // 添加 start 之前的维度
    for (int64_t i = 0; i < start; ++i) {
        new_shape.push_back(dims_[i]);
    }

    // 计算合并后的维度大小
    int64_t flattened_dim = 1;
    for (int64_t i = start; i <= end; ++i) {
        flattened_dim *= dims_[i];
    }
    new_shape.push_back(flattened_dim);

    // 添加 end 之后的维度
    for (int64_t i = end + 1; i < ndim; ++i) {
        new_shape.push_back(dims_[i]);
    }

    // 先确保内存连续，然后使用 View
    return Contiguous()->View(new_shape);
}
```

#### 实现Tensor的反向传播机制

难度：⭐

对应测例：`TEST(TensorAutogradTest, BackwardComputesGradient)`,`TEST(TensorAutogradTest, BackwardWithMultipleOutputs)`

代码位置：infini_train/src/tensor.cc

```c++
void Tensor::Backward(std::shared_ptr<Tensor> gradient, bool retain_graph, bool create_graph) const {
    // =================================== 作业 ===================================
    // TODO：实现自动微分反向传播
    // 功能描述：1. 计算当前张量对叶子节点的梯度    2. 支持多输出场景的梯度累加
    // HINT: 
    // =================================== 作业 ===================================
     // 如果没有梯度函数且不需要梯度，直接返回
    if (!grad_fn_ && !requires_grad_) {
        return;
    }

    // 如果没有提供梯度，创建全 1 的梯度（对标量）
    if (!gradient) {
        gradient = std::make_shared<Tensor>(dims_, dtype_, GetDevice());
        gradient->Fill<float>(1.0f);
    }

    // 用于拓扑排序的数据结构
    std::queue<std::shared_ptr<Tensor>> q;
    std::unordered_set<std::shared_ptr<Tensor>> visited;
    std::unordered_map<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> grad_map;

    // 初始化：将当前张量加入队列
    auto self = const_cast<Tensor*>(this)->shared_from_this();
    q.push(self);
    visited.insert(self);
    grad_map[self] = gradient;

    // BFS 遍历计算图
    while (!q.empty()) {
        auto current = q.front();
        q.pop();

        // 获取当前张量的梯度
        auto current_grad = grad_map[current];

        // 如果是叶子节点，累加梯度
        if (current->is_leaf()) {
            if (current->grad_) {
                // 累加梯度: grad = grad + current_grad
                current->grad_ = current->grad_->Add(current_grad);
            } else {
                current->grad_ = std::make_shared<Tensor>(*current_grad);
            }
            continue;
        }

        // 如果有梯度函数，调用反向传播
        if (current->grad_fn_) {
            auto input_grads = current->grad_fn_->Backward({current_grad});

            // 将梯度传递给输入张量
            const auto& inputs = current->grad_fn_->saved_tensors();
            for (size_t i = 0; i < inputs.size() && i < input_grads.size(); ++i) {
                if (inputs[i]->requires_grad_) {
                    auto& input = inputs[i];

                    // 累加梯度（多输出场景）
                    if (grad_map.count(input)) {
                        grad_map[input] = grad_map[input]->Add(input_grads[i]);
                    } else {
                        grad_map[input] = input_grads[i];
                    }

                    // 如果未访问过，加入队列
                    if (!visited.count(input)) {
                        visited.insert(input);
                        q.push(input);
                    }
                }
            }
        }
    }
}
```

#### 解决思路



#### 遇到问题



### 作业五 注册算子kernel的实现

难度：⭐⭐⭐

对应测例：`TEST(DispatcherTest, RegisterAndGetKernel)`,`TEST(DispatcherTest, DuplicateRegistration)`,`TEST(DispatcherTest, GetNonexistentKernel)`

代码位置：infini_train/include/dispatcher.h

```c++
template <typename RetT, class... ArgsT> RetT Call(ArgsT... args) const {
    // =================================== 作业 ===================================
    // TODO：实现通用kernel调用接口
    // 功能描述：将存储的函数指针转换为指定类型并调用
    // HINT: 
    // =================================== 作业 ===================================

    using FuncT = RetT (*)(ArgsT...);
        auto func = reinterpret_cast<FuncT>(func_ptr_);
        CHECK(func != nullptr) << "Kernel function pointer is null";
        return func(std::forward<ArgsT>(args)...);
}

template <typename FuncT> void Register(const KeyT &key, FuncT &&kernel) {
    // =================================== 作业 ===================================
    // TODO：实现kernel注册机制
    // 功能描述：将kernel函数与设备类型、名称绑定
    // =================================== 作业 ===================================

    CHECK(!key_to_kernel_map_.contains(key))
            << "Kernel already registered: " << key.second << " on device: " << static_cast<int>(key.first);
        key_to_kernel_map_.emplace(key, KernelFunction(std::forward<FuncT>(kernel)));
}

#define INF_TINYINF_TRAIN_CONCAT_INNER(a, b) a##b
#define INF_TINYINF_TRAIN_CONCAT(a, b) INF_TINYINF_TRAIN_CONCAT_INNER(a, b)
#define REGISTER_KERNEL(device, kernel_name, kernel_func) \
    // =================================== 作业 ===================================
    // TODO：实现自动注册宏
    // 功能描述：在全局静态区注册kernel，避免显式初始化代码
    // =================================== 作业 ===================================
    const auto INF_TINYINF_TRAIN_CONCAT(_kernel_reg_, __COUNTER__) = []() {                                            \
        ::infini_train::Dispatcher::Instance().Register({(device), #kernel_name}, (kernel_func));                      \
        return 0;                                                                                                      \
    }();
    

```

#### 解决思路



#### 遇到问题



### 作业六：实现GPT-2整体训练

难度：⭐⭐⭐⭐

对应测例：`TEST_F(GPT2TrainingTest, LogitsConsistency)`

#### 训练过程logits对比

完成以上所有作业，补齐训练框架的所有实现，理论上`TEST_F(GPT2TrainingTest, LogitsConsistency)`可以通过，在用例中判断比较预置的值和单步正向传播计算结果是否在误差允许范围内相等。

#### 数据读取实现

代码位置：example/common/tiny_shakespeare_dataset.cc

```c++
TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    /* =================================== 作业 ===================================
       TODO：实现二进制数据集文件解析
       文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token数据           |
    ----------------------------------------------------------------------------------
       =================================== 作业 =================================== */
}

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length) {
    // =================================== 作业 ===================================
    // TODO：初始化数据集实例
    // HINT: 调用ReadTinyShakespeareFile加载数据文件
    // =================================== 作业 ===================================
}
```

#### Tokenizer功能实现

代码位置：example/common/tokenizer.cc

```c++
Tokenizer::Tokenizer(const std::string &filepath) {
    /* ===================================== 作业 =====================================
    TODO：实现Tokenizer二进制文件加载

    文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | VOCAB TABLE                           |
    | magic(4B) | version(4B) | vocab_size(4B) | reserved(1012B) | token词表数据       |
    ----------------------------------------------------------------------------------
    ===================================== 作业 ===================================== */
}
```

```c++
std::string Tokenizer::Decode(uint32_t token_id) const {
    /* ===================================== 作业 =====================================
    TODO：实现token_id到文本的转换
    功能描述：根据token_id返回对应的文本片段
    ===================================== 作业 ===================================== */
}
```

```c++
void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    /* ...原代码... */
    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        /* ===================================== 作业 =====================================
        TODO：实现单步文本生成逻辑
        HINT：调用model.Forward推理获取logits，根据推理结果进行随机采样，调用Decode获取文本结果
        ===================================== 作业 ===================================== */
    }
    std::cout << std::endl;
}
```

#### 解决思路



#### 遇到问题

