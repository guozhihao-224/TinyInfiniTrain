#include "example/common/tokenizer.h"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/functional.h"

namespace infini_train {

constexpr uint32_t kGpt2Eot = 50256;
constexpr uint32_t kLLaMA3Eot = 128001;
constexpr uint64_t kRandomU32Multiplier = 0x2545F4914F6CDD1Dull;
constexpr float kF32Divisor = 16777216.0f; // 2^24
constexpr uint64_t kRngState = 1337;

using Version = Tokenizer::Version;

const std::unordered_map<uint32_t, uint32_t> kEotMap = {
    {20240328, kGpt2Eot},   // GPT-2
    {20240801, kLLaMA3Eot}, // LLaMA-3
};

const std::unordered_map<uint32_t, std::vector<uint32_t>> kPromptMap = {
    // e.g. "The meaning of life is"
    // ref: https://tiktokenizer.vercel.app/
    {20240328, std::vector<uint32_t>{464, 3616, 286, 1204, 318}}, // GPT-2
    {20240801, std::vector<uint32_t>{791, 7438, 315, 2324, 374}}, // LLaMA-3
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

unsigned int RandomU32(uint64_t &state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (state * kRandomU32Multiplier) >> 32;
}

float RandomF32(uint64_t &state) { // random float32 in [0,1)
    return (RandomU32(state) >> 8) / kF32Divisor;
}

int SampleMult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from RandomF32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

Tokenizer::Tokenizer(const std::string &filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open tokenizer file: " << filepath;

    // 读取 Header (1024 bytes)
    auto header = ReadSeveralBytesFromIfstream(1024, &ifs);

    // 解析 magic
    magic_number_ = BytesToType<uint32_t>(header, 0);

    // 解析 version
    uint32_t version = BytesToType<uint32_t>(header, 4);

    // 解析 vocab_size
    vocab_size_ = BytesToType<uint32_t>(header, 8);

    // 根据 magic 确定 EOT token
    if (kEotMap.count(magic_number_)) {
        eot_token_ = kEotMap.at(magic_number_);
    } else {
        LOG(WARNING) << "Unknown magic number, using default EOT token";
        eot_token_ = 50256; // GPT-2 default
    }

    // 读取词表
    token_table_.resize(vocab_size_);

    for (uint32_t i = 0; i < vocab_size_; ++i) {
        // 读取 token 长度 (1 byte)
        uint8_t len;
        ifs.read(reinterpret_cast<char *>(&len), 1);

        // 读取 token 内容
        std::string token(len, '\0');
        ifs.read(token.data(), len);

        token_table_[i] = token;
    }

    LOG(INFO) << "Loaded tokenizer: " << filepath
              << ", vocab_size=" << vocab_size_
              << ", magic=" << magic_number_;
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    CHECK_LT(token_id, token_table_.size())
        << "Token ID " << token_id << " out of range [0, " << token_table_.size() << ")";
    return token_table_[token_id];
}

void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    std::vector<int64_t> dims;
    dims.assign({batch_size, sequence_length});
    // x_tensor (FLAGS_batch_size, FLAGS_sequence_length) eq:(4, 64)
    infini_train::Tensor x_tensor = infini_train::Tensor(dims, DataType::kINT64);
    int64_t *x_buff = static_cast<int64_t *>(x_tensor.DataPtr());
    for (int i = 0; i < batch_size * sequence_length; ++i) { x_buff[i] = eot_token_; }

    // Give some contexts: "The meaning of life is "
    auto prompt = kPromptMap.at(magic_number_);
    auto prompt_len = prompt.size();
    for (int i = 0; i < prompt_len; ++i) { x_buff[i] = prompt[i]; }
    std::cout << "The meaning of life is";

    auto x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    uint64_t rng_state = kRngState;
    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        // 前向传播获取 logits
        auto outputs = model.Forward({x});
        auto logits = outputs[0]; // shape: [batch_size, sequence_length, vocab_size]

        // 获取最后一个位置的 logits
        auto last_logits = logits->Slice(1, t - 1, t, 1); // [batch_size, 1, vocab_size]

        // 将 logits 转换为概率（softmax）
        auto probs_tensor = nn::function::Softmax(last_logits, /*dim=*/-1);

        // 将结果转回 CPU 进行采样
        auto probs_cpu = probs_tensor->To(Device(DeviceType::kCPU, 0));
        float *probs = static_cast<float *>(probs_cpu.DataPtr());

        // 采样下一个 token（这里只取第一个 batch 的结果）
        float coin = RandomF32(rng_state);
        int next_token = SampleMult(probs, vocab_size_, coin);

        // 更新输入序列
        x_buff[t] = next_token;

        // 解码并输出
        std::string token_str = Decode(next_token);
        std::cout << token_str;
        std::cout.flush();
    }
    std::cout << std::endl;
}
} // namespace infini_train
