#include "example/common/tiny_shakespeare_dataset.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using TinyShakespeareType = TinyShakespeareDataset::TinyShakespeareType;
using TinyShakespeareFile = TinyShakespeareDataset::TinyShakespeareFile;

const std::unordered_map<int, TinyShakespeareType> kTypeMap = {
    {20240520, TinyShakespeareType::kUINT16}, // GPT-2
    {20240801, TinyShakespeareType::kUINT32}, // LLaMA 3
};

const std::unordered_map<TinyShakespeareType, size_t> kTypeToSize = {
    {TinyShakespeareType::kUINT16, 2},
    {TinyShakespeareType::kUINT32, 4},
};

const std::unordered_map<TinyShakespeareType, DataType> kTypeToDataType = {
    {TinyShakespeareType::kUINT16, DataType::kUINT16},
    {TinyShakespeareType::kUINT32, DataType::kINT32},
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

TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    /*
     ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token数据           |
    ----------------------------------------------------------------------------------
    */
    TinyShakespeareFile file;

    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open file: " << path;

    // 读取 Header (1024 bytes)
    auto header = ReadSeveralBytesFromIfstream(1024, &ifs);

    // 解析 magic (4 bytes)
    uint32_t magic = BytesToType<uint32_t>(header, 0);

    // 解析 version (4 bytes, offset 4)
    uint32_t version = BytesToType<uint32_t>(header, 4);

    // 解析 num_toks (4 bytes, offset 8)
    uint32_t num_toks = BytesToType<uint32_t>(header, 8);

    // 根据 magic 确定数据类型
    CHECK(kTypeMap.count(magic)) << "Unknown magic number: " << magic;
    file.type = kTypeMap.at(magic);

    size_t token_size = kTypeToSize.at(file.type);
    size_t data_size = num_toks * token_size;

    // 读取 token 数据
    std::vector<uint8_t> data_bytes(data_size);
    ifs.read(reinterpret_cast<char *>(data_bytes.data()), data_size);

    // 计算样本数
    size_t num_samples = num_toks / sequence_length - 1;

    // 创建 Tensor: [num_samples, sequence_length, 2]
    // 最后一维 2 表示 [input_token, target_token]
    // 使用 int64 类型，因为 GPT-2 的 Embedding 层期望 int64 输入
    file.dims = {static_cast<int64_t>(num_samples),
                 static_cast<int64_t>(sequence_length),
                 2};

    file.tensor = infini_train::Tensor(file.dims, infini_train::DataType::kINT64,
                                       infini_train::Device(infini_train::DeviceType::kCPU, 0));

    // 填充数据
    int64_t *tensor_data = static_cast<int64_t *>(file.tensor.DataPtr());
    if (file.type == TinyShakespeareType::kUINT16) {
        const uint16_t *token_data = reinterpret_cast<const uint16_t *>(data_bytes.data());
        for (size_t sample = 0; sample < num_samples; ++sample) {
            for (size_t pos = 0; pos < sequence_length; ++pos) {
                size_t idx = sample * sequence_length + pos;
                tensor_data[(sample * sequence_length + pos) * 2 + 0] = static_cast<int64_t>(token_data[idx]);
                tensor_data[(sample * sequence_length + pos) * 2 + 1] = static_cast<int64_t>(token_data[idx + 1]);
            }
        }
    } else {
        const uint32_t *token_data = reinterpret_cast<const uint32_t *>(data_bytes.data());
        for (size_t sample = 0; sample < num_samples; ++sample) {
            for (size_t pos = 0; pos < sequence_length; ++pos) {
                size_t idx = sample * sequence_length + pos;
                tensor_data[(sample * sequence_length + pos) * 2 + 0] = static_cast<int64_t>(token_data[idx]);
                tensor_data[(sample * sequence_length + pos) * 2 + 1] = static_cast<int64_t>(token_data[idx + 1]);
            }
        }
    }

    LOG(INFO) << "Loaded TinyShakespeare dataset: " << path
              << ", num_toks=" << num_toks
              << ", num_samples=" << num_samples
              << ", type=" << (file.type == TinyShakespeareType::kUINT16 ? "uint16" : "uint32");

    return file;
}
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    : sequence_length_(sequence_length),
      text_file_(ReadTinyShakespeareFile(filepath, sequence_length)),
      sequence_size_in_bytes_(sequence_length * (text_file_.type == TinyShakespeareType::kUINT16 ? 2 : 4)),
      num_samples_(text_file_.dims[0]) {
}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, text_file_.dims[0] - 1);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    // x: (seq_len), y: (seq_len) -> stack -> (bs, seq_len) (bs, seq_len)
    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, dims),
            std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_ + sizeof(int64_t),
                                                   dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
