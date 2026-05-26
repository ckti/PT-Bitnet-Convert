#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr uint32_t GGUF_FILE_MAGIC = 0x46554747u;

constexpr uint32_t GGUF_VALUE_UINT8   = 0;
constexpr uint32_t GGUF_VALUE_INT8    = 1;
constexpr uint32_t GGUF_VALUE_UINT16  = 2;
constexpr uint32_t GGUF_VALUE_INT16   = 3;
constexpr uint32_t GGUF_VALUE_UINT32  = 4;
constexpr uint32_t GGUF_VALUE_INT32   = 5;
constexpr uint32_t GGUF_VALUE_FLOAT32 = 6;
constexpr uint32_t GGUF_VALUE_BOOL    = 7;
constexpr uint32_t GGUF_VALUE_STRING  = 8;
constexpr uint32_t GGUF_VALUE_ARRAY   = 9;
constexpr uint32_t GGUF_VALUE_UINT64  = 10;
constexpr uint32_t GGUF_VALUE_INT64   = 11;
constexpr uint32_t GGUF_VALUE_FLOAT64 = 12;

constexpr uint64_t TQ_BLOCK_SIZE = 256;
constexpr uint64_t TQ_TYPE_SIZE  = 54;

struct quant_type_info {
    uint32_t type;
    uint64_t block_size;
    uint64_t type_size;
    const char * name;
};

const quant_type_info QUANT_TYPES[] = {
    { 0, 1,   4,  "F32"     },
    { 1, 1,   2,  "F16"     },
    { 2, 32,  18, "Q4_0"    },
    { 3, 32,  20, "Q4_1"    },
    { 6, 32,  22, "Q5_0"    },
    { 7, 32,  24, "Q5_1"    },
    { 8, 32,  34, "Q8_0"    },
    { 9, 32,  40, "Q8_1"    },
    { 10, 256, 84, "Q2_K"   },
    { 11, 256, 110, "Q3_K"  },
    { 12, 256, 144, "Q4_K"  },
    { 13, 256, 176, "Q5_K"  },
    { 14, 256, 210, "Q6_K"  },
    { 15, 256, 292, "Q8_K"  },
    { 16, 256, 66, "IQ2_XXS"},
    { 17, 256, 74, "IQ2_XS" },
    { 18, 256, 98, "IQ3_XXS"},
    { 19, 256, 50, "IQ1_S"  },
    { 20, 32,  18, "IQ4_NL" },
    { 21, 256, 114, "IQ3_S" },
    { 22, 256, 82, "IQ2_S"  },
    { 23, 256, 148, "IQ4_XS"},
    { 24, 1,   1,  "I8"      },
    { 25, 1,   2,  "I16"     },
    { 26, 1,   4,  "I32"     },
    { 27, 1,   8,  "I64"     },
    { 28, 1,   8,  "F64"     },
    { 29, 256, 56, "IQ1_M"  },
    { 30, 1,   2,  "BF16"    },
    { 34, 256, 54, "TQ1_0"   },
    { 35, 256, 66, "TQ2_0"   },
    { 39, 32,  17, "MXFP4"   },
    { 40, 64,  36, "NVFP4"   },
    { 41, 128, 18, "Q1_0"    },
    { 45, 32,  16, "TQ3_1S"  },
    { 46, 32,  20, "TQ4_1S"  },
};

struct tensor_info {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type = 0;
    uint64_t offset = 0;
    uint64_t nbytes = 0;
    uint64_t row_elems = 0;
    uint64_t nrows = 0;
};

struct gguf_model {
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t metadata_count = 0;
    uint32_t alignment = 32;
    uint64_t data_start = 0;
    std::vector<tensor_info> tensors;
    std::unordered_map<std::string, size_t> tensor_index;
};

struct options {
    std::string fp_model;
    std::string tq_model;
    std::string output_model;
    std::string calib_path;
    std::string include_regex;
    std::string exclude_regex;
    uint64_t chunk_mib = 512;
    uint64_t max_capture_mib = 256;
    int qat_steps = 16;
    float learning_rate = 0.02f;
    float grad_clip = 1.0f;
    float covariance_ridge = 1.0e-4f;
    float threshold_base = 0.75f;
    uint32_t capture_stride = 1;
    int32_t n_gpu_layers = -1;
    uint32_t n_ctx = 2048;
    uint32_t n_batch = 512;
    uint32_t n_ubatch = 512;
    uint32_t max_tokens = 2048;
    int32_t n_threads = 0;
    bool dry_run = false;
    bool no_mmap = false;
    bool parse_special = true;
};

struct activation_stat {
    uint64_t n_features = 0;
    uint64_t n_feature_blocks = 0;
    uint64_t samples = 0;
    uint64_t captures = 0;
    uint64_t skipped = 0;
    std::vector<float> covariance;
};

struct capture_state {
    std::unordered_map<std::string, activation_stat *> by_weight_name;
    uint64_t max_capture_bytes = 256ull * 1024ull * 1024ull;
    uint32_t capture_stride = 1;
    uint64_t matched_nodes = 0;
    uint64_t captured_nodes = 0;
    uint64_t skipped_nodes = 0;
    uint64_t unsupported_type_nodes = 0;
    uint64_t mismatched_shape_nodes = 0;
};

struct tensor_pair {
    const tensor_info * fp = nullptr;
    const tensor_info * tq = nullptr;
    activation_stat * stat = nullptr;
};

struct llama_backend_guard {
    llama_backend_guard() { llama_backend_init(); }
    ~llama_backend_guard() { llama_backend_free(); }
};

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__)); \
    } \
} while (0)

uint64_t align_up(uint64_t value, uint64_t alignment) {
    if (alignment == 0) {
        return value;
    }
    const uint64_t rem = value % alignment;
    return rem == 0 ? value : value + alignment - rem;
}

template <typename T>
T read_pod(std::istream & in) {
    T value{};
    in.read(reinterpret_cast<char *>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("unexpected end of file");
    }
    return value;
}

void skip_bytes(std::istream & in, uint64_t n) {
    in.seekg(static_cast<std::streamoff>(n), std::ios::cur);
    if (!in) {
        throw std::runtime_error("unexpected end of file while skipping bytes");
    }
}

std::string read_string(std::istream & in) {
    const uint64_t len = read_pod<uint64_t>(in);
    std::string out;
    out.resize(static_cast<size_t>(len));
    if (len > 0) {
        in.read(out.data(), static_cast<std::streamsize>(len));
        if (!in) {
            throw std::runtime_error("unexpected end of file while reading string");
        }
    }
    return out;
}

uint64_t primitive_value_size(uint32_t value_type) {
    switch (value_type) {
        case GGUF_VALUE_UINT8:
        case GGUF_VALUE_INT8:
        case GGUF_VALUE_BOOL:
            return 1;
        case GGUF_VALUE_UINT16:
        case GGUF_VALUE_INT16:
            return 2;
        case GGUF_VALUE_UINT32:
        case GGUF_VALUE_INT32:
        case GGUF_VALUE_FLOAT32:
            return 4;
        case GGUF_VALUE_UINT64:
        case GGUF_VALUE_INT64:
        case GGUF_VALUE_FLOAT64:
            return 8;
        default:
            throw std::runtime_error("value type is not primitive");
    }
}

void skip_value(std::istream & in, uint32_t value_type) {
    if (value_type == GGUF_VALUE_STRING) {
        const uint64_t len = read_pod<uint64_t>(in);
        skip_bytes(in, len);
        return;
    }

    if (value_type == GGUF_VALUE_ARRAY) {
        const uint32_t arr_type = read_pod<uint32_t>(in);
        const uint64_t arr_len = read_pod<uint64_t>(in);
        if (arr_type == GGUF_VALUE_STRING) {
            for (uint64_t i = 0; i < arr_len; ++i) {
                const uint64_t len = read_pod<uint64_t>(in);
                skip_bytes(in, len);
            }
            return;
        }
        if (arr_type == GGUF_VALUE_ARRAY) {
            throw std::runtime_error("nested GGUF arrays are not supported");
        }
        skip_bytes(in, primitive_value_size(arr_type) * arr_len);
        return;
    }

    skip_bytes(in, primitive_value_size(value_type));
}

const quant_type_info * find_type(uint32_t type) {
    for (const quant_type_info & info : QUANT_TYPES) {
        if (info.type == type) {
            return &info;
        }
    }
    return nullptr;
}

const char * type_name(uint32_t type) {
    const quant_type_info * info = find_type(type);
    return info ? info->name : "UNKNOWN";
}

uint64_t checked_mul(uint64_t a, uint64_t b, const char * what) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::runtime_error(std::string("overflow while computing ") + what);
    }
    return a * b;
}

uint64_t row_size_for_type(uint32_t type, uint64_t row_elems) {
    const quant_type_info * info = find_type(type);
    if (info == nullptr) {
        throw std::runtime_error("unsupported GGML tensor type id " + std::to_string(type));
    }
    if (row_elems % info->block_size != 0) {
        throw std::runtime_error("tensor row size is not divisible by block size for type " + std::string(info->name));
    }
    return checked_mul(row_elems / info->block_size, info->type_size, "row size");
}

uint64_t tensor_nbytes(uint32_t type, const std::vector<uint64_t> & dims) {
    if (dims.empty()) {
        throw std::runtime_error("tensor has no dimensions");
    }
    uint64_t rows = 1;
    for (size_t i = 1; i < dims.size(); ++i) {
        rows = checked_mul(rows, dims[i], "row count");
    }
    return checked_mul(rows, row_size_for_type(type, dims[0]), "tensor size");
}

bool is_dense_float_type(uint32_t type) {
    return type == static_cast<uint32_t>(GGML_TYPE_F32) ||
           type == static_cast<uint32_t>(GGML_TYPE_F16) ||
           type == static_cast<uint32_t>(GGML_TYPE_BF16);
}

bool same_dims(const std::vector<uint64_t> & a, const std::vector<uint64_t> & b) {
    return a == b;
}

gguf_model read_gguf_header(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open model: " + path);
    }

    const uint32_t magic = read_pod<uint32_t>(in);
    if (magic != GGUF_FILE_MAGIC) {
        throw std::runtime_error("not a GGUF file: " + path);
    }

    gguf_model model;
    model.version = read_pod<uint32_t>(in);
    model.tensor_count = read_pod<uint64_t>(in);
    model.metadata_count = read_pod<uint64_t>(in);

    for (uint64_t i = 0; i < model.metadata_count; ++i) {
        const std::string key = read_string(in);
        const uint32_t value_type = read_pod<uint32_t>(in);
        if (key == "general.alignment" && value_type == GGUF_VALUE_UINT32) {
            model.alignment = read_pod<uint32_t>(in);
        } else {
            skip_value(in, value_type);
        }
    }

    model.tensors.reserve(static_cast<size_t>(model.tensor_count));
    for (uint64_t i = 0; i < model.tensor_count; ++i) {
        tensor_info tensor;
        tensor.name = read_string(in);
        const uint32_t n_dims = read_pod<uint32_t>(in);
        tensor.dims.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            tensor.dims[d] = read_pod<uint64_t>(in);
        }
        tensor.type = read_pod<uint32_t>(in);
        tensor.offset = read_pod<uint64_t>(in);
        tensor.nbytes = tensor_nbytes(tensor.type, tensor.dims);
        tensor.row_elems = tensor.dims.empty() ? 0 : tensor.dims[0];
        tensor.nrows = 1;
        for (size_t d = 1; d < tensor.dims.size(); ++d) {
            tensor.nrows = checked_mul(tensor.nrows, tensor.dims[d], "tensor row count");
        }
        model.tensor_index[tensor.name] = model.tensors.size();
        model.tensors.push_back(std::move(tensor));
    }

    model.data_start = align_up(static_cast<uint64_t>(in.tellg()), model.alignment);
    return model;
}

std::string dims_to_string(const std::vector<uint64_t> & dims) {
    std::string out = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i != 0) {
            out += ",";
        }
        out += std::to_string(dims[i]);
    }
    out += "]";
    return out;
}

std::string read_text_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open calibration text: " + path);
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (text.empty()) {
        throw std::runtime_error("calibration text is empty: " + path);
    }
    return text;
}

void print_usage(const char * argv0) {
    std::cerr << "usage: " << argv0 << " --fp-model original.gguf --tq-model input-tq1_0.gguf --calib text.txt --out qat.gguf [options]\n"
              << "\n"
              << "Experimental ternary-aware local QAT for TQ1_0 tensors. Runs calibration text through llama.cpp,\n"
              << "captures matmul input activation covariance, then applies CUDA STE-style block-output optimization.\n"
              << "The original dense GGUF is used as teacher weights; this is not full transformer backpropagation.\n"
              << "\n"
              << "Options:\n"
              << "  --ctx N                 Calibration context length (default: 2048)\n"
              << "  --batch N               Logical llama_decode batch size (default: 512)\n"
              << "  --ubatch N              Physical llama_decode batch size (default: same as --batch)\n"
              << "  --max-tokens N          Max calibration tokens to evaluate (default: 2048)\n"
              << "  --n-gpu-layers N        llama.cpp GPU offload layers; negative means auto/all (default: -1)\n"
              << "  --threads N             llama.cpp CPU threads; 0 keeps llama.cpp default\n"
              << "  --no-mmap               Disable mmap while loading the calibration model\n"
              << "  --no-parse-special      Treat special-token text literally during tokenization\n"
              << "  --chunk-mib N           Tensor chunk size for CUDA QAT packing (default: 512)\n"
              << "  --max-capture-mib N     Skip any single activation tensor larger than this (default: 256)\n"
              << "  --qat-steps N           CUDA STE update steps per 256-weight block (default: 16)\n"
              << "  --learning-rate X       STE latent-weight learning rate (default: 0.02)\n"
              << "  --grad-clip X           Per-weight gradient clip; 0 disables clipping (default: 1.0)\n"
              << "  --cov-ridge X           Ridge added to normalized covariance diagonal (default: 1e-4)\n"
              << "  --capture-stride N      Use every Nth activation vector for covariance (default: 1)\n"
              << "  --threshold-base X      Initial threshold = X * weighted mean(abs(W)) (default: 0.75)\n"
              << "  --include-regex REGEX   Only optimize matching tensor names\n"
              << "  --exclude-regex REGEX   Do not optimize matching tensor names\n"
              << "  --dry-run               Report targets without model execution or output writes\n";
}

options parse_args(int argc, char ** argv) {
    options opt;
    bool ubatch_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char * flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + flag);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--fp-model") {
            opt.fp_model = need_value("--fp-model");
        } else if (arg == "--tq-model") {
            opt.tq_model = need_value("--tq-model");
        } else if (arg == "--calib") {
            opt.calib_path = need_value("--calib");
        } else if (arg == "--out") {
            opt.output_model = need_value("--out");
        } else if (arg == "--ctx") {
            opt.n_ctx = static_cast<uint32_t>(std::stoul(need_value("--ctx")));
        } else if (arg == "--batch") {
            opt.n_batch = static_cast<uint32_t>(std::stoul(need_value("--batch")));
        } else if (arg == "--ubatch") {
            opt.n_ubatch = static_cast<uint32_t>(std::stoul(need_value("--ubatch")));
            ubatch_set = true;
        } else if (arg == "--max-tokens") {
            opt.max_tokens = static_cast<uint32_t>(std::stoul(need_value("--max-tokens")));
        } else if (arg == "--n-gpu-layers") {
            opt.n_gpu_layers = static_cast<int32_t>(std::stoi(need_value("--n-gpu-layers")));
        } else if (arg == "--threads") {
            opt.n_threads = static_cast<int32_t>(std::stoi(need_value("--threads")));
        } else if (arg == "--no-mmap") {
            opt.no_mmap = true;
        } else if (arg == "--no-parse-special") {
            opt.parse_special = false;
        } else if (arg == "--chunk-mib") {
            opt.chunk_mib = std::stoull(need_value("--chunk-mib"));
        } else if (arg == "--max-capture-mib") {
            opt.max_capture_mib = std::stoull(need_value("--max-capture-mib"));
        } else if (arg == "--qat-steps") {
            opt.qat_steps = std::stoi(need_value("--qat-steps"));
        } else if (arg == "--learning-rate") {
            opt.learning_rate = std::stof(need_value("--learning-rate"));
        } else if (arg == "--grad-clip") {
            opt.grad_clip = std::stof(need_value("--grad-clip"));
        } else if (arg == "--cov-ridge") {
            opt.covariance_ridge = std::stof(need_value("--cov-ridge"));
        } else if (arg == "--capture-stride") {
            opt.capture_stride = static_cast<uint32_t>(std::stoul(need_value("--capture-stride")));
        } else if (arg == "--threshold-base") {
            opt.threshold_base = std::stof(need_value("--threshold-base"));
        } else if (arg == "--include-regex") {
            opt.include_regex = need_value("--include-regex");
        } else if (arg == "--exclude-regex") {
            opt.exclude_regex = need_value("--exclude-regex");
        } else if (arg == "--dry-run") {
            opt.dry_run = true;
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (!ubatch_set) {
        opt.n_ubatch = opt.n_batch;
    }
    if (opt.fp_model.empty() || opt.tq_model.empty() || opt.output_model.empty() || opt.calib_path.empty()) {
        print_usage(argv[0]);
        throw std::runtime_error("--fp-model, --tq-model, --calib and --out are required");
    }
    if (opt.n_ctx < 16 || opt.n_batch < 1 || opt.n_ubatch < 1 || opt.max_tokens < 1) {
        throw std::runtime_error("--ctx must be >= 16 and batch/token limits must be >= 1");
    }
    if (opt.chunk_mib < 1 || opt.max_capture_mib < 1) {
        throw std::runtime_error("--chunk-mib and --max-capture-mib must be >= 1");
    }
    if (opt.qat_steps < 1) {
        throw std::runtime_error("--qat-steps must be >= 1");
    }
    if (opt.learning_rate <= 0.0f) {
        throw std::runtime_error("--learning-rate must be > 0");
    }
    if (opt.grad_clip < 0.0f) {
        throw std::runtime_error("--grad-clip must be >= 0");
    }
    if (opt.covariance_ridge < 0.0f) {
        throw std::runtime_error("--cov-ridge must be >= 0");
    }
    if (opt.capture_stride < 1) {
        throw std::runtime_error("--capture-stride must be >= 1");
    }
    if (opt.threshold_base < 0.0f) {
        throw std::runtime_error("--threshold-base must be >= 0");
    }
    if (opt.output_model == opt.tq_model || opt.output_model == opt.fp_model) {
        throw std::runtime_error("--out must differ from input model paths");
    }
    return opt;
}

float read_scalar_from_bytes(const uint8_t * p, enum ggml_type type) {
    if (type == GGML_TYPE_F32) {
        float value;
        std::memcpy(&value, p, sizeof(value));
        return value;
    }
    if (type == GGML_TYPE_F16) {
        uint16_t bits;
        std::memcpy(&bits, p, sizeof(bits));
        __half h;
        std::memcpy(&h, &bits, sizeof(bits));
        return __half2float(h);
    }
    if (type == GGML_TYPE_BF16) {
        uint16_t bits16;
        std::memcpy(&bits16, p, sizeof(bits16));
        const uint32_t bits32 = static_cast<uint32_t>(bits16) << 16;
        float value;
        std::memcpy(&value, &bits32, sizeof(value));
        return value;
    }
    return 0.0f;
}

bool is_supported_activation_type(enum ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

void capture_activation_tensor(const ggml_tensor * tensor, activation_stat & stat, capture_state & state) {
    if (tensor == nullptr) {
        state.skipped_nodes += 1;
        stat.skipped += 1;
        return;
    }
    if (!is_supported_activation_type(tensor->type)) {
        state.unsupported_type_nodes += 1;
        stat.skipped += 1;
        return;
    }
    if (tensor->ne[0] != static_cast<int64_t>(stat.n_features)) {
        state.mismatched_shape_nodes += 1;
        stat.skipped += 1;
        return;
    }

    const size_t nbytes = ggml_nbytes(tensor);
    if (nbytes == 0 || nbytes > state.max_capture_bytes) {
        state.skipped_nodes += 1;
        stat.skipped += 1;
        return;
    }

    std::vector<uint8_t> data(nbytes);
    ggml_backend_tensor_get(tensor, data.data(), 0, nbytes);

    const int64_t n0 = tensor->ne[0];
    if (n0 % static_cast<int64_t>(TQ_BLOCK_SIZE) != 0) {
        state.mismatched_shape_nodes += 1;
        stat.skipped += 1;
        return;
    }
    const int64_t n1 = std::max<int64_t>(1, tensor->ne[1]);
    const int64_t n2 = std::max<int64_t>(1, tensor->ne[2]);
    const int64_t n3 = std::max<int64_t>(1, tensor->ne[3]);

    const size_t elem_size = ggml_type_size(tensor->type);
    uint64_t vector_index = 0;
    uint64_t used_vectors = 0;
    float xb[TQ_BLOCK_SIZE];

    for (int64_t i3 = 0; i3 < n3; ++i3) {
        for (int64_t i2 = 0; i2 < n2; ++i2) {
            for (int64_t i1 = 0; i1 < n1; ++i1) {
                if ((vector_index++ % state.capture_stride) != 0) {
                    continue;
                }
                const size_t base = static_cast<size_t>(i1) * tensor->nb[1] +
                                    static_cast<size_t>(i2) * tensor->nb[2] +
                                    static_cast<size_t>(i3) * tensor->nb[3];

                for (uint64_t b = 0; b < stat.n_feature_blocks; ++b) {
                    for (uint64_t j = 0; j < TQ_BLOCK_SIZE; ++j) {
                        const uint64_t feature = b * TQ_BLOCK_SIZE + j;
                        const size_t off = base + static_cast<size_t>(feature) * tensor->nb[0];
                        xb[j] = off + elem_size <= data.size() ? read_scalar_from_bytes(data.data() + off, tensor->type) : 0.0f;
                    }

                    float * cov = stat.covariance.data() + static_cast<size_t>(b * TQ_BLOCK_SIZE * TQ_BLOCK_SIZE);
                    for (uint64_t r = 0; r < TQ_BLOCK_SIZE; ++r) {
                        const float vr = xb[r];
                        float * row = cov + static_cast<size_t>(r * TQ_BLOCK_SIZE);
                        for (uint64_t c = 0; c < TQ_BLOCK_SIZE; ++c) {
                            row[c] += vr * xb[c];
                        }
                    }
                }
                used_vectors += 1;
            }
        }
    }

    stat.samples += used_vectors;
    stat.captures += 1;
    state.captured_nodes += 1;
}

bool calibration_eval_callback(ggml_tensor * t, bool ask, void * user_data) {
    auto * state = static_cast<capture_state *>(user_data);
    if (t == nullptr || t->op != GGML_OP_MUL_MAT || t->src[0] == nullptr || t->src[1] == nullptr) {
        return ask ? false : true;
    }

    const char * weight_name_c = ggml_get_name(t->src[0]);
    if (weight_name_c == nullptr || weight_name_c[0] == '\0') {
        return ask ? false : true;
    }
    auto it = state->by_weight_name.find(weight_name_c);
    if (it == state->by_weight_name.end()) {
        return ask ? false : true;
    }

    if (ask) {
        state->matched_nodes += 1;
        return true;
    }

    capture_activation_tensor(t->src[1], *it->second, *state);
    return true;
}

std::vector<llama_token> tokenize_calibration_text(const llama_vocab * vocab, const std::string & text, bool parse_special) {
    std::vector<llama_token> tokens(std::max<size_t>(32, text.size() + 8));
    int32_t n_tokens = llama_tokenize(
            vocab,
            text.data(),
            static_cast<int32_t>(std::min<size_t>(text.size(), static_cast<size_t>(std::numeric_limits<int32_t>::max()))),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            true,
            parse_special);
    if (n_tokens < 0) {
        tokens.resize(static_cast<size_t>(-n_tokens));
        n_tokens = llama_tokenize(
                vocab,
                text.data(),
                static_cast<int32_t>(std::min<size_t>(text.size(), static_cast<size_t>(std::numeric_limits<int32_t>::max()))),
                tokens.data(),
                static_cast<int32_t>(tokens.size()),
                true,
                parse_special);
    }
    if (n_tokens <= 0) {
        throw std::runtime_error("failed to tokenize calibration text");
    }
    tokens.resize(static_cast<size_t>(n_tokens));
    return tokens;
}

void run_calibration(const options & opt, capture_state & state) {
    llama_backend_guard backend;

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = opt.n_gpu_layers;
    mparams.use_mmap = !opt.no_mmap;

    std::cerr << "loading calibration model: " << opt.fp_model << "\n";
    llama_model * model = llama_model_load_from_file(opt.fp_model.c_str(), mparams);
    if (model == nullptr) {
        throw std::runtime_error("failed to load calibration model with llama.cpp");
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = opt.n_ctx;
    cparams.n_batch = opt.n_batch;
    cparams.n_ubatch = opt.n_ubatch;
    cparams.cb_eval = calibration_eval_callback;
    cparams.cb_eval_user_data = &state;
    cparams.no_perf = true;
    if (opt.n_threads > 0) {
        cparams.n_threads = opt.n_threads;
        cparams.n_threads_batch = opt.n_threads;
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        llama_model_free(model);
        throw std::runtime_error("failed to create llama.cpp context");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string text = read_text_file(opt.calib_path);
    std::vector<llama_token> tokens = tokenize_calibration_text(vocab, text, opt.parse_special);

    const uint32_t actual_ctx = llama_n_ctx(ctx);
    uint32_t eval_tokens = std::min<uint32_t>(opt.max_tokens, static_cast<uint32_t>(tokens.size()));
    if (eval_tokens > actual_ctx) {
        std::cerr << "calibration tokens limited from " << eval_tokens << " to context " << actual_ctx << "\n";
        eval_tokens = actual_ctx;
    }
    if (eval_tokens == 0) {
        llama_free(ctx);
        llama_model_free(model);
        throw std::runtime_error("no calibration tokens available after limits");
    }

    std::cerr << "calibration: " << eval_tokens << " tokens, ctx=" << actual_ctx
              << ", batch=" << llama_n_batch(ctx) << ", ubatch=" << opt.n_ubatch << "\n";

    uint32_t pos = 0;
    while (pos < eval_tokens) {
        const uint32_t n_eval = std::min<uint32_t>(llama_n_batch(ctx), eval_tokens - pos);
        llama_batch batch = llama_batch_get_one(tokens.data() + pos, static_cast<int32_t>(n_eval));
        const int32_t rc = llama_decode(ctx, batch);
        if (rc != 0) {
            llama_free(ctx);
            llama_model_free(model);
            throw std::runtime_error("llama_decode failed during calibration with code " + std::to_string(rc));
        }
        pos += n_eval;
        std::cerr << "  decoded " << pos << "/" << eval_tokens
                  << " tokens, captured nodes=" << state.captured_nodes << "\r" << std::flush;
    }
    std::cerr << "\n";

    llama_free(ctx);
    llama_model_free(model);
}

void normalize_covariance(const activation_stat & stat, std::vector<float> & covariance, float ridge) {
    const uint64_t block_values = TQ_BLOCK_SIZE * TQ_BLOCK_SIZE;
    covariance.resize(static_cast<size_t>(stat.n_feature_blocks * block_values));

    if (stat.samples == 0) {
        std::fill(covariance.begin(), covariance.end(), 0.0f);
        for (uint64_t b = 0; b < stat.n_feature_blocks; ++b) {
            float * cov = covariance.data() + static_cast<size_t>(b * block_values);
            for (uint64_t i = 0; i < TQ_BLOCK_SIZE; ++i) {
                cov[i * TQ_BLOCK_SIZE + i] = 1.0f + ridge;
            }
        }
        return;
    }

    for (uint64_t b = 0; b < stat.n_feature_blocks; ++b) {
        const float * src = stat.covariance.data() + static_cast<size_t>(b * block_values);
        float * dst = covariance.data() + static_cast<size_t>(b * block_values);

        double mean_diag = 0.0;
        for (uint64_t i = 0; i < TQ_BLOCK_SIZE; ++i) {
            mean_diag += static_cast<double>(src[i * TQ_BLOCK_SIZE + i]) / static_cast<double>(stat.samples);
        }
        mean_diag /= static_cast<double>(TQ_BLOCK_SIZE);
        const float inv_scale = mean_diag > 1.0e-20 ? static_cast<float>(1.0 / mean_diag) : 1.0f;

        for (uint64_t i = 0; i < block_values; ++i) {
            dst[i] = (src[i] / static_cast<float>(stat.samples)) * inv_scale;
        }
        for (uint64_t i = 0; i < TQ_BLOCK_SIZE; ++i) {
            dst[i * TQ_BLOCK_SIZE + i] += ridge;
        }
    }
}

__device__ float load_dense_value(const uint8_t * input, uint64_t idx, uint32_t type) {
    if (type == static_cast<uint32_t>(GGML_TYPE_F32)) {
        return reinterpret_cast<const float *>(input)[idx];
    }
    if (type == static_cast<uint32_t>(GGML_TYPE_F16)) {
        const uint16_t bits = reinterpret_cast<const uint16_t *>(input)[idx];
        __half h;
        reinterpret_cast<uint16_t *>(&h)[0] = bits;
        return __half2float(h);
    }

    const uint16_t bits16 = reinterpret_cast<const uint16_t *>(input)[idx];
    const uint32_t bits32 = static_cast<uint32_t>(bits16) << 16;
    return __uint_as_float(bits32);
}

__device__ uint8_t encode_trits(uint32_t q) {
    return static_cast<uint8_t>((q * 256u + 242u) / 243u);
}

__device__ int8_t choose_ternary(float w, float alpha) {
    if (alpha <= 0.0f) {
        return 0;
    }
    const float half_alpha = 0.5f * alpha;
    if (w > half_alpha) {
        return 1;
    }
    if (w < -half_alpha) {
        return -1;
    }
    return 0;
}

__device__ float block_sum_256(float value, float * scratch) {
    const int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    return scratch[0];
}

__global__ void qat_tq1_cov_kernel(
        const uint8_t * input,
        uint8_t * output,
        const float * covariance,
        uint64_t n_blocks,
        uint64_t blocks_per_row,
        uint32_t input_type,
        int qat_steps,
        float learning_rate,
        float grad_clip,
        float threshold_base) {
    const uint64_t block_id = static_cast<uint64_t>(blockIdx.x);
    const int tid = threadIdx.x;
    if (block_id >= n_blocks || tid >= static_cast<int>(TQ_BLOCK_SIZE)) {
        return;
    }

    const uint64_t elem0 = block_id * TQ_BLOCK_SIZE;
    const uint64_t col_block = block_id % blocks_per_row;
    const float * cov = covariance + col_block * TQ_BLOCK_SIZE * TQ_BLOCK_SIZE;

    __shared__ float w[TQ_BLOCK_SIZE];
    __shared__ float u[TQ_BLOCK_SIZE];
    __shared__ float qf[TQ_BLOCK_SIZE];
    __shared__ float e[TQ_BLOCK_SIZE];
    __shared__ float cw[TQ_BLOCK_SIZE];
    __shared__ float scratch[TQ_BLOCK_SIZE];
    __shared__ float alpha;
    __shared__ float delta;

    const float wi = load_dense_value(input, elem0 + tid, input_type);
    const float diag = fmaxf(cov[tid * TQ_BLOCK_SIZE + tid], 1.0e-8f);
    w[tid] = wi;
    u[tid] = wi;

    const float sum_abs = block_sum_256(diag * fabsf(wi), scratch);
    const float sum_diag = block_sum_256(diag, scratch);
    if (tid == 0) {
        delta = threshold_base * sum_abs / fmaxf(sum_diag, 1.0e-8f);
    }
    __syncthreads();

    qf[tid] = fabsf(wi) > delta ? (wi > 0.0f ? 1.0f : -1.0f) : 0.0f;
    __syncthreads();

    float cwi = 0.0f;
    for (int j = 0; j < static_cast<int>(TQ_BLOCK_SIZE); ++j) {
        cwi += cov[tid * TQ_BLOCK_SIZE + j] * w[j];
    }
    cw[tid] = cwi;
    __syncthreads();

    for (int step = 0; step < qat_steps; ++step) {
        float cq = 0.0f;
        for (int j = 0; j < static_cast<int>(TQ_BLOCK_SIZE); ++j) {
            cq += cov[tid * TQ_BLOCK_SIZE + j] * qf[j];
        }

        const float numer = block_sum_256(qf[tid] * cw[tid], scratch);
        const float denom = block_sum_256(qf[tid] * cq, scratch);
        if (tid == 0) {
            alpha = denom > 1.0e-12f ? fmaxf(0.0f, numer / denom) : 0.0f;
        }
        __syncthreads();

        e[tid] = alpha * qf[tid] - w[tid];
        __syncthreads();

        float grad = 0.0f;
        for (int j = 0; j < static_cast<int>(TQ_BLOCK_SIZE); ++j) {
            grad += cov[tid * TQ_BLOCK_SIZE + j] * e[j];
        }
        if (grad_clip > 0.0f) {
            grad = fminf(grad_clip, fmaxf(-grad_clip, grad));
        }
        u[tid] -= learning_rate * grad;

        qf[tid] = choose_ternary(u[tid], alpha);
        __syncthreads();
    }

    float cq = 0.0f;
    for (int j = 0; j < static_cast<int>(TQ_BLOCK_SIZE); ++j) {
        cq += cov[tid * TQ_BLOCK_SIZE + j] * qf[j];
    }
    const float numer = block_sum_256(qf[tid] * cw[tid], scratch);
    const float denom = block_sum_256(qf[tid] * cq, scratch);
    if (tid == 0) {
        alpha = denom > 1.0e-12f ? fmaxf(0.0f, numer / denom) : 0.0f;
    }
    __syncthreads();

    uint8_t * dst = output + block_id * TQ_TYPE_SIZE;

    if (tid < 32) {
        uint32_t code = 0;
        for (int n = 0; n < 5; ++n) {
            code = code * 3u + static_cast<uint32_t>(static_cast<int>(qf[tid + n * 32]) + 1);
        }
        dst[tid] = encode_trits(code);
    }

    if (tid < 16) {
        uint32_t code = 0;
        for (int n = 0; n < 5; ++n) {
            code = code * 3u + static_cast<uint32_t>(static_cast<int>(qf[160 + tid + n * 16]) + 1);
        }
        dst[32 + tid] = encode_trits(code);
    }

    if (tid < 4) {
        uint32_t code = 0;
        for (int m = 0; m < 4; ++m) {
            code = code * 3u + static_cast<uint32_t>(static_cast<int>(qf[240 + tid + m * 4]) + 1);
        }
        code *= 3u;
        dst[48 + tid] = encode_trits(code);
    }

    if (tid == 0) {
        __half half_alpha = __float2half(alpha);
        const uint16_t bits = reinterpret_cast<uint16_t *>(&half_alpha)[0];
        dst[52] = static_cast<uint8_t>(bits & 0xffu);
        dst[53] = static_cast<uint8_t>((bits >> 8) & 0xffu);
    }
}

void refine_tensor_cuda(
        std::ifstream & fp_in,
        std::fstream & out,
        const gguf_model & fp_model,
        const gguf_model & tq_model,
        const tensor_info & fp_tensor,
        const tensor_info & tq_tensor,
        const activation_stat & stat,
        const options & opt) {
    const uint64_t input_row_bytes = row_size_for_type(fp_tensor.type, fp_tensor.row_elems);
    const uint64_t output_row_bytes = row_size_for_type(tq_tensor.type, tq_tensor.row_elems);
    const uint64_t blocks_per_row = tq_tensor.row_elems / TQ_BLOCK_SIZE;
    const uint64_t rows_per_chunk = std::max<uint64_t>(1, (opt.chunk_mib * 1024ull * 1024ull) / input_row_bytes);

    std::vector<float> covariance;
    normalize_covariance(stat, covariance, opt.covariance_ridge);

    float * dev_covariance = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_covariance), covariance.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev_covariance, covariance.data(), covariance.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<uint8_t> host_input;
    std::vector<uint8_t> host_output;

    uint64_t rows_done = 0;
    while (rows_done < fp_tensor.nrows) {
        const uint64_t rows = std::min<uint64_t>(rows_per_chunk, fp_tensor.nrows - rows_done);
        const uint64_t input_bytes = checked_mul(rows, input_row_bytes, "input chunk size");
        const uint64_t output_bytes = checked_mul(rows, output_row_bytes, "output chunk size");
        const uint64_t n_blocks = output_bytes / TQ_TYPE_SIZE;

        host_input.resize(static_cast<size_t>(input_bytes));
        host_output.resize(static_cast<size_t>(output_bytes));

        fp_in.seekg(static_cast<std::streamoff>(fp_model.data_start + fp_tensor.offset + rows_done * input_row_bytes), std::ios::beg);
        fp_in.read(reinterpret_cast<char *>(host_input.data()), static_cast<std::streamsize>(input_bytes));
        if (!fp_in) {
            CUDA_CHECK(cudaFree(dev_covariance));
            throw std::runtime_error("failed to read original tensor: " + fp_tensor.name);
        }

        uint8_t * dev_input = nullptr;
        uint8_t * dev_output = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_input), static_cast<size_t>(input_bytes)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_output), static_cast<size_t>(output_bytes)));
        CUDA_CHECK(cudaMemcpy(dev_input, host_input.data(), static_cast<size_t>(input_bytes), cudaMemcpyHostToDevice));

        constexpr int threads = static_cast<int>(TQ_BLOCK_SIZE);
        qat_tq1_cov_kernel<<<static_cast<uint32_t>(n_blocks), threads>>>(
                dev_input,
                dev_output,
                dev_covariance,
                n_blocks,
                blocks_per_row,
                fp_tensor.type,
                opt.qat_steps,
                opt.learning_rate,
                opt.grad_clip,
                opt.threshold_base);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(host_output.data(), dev_output, static_cast<size_t>(output_bytes), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dev_input));
        CUDA_CHECK(cudaFree(dev_output));

        out.seekp(static_cast<std::streamoff>(tq_model.data_start + tq_tensor.offset + rows_done * output_row_bytes), std::ios::beg);
        out.write(reinterpret_cast<const char *>(host_output.data()), static_cast<std::streamsize>(output_bytes));
        if (!out) {
            CUDA_CHECK(cudaFree(dev_covariance));
            throw std::runtime_error("failed to write QAT tensor: " + tq_tensor.name);
        }

        rows_done += rows;
    }

    CUDA_CHECK(cudaFree(dev_covariance));
}

bool should_consider(const tensor_info & tensor, const options &, const std::regex * include_re, const std::regex * exclude_re) {
    if (tensor.type != static_cast<uint32_t>(GGML_TYPE_TQ1_0)) {
        return false;
    }
    if (include_re != nullptr && !std::regex_search(tensor.name, *include_re)) {
        return false;
    }
    if (exclude_re != nullptr && std::regex_search(tensor.name, *exclude_re)) {
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const options opt = parse_args(argc, argv);
        const gguf_model fp_model = read_gguf_header(opt.fp_model);
        const gguf_model tq_model = read_gguf_header(opt.tq_model);

        std::regex include_re_storage;
        std::regex exclude_re_storage;
        const std::regex * include_re = nullptr;
        const std::regex * exclude_re = nullptr;
        if (!opt.include_regex.empty()) {
            include_re_storage = std::regex(opt.include_regex);
            include_re = &include_re_storage;
        }
        if (!opt.exclude_regex.empty()) {
            exclude_re_storage = std::regex(opt.exclude_regex);
            exclude_re = &exclude_re_storage;
        }

        std::unordered_map<std::string, activation_stat> stats;
        std::vector<tensor_pair> refine;

        for (const tensor_info & tq_tensor : tq_model.tensors) {
            if (!should_consider(tq_tensor, opt, include_re, exclude_re)) {
                continue;
            }
            auto it = fp_model.tensor_index.find(tq_tensor.name);
            if (it == fp_model.tensor_index.end()) {
                continue;
            }
            const tensor_info & fp_tensor = fp_model.tensors[it->second];
            if (!is_dense_float_type(fp_tensor.type) || !same_dims(fp_tensor.dims, tq_tensor.dims)) {
                continue;
            }
            if (tq_tensor.row_elems % TQ_BLOCK_SIZE != 0) {
                continue;
            }
            activation_stat stat;
            stat.n_features = tq_tensor.row_elems;
            stat.n_feature_blocks = stat.n_features / TQ_BLOCK_SIZE;
            stat.covariance.assign(static_cast<size_t>(stat.n_feature_blocks * TQ_BLOCK_SIZE * TQ_BLOCK_SIZE), 0.0f);
            auto inserted = stats.emplace(tq_tensor.name, std::move(stat));
            refine.push_back({ &fp_tensor, &tq_tensor, &inserted.first->second });
        }

        std::cerr << "original: " << opt.fp_model << "\n";
        std::cerr << "ternary:  " << opt.tq_model << "\n";
        std::cerr << "calib:    " << opt.calib_path << "\n";
        std::cerr << "output:   " << opt.output_model << "\n";
        std::cerr << "targets:  " << refine.size() << " TQ1_0 tensors\n";
        for (const tensor_pair & p : refine) {
            std::cerr << "  target " << p.tq->name << " " << dims_to_string(p.tq->dims)
                      << " " << type_name(p.fp->type) << " + covariance QAT -> TQ1_0\n";
        }

        if (opt.dry_run) {
            return 0;
        }
        if (refine.empty()) {
            throw std::runtime_error("no matching TQ1_0 tensors found to optimize");
        }

        capture_state state;
        state.max_capture_bytes = opt.max_capture_mib * 1024ull * 1024ull;
        state.capture_stride = opt.capture_stride;
        for (auto & kv : stats) {
            state.by_weight_name.emplace(kv.first, &kv.second);
        }

        run_calibration(opt, state);

        std::cerr << "activation capture summary:\n";
        std::cerr << "  matched matmul callbacks: " << state.matched_nodes << "\n";
        std::cerr << "  captured activations:     " << state.captured_nodes << "\n";
        std::cerr << "  skipped activations:      " << state.skipped_nodes << "\n";
        std::cerr << "  unsupported types:        " << state.unsupported_type_nodes << "\n";
        std::cerr << "  mismatched shapes:        " << state.mismatched_shape_nodes << "\n";

        size_t captured_targets = 0;
        for (const tensor_pair & p : refine) {
            if (p.stat->samples > 0) {
                captured_targets += 1;
            }
        }
        std::cerr << "  tensors with samples:     " << captured_targets << "/" << refine.size() << "\n";
        if (captured_targets == 0) {
            throw std::runtime_error("no target tensor activations were captured; check tensor names and llama.cpp callback support");
        }

        fs::copy_file(opt.tq_model, opt.output_model, fs::copy_options::overwrite_existing);

        std::ifstream fp_in(opt.fp_model, std::ios::binary);
        if (!fp_in) {
            throw std::runtime_error("failed to reopen original model");
        }
        std::fstream out(opt.output_model, std::ios::binary | std::ios::in | std::ios::out);
        if (!out) {
            throw std::runtime_error("failed to open output model for patching");
        }

        for (size_t i = 0; i < refine.size(); ++i) {
            const tensor_pair & p = refine[i];
            if (p.stat->samples == 0) {
                std::cerr << "[" << (i + 1) << "/" << refine.size() << "] QAT " << p.tq->name
                          << " (no activation samples; using identity covariance)\n";
            } else {
                std::cerr << "[" << (i + 1) << "/" << refine.size() << "] QAT " << p.tq->name
                          << " (samples=" << p.stat->samples << ", captures=" << p.stat->captures << ")\n";
            }
            refine_tensor_cuda(fp_in, out, fp_model, tq_model, *p.fp, *p.tq, *p.stat, opt);
        }

        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
