#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t GGUF_MAGIC = 0x46554747u; // "GGUF" little-endian

constexpr uint32_t GGML_TYPE_F32   = 0;
constexpr uint32_t GGML_TYPE_F16   = 1;
constexpr uint32_t GGML_TYPE_BF16  = 30;
constexpr uint32_t GGML_TYPE_TQ1_0 = 34;

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

constexpr uint32_t LLAMA_FTYPE_MOSTLY_TQ1_0 = 36;

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
    uint32_t input_type = 0;
    uint32_t output_type = 0;
    uint64_t input_offset = 0;
    uint64_t output_offset = 0;
    uint64_t input_nbytes = 0;
    uint64_t output_nbytes = 0;
    uint64_t row_elems = 0;
    uint64_t nrows = 0;
    bool quantize = false;
};

struct options {
    std::string input_path;
    std::string output_path;
    std::string include_regex;
    std::string exclude_regex;
    uint64_t chunk_mib = 512;
    int threshold_grid = 20;
    float threshold_base = 0.75f;
    float threshold_min = 0.5f;
    float threshold_max = 1.5f;
    bool quantize_output = false;
    bool quantize_token_embd = false;
    bool patch_file_type = true;
    bool dry_run = false;
};

struct gguf_model {
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t metadata_count = 0;
    uint32_t alignment = 32;
    uint64_t input_data_start = 0;
    std::vector<uint8_t> metadata_bytes;
    size_t file_type_value_offset = std::numeric_limits<size_t>::max();
    std::vector<tensor_info> tensors;
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

template <typename T>
void write_pod(std::ostream & out, T value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(T));
    if (!out) {
        throw std::runtime_error("failed to write output");
    }
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

void write_string(std::ostream & out, const std::string & str) {
    write_pod<uint64_t>(out, static_cast<uint64_t>(str.size()));
    out.write(str.data(), static_cast<std::streamsize>(str.size()));
    if (!out) {
        throw std::runtime_error("failed to write string");
    }
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
            throw std::runtime_error("nested GGUF arrays are not supported by this converter");
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
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

bool has_substr(const std::string & s, const char * needle) {
    return s.find(needle) != std::string::npos;
}

bool default_excluded(const tensor_info & tensor, const options & opt) {
    if (!opt.quantize_token_embd && has_substr(tensor.name, "token_embd")) {
        return true;
    }
    if (!opt.quantize_output && tensor.name == "output.weight") {
        return true;
    }
    return false;
}

void patch_u32(std::vector<uint8_t> & bytes, size_t offset, uint32_t value) {
    if (offset + sizeof(uint32_t) > bytes.size()) {
        throw std::runtime_error("metadata patch offset is out of range");
    }
    bytes[offset + 0] = static_cast<uint8_t>((value >> 0) & 0xff);
    bytes[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xff);
    bytes[offset + 2] = static_cast<uint8_t>((value >> 16) & 0xff);
    bytes[offset + 3] = static_cast<uint8_t>((value >> 24) & 0xff);
}

gguf_model read_gguf_header(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open input: " + path);
    }

    const uint32_t magic = read_pod<uint32_t>(in);
    if (magic != GGUF_MAGIC) {
        throw std::runtime_error("input is not a GGUF file");
    }

    gguf_model model;
    model.version = read_pod<uint32_t>(in);
    model.tensor_count = read_pod<uint64_t>(in);
    model.metadata_count = read_pod<uint64_t>(in);

    const uint64_t metadata_start = static_cast<uint64_t>(in.tellg());

    for (uint64_t i = 0; i < model.metadata_count; ++i) {
        const std::string key = read_string(in);
        const uint32_t value_type = read_pod<uint32_t>(in);
        const uint64_t value_start = static_cast<uint64_t>(in.tellg());

        if (key == "general.alignment" && value_type == GGUF_VALUE_UINT32) {
            model.alignment = read_pod<uint32_t>(in);
        } else if (key == "general.file_type" &&
                   (value_type == GGUF_VALUE_UINT32 || value_type == GGUF_VALUE_INT32)) {
            model.file_type_value_offset = static_cast<size_t>(value_start - metadata_start);
            skip_value(in, value_type);
        } else {
            skip_value(in, value_type);
        }
    }

    const uint64_t metadata_end = static_cast<uint64_t>(in.tellg());
    in.seekg(static_cast<std::streamoff>(metadata_start), std::ios::beg);
    model.metadata_bytes.resize(static_cast<size_t>(metadata_end - metadata_start));
    if (!model.metadata_bytes.empty()) {
        in.read(reinterpret_cast<char *>(model.metadata_bytes.data()), static_cast<std::streamsize>(model.metadata_bytes.size()));
        if (!in) {
            throw std::runtime_error("failed to read metadata bytes");
        }
    }
    in.seekg(static_cast<std::streamoff>(metadata_end), std::ios::beg);

    model.tensors.reserve(static_cast<size_t>(model.tensor_count));
    for (uint64_t i = 0; i < model.tensor_count; ++i) {
        tensor_info tensor;
        tensor.name = read_string(in);
        const uint32_t n_dims = read_pod<uint32_t>(in);
        tensor.dims.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            tensor.dims[d] = read_pod<uint64_t>(in);
        }
        tensor.input_type = read_pod<uint32_t>(in);
        tensor.output_type = tensor.input_type;
        tensor.input_offset = read_pod<uint64_t>(in);
        tensor.input_nbytes = tensor_nbytes(tensor.input_type, tensor.dims);
        tensor.output_nbytes = tensor.input_nbytes;
        tensor.row_elems = tensor.dims.empty() ? 0 : tensor.dims[0];
        tensor.nrows = 1;
        for (size_t d = 1; d < tensor.dims.size(); ++d) {
            tensor.nrows = checked_mul(tensor.nrows, tensor.dims[d], "tensor row count");
        }
        model.tensors.push_back(std::move(tensor));
    }

    model.input_data_start = align_up(static_cast<uint64_t>(in.tellg()), model.alignment);
    return model;
}

void write_padding(std::ostream & out, uint64_t n) {
    static const uint8_t zeros[4096] = {};
    while (n > 0) {
        const uint64_t chunk = std::min<uint64_t>(n, sizeof(zeros));
        out.write(reinterpret_cast<const char *>(zeros), static_cast<std::streamsize>(chunk));
        if (!out) {
            throw std::runtime_error("failed to write padding");
        }
        n -= chunk;
    }
}

void copy_bytes(std::ifstream & in, std::ofstream & out, uint64_t n) {
    std::vector<char> buffer(8 * 1024 * 1024);
    while (n > 0) {
        const uint64_t chunk = std::min<uint64_t>(n, buffer.size());
        in.read(buffer.data(), static_cast<std::streamsize>(chunk));
        if (!in) {
            throw std::runtime_error("failed to read tensor data for copy");
        }
        out.write(buffer.data(), static_cast<std::streamsize>(chunk));
        if (!out) {
            throw std::runtime_error("failed to write copied tensor data");
        }
        n -= chunk;
    }
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

void print_usage(const char * argv0) {
    std::cerr << "usage: " << argv0 << " [options] input.gguf output-tq1_0.gguf\n"
              << "\n"
              << "Options:\n"
              << "  --dry-run                  Parse and report without writing output\n"
              << "  --chunk-mib N              Tensor chunk size for GPU conversion (default: 512)\n"
              << "  --threshold-grid N         Grid points for PT-BitNet-style threshold search (default: 20)\n"
              << "  --threshold-base X         Base threshold = X * mean(abs(W)) (default: 0.75)\n"
              << "  --threshold-min X          Grid lower factor around base threshold (default: 0.5)\n"
              << "  --threshold-max X          Grid upper factor around base threshold (default: 1.5)\n"
              << "  --include-regex REGEX      Only quantize matching tensor names\n"
              << "  --exclude-regex REGEX      Do not quantize matching tensor names\n"
              << "  --quantize-output          Quantize output tensors too\n"
              << "  --quantize-token-embd      Quantize token embedding tensors too\n"
              << "  --no-patch-file-type       Do not patch general.file_type to MOSTLY_TQ1_0\n";
}

options parse_args(int argc, char ** argv) {
    options opt;
    std::vector<std::string> positional;

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
        } else if (arg == "--dry-run") {
            opt.dry_run = true;
        } else if (arg == "--chunk-mib") {
            opt.chunk_mib = std::stoull(need_value("--chunk-mib"));
        } else if (arg == "--threshold-grid") {
            opt.threshold_grid = std::stoi(need_value("--threshold-grid"));
        } else if (arg == "--threshold-base") {
            opt.threshold_base = std::stof(need_value("--threshold-base"));
        } else if (arg == "--threshold-min") {
            opt.threshold_min = std::stof(need_value("--threshold-min"));
        } else if (arg == "--threshold-max") {
            opt.threshold_max = std::stof(need_value("--threshold-max"));
        } else if (arg == "--include-regex") {
            opt.include_regex = need_value("--include-regex");
        } else if (arg == "--exclude-regex") {
            opt.exclude_regex = need_value("--exclude-regex");
        } else if (arg == "--quantize-output") {
            opt.quantize_output = true;
        } else if (arg == "--quantize-token-embd") {
            opt.quantize_token_embd = true;
        } else if (arg == "--no-patch-file-type") {
            opt.patch_file_type = false;
        } else if (!arg.empty() && arg[0] == '-') {
            throw std::runtime_error("unknown option: " + arg);
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.size() != 2) {
        print_usage(argv[0]);
        throw std::runtime_error("expected input and output GGUF paths");
    }
    if (opt.threshold_grid < 1) {
        throw std::runtime_error("--threshold-grid must be >= 1");
    }
    if (opt.threshold_base < 0.0f || opt.threshold_min < 0.0f || opt.threshold_max < opt.threshold_min) {
        throw std::runtime_error("invalid threshold search range");
    }

    opt.input_path = positional[0];
    opt.output_path = positional[1];
    return opt;
}

void plan_quantization(gguf_model & model, const options & opt) {
    std::regex include_re;
    std::regex exclude_re;
    const bool use_include = !opt.include_regex.empty();
    const bool use_exclude = !opt.exclude_regex.empty();
    if (use_include) {
        include_re = std::regex(opt.include_regex);
    }
    if (use_exclude) {
        exclude_re = std::regex(opt.exclude_regex);
    }

    uint64_t offset = 0;
    for (tensor_info & tensor : model.tensors) {
        bool quantize = is_dense_float_type(tensor.input_type) && tensor.dims.size() >= 2 && tensor.row_elems % TQ_BLOCK_SIZE == 0;
        if (quantize && default_excluded(tensor, opt)) {
            quantize = false;
        }
        if (quantize && use_include && !std::regex_search(tensor.name, include_re)) {
            quantize = false;
        }
        if (quantize && use_exclude && std::regex_search(tensor.name, exclude_re)) {
            quantize = false;
        }

        tensor.quantize = quantize;
        tensor.output_type = quantize ? GGML_TYPE_TQ1_0 : tensor.input_type;
        tensor.output_nbytes = tensor_nbytes(tensor.output_type, tensor.dims);
        tensor.output_offset = offset;
        offset = align_up(offset + tensor.output_nbytes, model.alignment);
    }

    if (opt.patch_file_type && model.file_type_value_offset != std::numeric_limits<size_t>::max()) {
        patch_u32(model.metadata_bytes, model.file_type_value_offset, LLAMA_FTYPE_MOSTLY_TQ1_0);
    }
}

__device__ float load_dense_value(const uint8_t * input, uint64_t idx, uint32_t type) {
    if (type == GGML_TYPE_F32) {
        return reinterpret_cast<const float *>(input)[idx];
    }
    if (type == GGML_TYPE_F16) {
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

__device__ int ternary_signed(float v, float delta) {
    if (v > delta) {
        return 1;
    }
    if (v < -delta) {
        return -1;
    }
    return 0;
}

__device__ uint32_t ternary_code(float v, float delta) {
    return static_cast<uint32_t>(ternary_signed(v, delta) + 1);
}

__global__ void quantize_tq1_kernel(
        const uint8_t * input,
        uint8_t * output,
        uint64_t n_blocks,
        uint32_t input_type,
        int threshold_grid,
        float threshold_base,
        float threshold_min,
        float threshold_max) {
    const uint64_t block_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (block_id >= n_blocks) {
        return;
    }

    const uint64_t elem0 = block_id * TQ_BLOCK_SIZE;

    float sum_abs = 0.0f;
    for (int i = 0; i < static_cast<int>(TQ_BLOCK_SIZE); ++i) {
        sum_abs += fabsf(load_dense_value(input, elem0 + i, input_type));
    }

    const float base_delta = threshold_base * sum_abs / static_cast<float>(TQ_BLOCK_SIZE);
    float best_delta = base_delta;
    float best_alpha = 0.0f;
    float best_err = 3.402823466e+38F;

    for (int g = 0; g < threshold_grid; ++g) {
        const float t = threshold_grid == 1 ? 1.0f : static_cast<float>(g) / static_cast<float>(threshold_grid - 1);
        const float delta = base_delta * (threshold_min + (threshold_max - threshold_min) * t);

        float active_abs = 0.0f;
        int active_count = 0;
        for (int i = 0; i < static_cast<int>(TQ_BLOCK_SIZE); ++i) {
            const float v = load_dense_value(input, elem0 + i, input_type);
            if (fabsf(v) > delta) {
                active_abs += fabsf(v);
                active_count += 1;
            }
        }

        const float alpha = active_count > 0 ? active_abs / static_cast<float>(active_count) : 0.0f;
        float err = 0.0f;
        for (int i = 0; i < static_cast<int>(TQ_BLOCK_SIZE); ++i) {
            const float v = load_dense_value(input, elem0 + i, input_type);
            const int q = ternary_signed(v, delta);
            const float diff = v - alpha * static_cast<float>(q);
            err += diff * diff;
        }

        if (err < best_err) {
            best_err = err;
            best_delta = delta;
            best_alpha = alpha;
        }
    }

    uint8_t * dst = output + block_id * TQ_TYPE_SIZE;

    for (int m = 0; m < 32; ++m) {
        uint32_t q = 0;
        for (int n = 0; n < 5; ++n) {
            const float v = load_dense_value(input, elem0 + m + n * 32, input_type);
            q = q * 3u + ternary_code(v, best_delta);
        }
        dst[m] = encode_trits(q);
    }

    for (int m = 0; m < 16; ++m) {
        uint32_t q = 0;
        for (int n = 0; n < 5; ++n) {
            const float v = load_dense_value(input, elem0 + 160 + m + n * 16, input_type);
            q = q * 3u + ternary_code(v, best_delta);
        }
        dst[32 + m] = encode_trits(q);
    }

    for (int j = 0; j < 4; ++j) {
        uint32_t q = 0;
        for (int m = 0; m < 4; ++m) {
            const float v = load_dense_value(input, elem0 + 240 + j + m * 4, input_type);
            q = q * 3u + ternary_code(v, best_delta);
        }
        q *= 3u;
        dst[48 + j] = encode_trits(q);
    }

    __half h = __float2half(best_alpha);
    const uint16_t bits = reinterpret_cast<uint16_t *>(&h)[0];
    dst[52] = static_cast<uint8_t>(bits & 0xffu);
    dst[53] = static_cast<uint8_t>((bits >> 8) & 0xffu);
}

void quantize_tensor_cuda(
        std::ifstream & in,
        std::ofstream & out,
        const gguf_model & model,
        const tensor_info & tensor,
        const options & opt) {
    const uint64_t input_row_bytes = row_size_for_type(tensor.input_type, tensor.row_elems);
    const uint64_t output_row_bytes = row_size_for_type(tensor.output_type, tensor.row_elems);
    const uint64_t chunk_bytes_target = std::max<uint64_t>(1, opt.chunk_mib) * 1024ull * 1024ull;
    const uint64_t rows_per_chunk = std::max<uint64_t>(1, chunk_bytes_target / input_row_bytes);

    std::vector<uint8_t> host_input;
    std::vector<uint8_t> host_output;

    uint64_t rows_done = 0;
    while (rows_done < tensor.nrows) {
        const uint64_t rows = std::min<uint64_t>(rows_per_chunk, tensor.nrows - rows_done);
        const uint64_t input_bytes = checked_mul(rows, input_row_bytes, "input chunk size");
        const uint64_t output_bytes = checked_mul(rows, output_row_bytes, "output chunk size");
        const uint64_t n_blocks = output_bytes / TQ_TYPE_SIZE;

        host_input.resize(static_cast<size_t>(input_bytes));
        host_output.resize(static_cast<size_t>(output_bytes));

        const uint64_t input_abs = model.input_data_start + tensor.input_offset + rows_done * input_row_bytes;
        in.seekg(static_cast<std::streamoff>(input_abs), std::ios::beg);
        in.read(reinterpret_cast<char *>(host_input.data()), static_cast<std::streamsize>(input_bytes));
        if (!in) {
            throw std::runtime_error("failed to read tensor data for quantization: " + tensor.name);
        }

        uint8_t * dev_input = nullptr;
        uint8_t * dev_output = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_input), static_cast<size_t>(input_bytes)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_output), static_cast<size_t>(output_bytes)));
        CUDA_CHECK(cudaMemcpy(dev_input, host_input.data(), static_cast<size_t>(input_bytes), cudaMemcpyHostToDevice));

        constexpr int threads = 128;
        const uint64_t grid64 = (n_blocks + threads - 1) / threads;
        if (grid64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            CUDA_CHECK(cudaFree(dev_input));
            CUDA_CHECK(cudaFree(dev_output));
            throw std::runtime_error("tensor chunk has too many CUDA blocks; reduce --chunk-mib");
        }

        quantize_tq1_kernel<<<static_cast<uint32_t>(grid64), threads>>>(
                dev_input,
                dev_output,
                n_blocks,
                tensor.input_type,
                opt.threshold_grid,
                opt.threshold_base,
                opt.threshold_min,
                opt.threshold_max);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(host_output.data(), dev_output, static_cast<size_t>(output_bytes), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dev_input));
        CUDA_CHECK(cudaFree(dev_output));

        out.write(reinterpret_cast<const char *>(host_output.data()), static_cast<std::streamsize>(output_bytes));
        if (!out) {
            throw std::runtime_error("failed to write quantized tensor: " + tensor.name);
        }

        rows_done += rows;
    }
}

void write_output(const gguf_model & model, const options & opt) {
    std::ifstream in(opt.input_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to reopen input: " + opt.input_path);
    }
    std::ofstream out(opt.output_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open output: " + opt.output_path);
    }

    write_pod<uint32_t>(out, GGUF_MAGIC);
    write_pod<uint32_t>(out, model.version);
    write_pod<uint64_t>(out, model.tensor_count);
    write_pod<uint64_t>(out, model.metadata_count);
    if (!model.metadata_bytes.empty()) {
        out.write(reinterpret_cast<const char *>(model.metadata_bytes.data()), static_cast<std::streamsize>(model.metadata_bytes.size()));
        if (!out) {
            throw std::runtime_error("failed to write metadata");
        }
    }

    for (const tensor_info & tensor : model.tensors) {
        write_string(out, tensor.name);
        write_pod<uint32_t>(out, static_cast<uint32_t>(tensor.dims.size()));
        for (uint64_t dim : tensor.dims) {
            write_pod<uint64_t>(out, dim);
        }
        write_pod<uint32_t>(out, tensor.output_type);
        write_pod<uint64_t>(out, tensor.output_offset);
    }

    const uint64_t data_start = align_up(static_cast<uint64_t>(out.tellp()), model.alignment);
    write_padding(out, data_start - static_cast<uint64_t>(out.tellp()));

    for (size_t i = 0; i < model.tensors.size(); ++i) {
        const tensor_info & tensor = model.tensors[i];
        const uint64_t expected_pos = data_start + tensor.output_offset;
        const uint64_t current_pos = static_cast<uint64_t>(out.tellp());
        if (current_pos > expected_pos) {
            throw std::runtime_error("internal output offset error");
        }
        write_padding(out, expected_pos - current_pos);

        std::cerr << "[" << (i + 1) << "/" << model.tensors.size() << "] "
                  << (tensor.quantize ? "quant " : "copy  ")
                  << tensor.name << " " << dims_to_string(tensor.dims)
                  << " " << type_name(tensor.input_type) << " -> " << type_name(tensor.output_type)
                  << "\n";

        if (tensor.quantize) {
            quantize_tensor_cuda(in, out, model, tensor, opt);
        } else {
            const uint64_t input_abs = model.input_data_start + tensor.input_offset;
            in.seekg(static_cast<std::streamoff>(input_abs), std::ios::beg);
            copy_bytes(in, out, tensor.input_nbytes);
        }
    }
}

void print_plan(const gguf_model & model, const options & opt) {
    uint64_t quantized = 0;
    uint64_t copied = 0;
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;

    for (const tensor_info & tensor : model.tensors) {
        if (tensor.quantize) {
            quantized += 1;
        } else {
            copied += 1;
        }
        input_bytes += tensor.input_nbytes;
        output_bytes += tensor.output_nbytes;
    }

    std::cerr << "input:      " << opt.input_path << "\n";
    std::cerr << "output:     " << opt.output_path << "\n";
    std::cerr << "version:    " << model.version << "\n";
    std::cerr << "alignment:  " << model.alignment << "\n";
    std::cerr << "tensors:    " << model.tensors.size() << " (" << quantized << " quantized, " << copied << " copied)\n";
    std::cerr << "weights:    " << input_bytes << " bytes -> " << output_bytes << " bytes\n";
    if (opt.patch_file_type && model.file_type_value_offset == std::numeric_limits<size_t>::max()) {
        std::cerr << "note:       general.file_type metadata was not found; tensor types are still written as TQ1_0\n";
    }

    for (const tensor_info & tensor : model.tensors) {
        if (tensor.quantize || opt.dry_run) {
            std::cerr << (tensor.quantize ? "  quant " : "  copy  ")
                      << tensor.name << " " << dims_to_string(tensor.dims)
                      << " " << type_name(tensor.input_type) << " -> " << type_name(tensor.output_type)
                      << "\n";
        }
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const options opt = parse_args(argc, argv);
        gguf_model model = read_gguf_header(opt.input_path);
        plan_quantization(model, opt);
        print_plan(model, opt);
        if (!opt.dry_run) {
            write_output(model, opt);
        }
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
