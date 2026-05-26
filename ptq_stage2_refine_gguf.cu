#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
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

constexpr uint32_t GGUF_MAGIC = 0x46554747u;

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
    std::string include_regex;
    std::string exclude_regex;
    uint64_t chunk_mib = 512;
    int iterations = 8;
    float threshold_base = 0.75f;
    bool dry_run = false;
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
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
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
    if (magic != GGUF_MAGIC) {
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

void print_usage(const char * argv0) {
    std::cerr << "usage: " << argv0 << " --fp-model original.gguf --tq-model input-tq1_0.gguf --out refined.gguf [options]\n"
              << "\n"
              << "Second-pass CUDA refinement for TQ1_0 tensors. Requires the original dense GGUF.\n"
              << "This refines ternary assignments against original weights; it is not activation-based PT-BitNet block reconstruction.\n"
              << "\n"
              << "Options:\n"
              << "  --chunk-mib N           Tensor chunk size (default: 512)\n"
              << "  --iterations N          Lloyd/coordinate refinement iterations per TQ block (default: 8)\n"
              << "  --threshold-base X      Initial threshold = X * mean(abs(W)) (default: 0.75)\n"
              << "  --include-regex REGEX   Only refine matching tensor names\n"
              << "  --exclude-regex REGEX   Do not refine matching tensor names\n"
              << "  --dry-run               Report what would be refined without writing output\n";
}

options parse_args(int argc, char ** argv) {
    options opt;
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
        } else if (arg == "--out") {
            opt.output_model = need_value("--out");
        } else if (arg == "--chunk-mib") {
            opt.chunk_mib = std::stoull(need_value("--chunk-mib"));
        } else if (arg == "--iterations") {
            opt.iterations = std::stoi(need_value("--iterations"));
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

    if (opt.fp_model.empty() || opt.tq_model.empty() || opt.output_model.empty()) {
        print_usage(argv[0]);
        throw std::runtime_error("--fp-model, --tq-model and --out are required");
    }
    if (opt.chunk_mib < 1) {
        throw std::runtime_error("--chunk-mib must be >= 1");
    }
    if (opt.iterations < 1) {
        throw std::runtime_error("--iterations must be >= 1");
    }
    if (opt.threshold_base < 0.0f) {
        throw std::runtime_error("--threshold-base must be >= 0");
    }
    if (opt.output_model == opt.tq_model || opt.output_model == opt.fp_model) {
        throw std::runtime_error("--out must differ from input model paths");
    }
    return opt;
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

__global__ void refine_tq1_kernel(
        const uint8_t * input,
        uint8_t * output,
        uint64_t n_blocks,
        uint32_t input_type,
        int iterations,
        float threshold_base) {
    const uint64_t block_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (block_id >= n_blocks) {
        return;
    }

    const uint64_t elem0 = block_id * TQ_BLOCK_SIZE;
    float w[TQ_BLOCK_SIZE];
    int8_t q[TQ_BLOCK_SIZE];

    float sum_abs = 0.0f;
    for (int i = 0; i < static_cast<int>(TQ_BLOCK_SIZE); ++i) {
        const float v = load_dense_value(input, elem0 + i, input_type);
        w[i] = v;
        sum_abs += fabsf(v);
    }

    const float delta = threshold_base * sum_abs / static_cast<float>(TQ_BLOCK_SIZE);
    float active_abs = 0.0f;
    int active_count = 0;
    for (int i = 0; i < static_cast<int>(TQ_BLOCK_SIZE); ++i) {
        if (fabsf(w[i]) > delta) {
            q[i] = w[i] > 0.0f ? 1 : -1;
            active_abs += fabsf(w[i]);
            active_count += 1;
        } else {
            q[i] = 0;
        }
    }

    float alpha = active_count > 0 ? active_abs / static_cast<float>(active_count) : 0.0f;

    for (int it = 0; it < iterations; ++it) {
        float dot = 0.0f;
        int nnz = 0;
        for (int i = 0; i < static_cast<int>(TQ_BLOCK_SIZE); ++i) {
            dot += w[i] * static_cast<float>(q[i]);
            nnz += q[i] != 0 ? 1 : 0;
        }
        alpha = nnz > 0 ? fmaxf(0.0f, dot / static_cast<float>(nnz)) : 0.0f;

        int changes = 0;
        for (int i = 0; i < static_cast<int>(TQ_BLOCK_SIZE); ++i) {
            const int8_t nq = choose_ternary(w[i], alpha);
            changes += nq != q[i] ? 1 : 0;
            q[i] = nq;
        }
        if (changes == 0) {
            break;
        }
    }

    float dot = 0.0f;
    int nnz = 0;
    for (int i = 0; i < static_cast<int>(TQ_BLOCK_SIZE); ++i) {
        dot += w[i] * static_cast<float>(q[i]);
        nnz += q[i] != 0 ? 1 : 0;
    }
    alpha = nnz > 0 ? fmaxf(0.0f, dot / static_cast<float>(nnz)) : 0.0f;

    uint8_t * dst = output + block_id * TQ_TYPE_SIZE;

    for (int m = 0; m < 32; ++m) {
        uint32_t code = 0;
        for (int n = 0; n < 5; ++n) {
            code = code * 3u + static_cast<uint32_t>(q[m + n * 32] + 1);
        }
        dst[m] = encode_trits(code);
    }

    for (int m = 0; m < 16; ++m) {
        uint32_t code = 0;
        for (int n = 0; n < 5; ++n) {
            code = code * 3u + static_cast<uint32_t>(q[160 + m + n * 16] + 1);
        }
        dst[32 + m] = encode_trits(code);
    }

    for (int j = 0; j < 4; ++j) {
        uint32_t code = 0;
        for (int m = 0; m < 4; ++m) {
            code = code * 3u + static_cast<uint32_t>(q[240 + j + m * 4] + 1);
        }
        code *= 3u;
        dst[48 + j] = encode_trits(code);
    }

    __half h = __float2half(alpha);
    const uint16_t bits = reinterpret_cast<uint16_t *>(&h)[0];
    dst[52] = static_cast<uint8_t>(bits & 0xffu);
    dst[53] = static_cast<uint8_t>((bits >> 8) & 0xffu);
}

void refine_tensor_cuda(
        std::ifstream & fp_in,
        std::fstream & out,
        const gguf_model & fp_model,
        const gguf_model & tq_model,
        const tensor_info & fp_tensor,
        const tensor_info & tq_tensor,
        const options & opt) {
    const uint64_t input_row_bytes = row_size_for_type(fp_tensor.type, fp_tensor.row_elems);
    const uint64_t output_row_bytes = row_size_for_type(tq_tensor.type, tq_tensor.row_elems);
    const uint64_t rows_per_chunk = std::max<uint64_t>(1, (opt.chunk_mib * 1024ull * 1024ull) / input_row_bytes);

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
            throw std::runtime_error("failed to read original tensor: " + fp_tensor.name);
        }

        uint8_t * dev_input = nullptr;
        uint8_t * dev_output = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_input), static_cast<size_t>(input_bytes)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_output), static_cast<size_t>(output_bytes)));
        CUDA_CHECK(cudaMemcpy(dev_input, host_input.data(), static_cast<size_t>(input_bytes), cudaMemcpyHostToDevice));

        constexpr int threads = 128;
        const uint64_t grid64 = (n_blocks + threads - 1) / threads;
        refine_tq1_kernel<<<static_cast<uint32_t>(grid64), threads>>>(
                dev_input,
                dev_output,
                n_blocks,
                fp_tensor.type,
                opt.iterations,
                opt.threshold_base);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(host_output.data(), dev_output, static_cast<size_t>(output_bytes), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dev_input));
        CUDA_CHECK(cudaFree(dev_output));

        out.seekp(static_cast<std::streamoff>(tq_model.data_start + tq_tensor.offset + rows_done * output_row_bytes), std::ios::beg);
        out.write(reinterpret_cast<const char *>(host_output.data()), static_cast<std::streamsize>(output_bytes));
        if (!out) {
            throw std::runtime_error("failed to write refined tensor: " + tq_tensor.name);
        }

        rows_done += rows;
    }
}

bool should_consider(const tensor_info & tensor, const options & opt, const std::regex * include_re, const std::regex * exclude_re) {
    if (tensor.type != GGML_TYPE_TQ1_0) {
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

        struct pair_ref {
            const tensor_info * fp;
            const tensor_info * tq;
        };
        std::vector<pair_ref> refine;

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
            refine.push_back({ &fp_tensor, &tq_tensor });
        }

        std::cerr << "original: " << opt.fp_model << "\n";
        std::cerr << "ternary:  " << opt.tq_model << "\n";
        std::cerr << "output:   " << opt.output_model << "\n";
        std::cerr << "refine:   " << refine.size() << " TQ1_0 tensors\n";
        for (const pair_ref & p : refine) {
            std::cerr << "  refine " << p.tq->name << " " << dims_to_string(p.tq->dims)
                      << " " << type_name(p.fp->type) << " + TQ1_0 -> TQ1_0\n";
        }

        if (opt.dry_run) {
            return 0;
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
            std::cerr << "[" << (i + 1) << "/" << refine.size() << "] refine " << refine[i].tq->name << "\n";
            refine_tensor_cuda(fp_in, out, fp_model, tq_model, *refine[i].fp, *refine[i].tq, opt);
        }

        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
