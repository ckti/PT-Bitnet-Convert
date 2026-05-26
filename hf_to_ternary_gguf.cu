#include <cuda_runtime.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct options {
    std::string hf_dir;
    std::string output_gguf;
    std::string intermediate_gguf;
    std::string llama_cpp_dir = "";
    std::string converter_script;
    std::string ternary_converter;
    std::string python = "python3";
    std::string outtype = "f16";
    std::string model_name;
    std::string include_regex;
    std::string exclude_regex;
    std::string extra_hf_args;
    std::string extra_ternary_args;
    int chunk_mib = 512;
    bool keep_intermediate = false;
    bool dry_run = false;
    bool direct_python_tq = false;
    bool use_temp_file = false;
    bool no_lazy = false;
};

std::string shell_quote(const std::string & s) {
    if (s.empty()) {
        return "''";
    }

    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

int run_command(const std::string & cmd, bool dry_run) {
    std::cerr << "+ " << cmd << "\n";
    if (dry_run) {
        return 0;
    }

    const int rc = std::system(cmd.c_str());
    if (rc == -1) {
        throw std::runtime_error(std::string("failed to execute command: ") + std::strerror(errno));
    }
    return rc;
}

void check_success(int rc, const std::string & what) {
    if (rc != 0) {
        std::ostringstream oss;
        oss << what << " failed with exit status " << rc;
        throw std::runtime_error(oss.str());
    }
}

std::string default_intermediate_path(const std::string & output) {
    fs::path out(output);
    fs::path dir = out.parent_path();
    std::string stem = out.stem().string();
    if (stem.empty()) {
        stem = "model";
    }
    return (dir / (stem + ".intermediate-f16.gguf")).string();
}

void print_usage(const char * argv0) {
    std::cerr
        << "usage: " << argv0 << " --hf-dir MODEL_DIR --out OUTPUT.gguf [options]\n"
        << "\n"
        << "This NVCC-built driver converts a Hugging Face safetensors model directory to a\n"
        << "runnable ternary GGUF by using llama.cpp for HF metadata/tokenizer conversion\n"
        << "and ptq_ternary_gguf for CUDA TQ1_0 packing.\n"
        << "\n"
        << "Required:\n"
        << "  --hf-dir DIR                 Hugging Face model directory containing config/tokenizer/safetensors\n"
        << "  --out FILE                   final TQ1_0 GGUF path\n"
        << "\n"
        << "Paths:\n"
        << "  --llama-cpp-dir DIR          llama.cpp checkout with convert_hf_to_gguf.py\n"
        << "  --converter-script FILE      explicit convert_hf_to_gguf.py path\n"
        << "  --ternary-converter FILE     explicit ptq_ternary_gguf path\n"
        << "  --intermediate FILE          intermediate F16/BF16 GGUF path\n"
        << "  --python EXE                 Python executable (default: python3)\n"
        << "\n"
        << "Conversion options:\n"
        << "  --outtype f16|bf16|f32       intermediate precision (default: f16)\n"
        << "  --chunk-mib N                CUDA quantizer chunk size (default: 512)\n"
        << "  --include-regex REGEX        pass through to ptq_ternary_gguf\n"
        << "  --exclude-regex REGEX        pass through to ptq_ternary_gguf\n"
        << "  --model-name NAME            pass through to convert_hf_to_gguf.py\n"
        << "  --use-temp-file              pass through to convert_hf_to_gguf.py\n"
        << "  --no-lazy                    pass through to convert_hf_to_gguf.py\n"
        << "  --extra-hf-args TEXT         raw extra args for convert_hf_to_gguf.py\n"
        << "  --extra-ternary-args TEXT    raw extra args for ptq_ternary_gguf\n"
        << "  --direct-python-tq           use convert_hf_to_gguf.py --outtype tq1_0 directly\n"
        << "  --keep-intermediate          keep intermediate GGUF after successful conversion\n"
        << "  --dry-run                    print commands without running them\n";
}

options parse_args(int argc, char ** argv) {
    options opt;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char * flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + flag);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--hf-dir") {
            opt.hf_dir = value("--hf-dir");
        } else if (arg == "--out") {
            opt.output_gguf = value("--out");
        } else if (arg == "--intermediate") {
            opt.intermediate_gguf = value("--intermediate");
        } else if (arg == "--llama-cpp-dir") {
            opt.llama_cpp_dir = value("--llama-cpp-dir");
        } else if (arg == "--converter-script") {
            opt.converter_script = value("--converter-script");
        } else if (arg == "--ternary-converter") {
            opt.ternary_converter = value("--ternary-converter");
        } else if (arg == "--python") {
            opt.python = value("--python");
        } else if (arg == "--outtype") {
            opt.outtype = value("--outtype");
        } else if (arg == "--chunk-mib") {
            opt.chunk_mib = std::stoi(value("--chunk-mib"));
        } else if (arg == "--include-regex") {
            opt.include_regex = value("--include-regex");
        } else if (arg == "--exclude-regex") {
            opt.exclude_regex = value("--exclude-regex");
        } else if (arg == "--model-name") {
            opt.model_name = value("--model-name");
        } else if (arg == "--extra-hf-args") {
            opt.extra_hf_args = value("--extra-hf-args");
        } else if (arg == "--extra-ternary-args") {
            opt.extra_ternary_args = value("--extra-ternary-args");
        } else if (arg == "--keep-intermediate") {
            opt.keep_intermediate = true;
        } else if (arg == "--dry-run") {
            opt.dry_run = true;
        } else if (arg == "--direct-python-tq") {
            opt.direct_python_tq = true;
        } else if (arg == "--use-temp-file") {
            opt.use_temp_file = true;
        } else if (arg == "--no-lazy") {
            opt.no_lazy = true;
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (opt.hf_dir.empty() || opt.output_gguf.empty()) {
        print_usage("hf_to_ternary_gguf");
        throw std::runtime_error("--hf-dir and --out are required");
    }
    if (opt.outtype != "f16" && opt.outtype != "bf16" && opt.outtype != "f32") {
        throw std::runtime_error("--outtype must be f16, bf16, or f32 for the two-step CUDA path");
    }
    if (opt.chunk_mib < 1) {
        throw std::runtime_error("--chunk-mib must be >= 1");
    }

    if (opt.converter_script.empty()) {
        opt.converter_script = (fs::path(opt.llama_cpp_dir) / "convert_hf_to_gguf.py").string();
    }
    if (opt.ternary_converter.empty()) {
        opt.ternary_converter = "ptq_ternary_gguf";
    }
    if (opt.intermediate_gguf.empty()) {
        opt.intermediate_gguf = default_intermediate_path(opt.output_gguf);
    }

    return opt;
}

void validate_inputs(const options & opt) {
    if (!fs::is_directory(opt.hf_dir)) {
        throw std::runtime_error("HF model directory does not exist: " + opt.hf_dir);
    }
    if (!fs::is_regular_file(opt.converter_script)) {
        throw std::runtime_error("convert_hf_to_gguf.py not found: " + opt.converter_script);
    }
    if (!opt.direct_python_tq && !fs::is_regular_file(opt.ternary_converter)) {
        throw std::runtime_error("ptq_ternary_gguf not found: " + opt.ternary_converter + " (run make first)");
    }
}

std::string build_hf_command(const options & opt, const std::string & outfile, const std::string & outtype) {
    std::ostringstream cmd;
    cmd << shell_quote(opt.python) << " " << shell_quote(opt.converter_script)
        << " --outfile " << shell_quote(outfile)
        << " --outtype " << shell_quote(outtype);

    if (!opt.model_name.empty()) {
        cmd << " --model-name " << shell_quote(opt.model_name);
    }
    if (opt.use_temp_file) {
        cmd << " --use-temp-file";
    }
    if (opt.no_lazy) {
        cmd << " --no-lazy";
    }
    if (!opt.extra_hf_args.empty()) {
        cmd << " " << opt.extra_hf_args;
    }

    cmd << " " << shell_quote(opt.hf_dir);
    return cmd.str();
}

std::string build_ternary_command(const options & opt) {
    std::ostringstream cmd;
    cmd << shell_quote(opt.ternary_converter)
        << " --chunk-mib " << opt.chunk_mib;

    if (!opt.include_regex.empty()) {
        cmd << " --include-regex " << shell_quote(opt.include_regex);
    }
    if (!opt.exclude_regex.empty()) {
        cmd << " --exclude-regex " << shell_quote(opt.exclude_regex);
    }
    if (!opt.extra_ternary_args.empty()) {
        cmd << " " << opt.extra_ternary_args;
    }

    cmd << " " << shell_quote(opt.intermediate_gguf)
        << " " << shell_quote(opt.output_gguf);
    return cmd.str();
}

void print_cuda_info() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        std::cerr << "warning: no CUDA device reported now; the ternary conversion step will fail if CUDA is unavailable\n";
        return;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        std::cerr << "CUDA device 0: " << prop.name << "\n";
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        options opt = parse_args(argc, argv);
        validate_inputs(opt);
        print_cuda_info();

        if (opt.direct_python_tq) {
            const std::string cmd = build_hf_command(opt, opt.output_gguf, "tq1_0");
            check_success(run_command(cmd, opt.dry_run), "direct HF -> TQ1_0 GGUF conversion");
            return 0;
        }

        if (fs::equivalent(fs::path(opt.intermediate_gguf).parent_path().empty() ? fs::current_path() : fs::path(opt.intermediate_gguf).parent_path(),
                           fs::path(opt.output_gguf).parent_path().empty() ? fs::current_path() : fs::path(opt.output_gguf).parent_path()) &&
            fs::path(opt.intermediate_gguf).filename() == fs::path(opt.output_gguf).filename()) {
            throw std::runtime_error("intermediate path must differ from final output path");
        }

        const std::string hf_cmd = build_hf_command(opt, opt.intermediate_gguf, opt.outtype);
        check_success(run_command(hf_cmd, opt.dry_run), "HF -> intermediate GGUF conversion");

        const std::string ternary_cmd = build_ternary_command(opt);
        check_success(run_command(ternary_cmd, opt.dry_run), "intermediate GGUF -> CUDA TQ1_0 conversion");

        if (!opt.keep_intermediate && !opt.dry_run) {
            std::error_code ec;
            fs::remove(opt.intermediate_gguf, ec);
            if (ec) {
                std::cerr << "warning: failed to remove intermediate GGUF: " << ec.message() << "\n";
            }
        }

        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
