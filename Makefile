NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo
LLAMA_DIR ?= /home/ckti/github/llama.cpp.all/consolidated
LLAMA_BUILD ?= $(LLAMA_DIR)/build-ternary-turboquant
LLAMA_LIBDIR ?= $(LLAMA_BUILD)/bin
LLAMA_INC := -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include
LLAMA_SO ?= $(notdir $(firstword $(sort $(wildcard $(LLAMA_LIBDIR)/libllama.so*))))
GGML_SO ?= $(notdir $(firstword $(sort $(wildcard $(LLAMA_LIBDIR)/libggml.so*))))
GGML_BASE_SO ?= $(notdir $(firstword $(sort $(wildcard $(LLAMA_LIBDIR)/libggml-base.so*))))
GGML_CPU_SO ?= $(notdir $(firstword $(sort $(wildcard $(LLAMA_LIBDIR)/libggml-cpu.so*))))
GGML_CUDA_SO ?= $(notdir $(firstword $(sort $(wildcard $(LLAMA_LIBDIR)/libggml-cuda.so*))))
LLAMA_LIBS := -L$(LLAMA_LIBDIR) -Xlinker -rpath -Xlinker $(LLAMA_LIBDIR) \
	-l:$(LLAMA_SO) \
	-l:$(GGML_SO) \
	-l:$(GGML_BASE_SO) \
	-l:$(GGML_CPU_SO) \
	-l:$(GGML_CUDA_SO) \
	-ldl -lpthread
TARGETS := ptq_ternary_gguf hf_to_ternary_gguf ptq_stage2_refine_gguf ptq_calibrate_reconstruct_gguf ptq_qat_ternary_gguf ptq_qat_trainer_gguf

.PHONY: all clean

all: $(TARGETS)

ptq_ternary_gguf: ptq_ternary_gguf.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

hf_to_ternary_gguf: hf_to_ternary_gguf.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

ptq_stage2_refine_gguf: ptq_stage2_refine_gguf.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

ptq_calibrate_reconstruct_gguf: ptq_calibrate_reconstruct_gguf.cu
	$(NVCC) $(NVCCFLAGS) $(LLAMA_INC) -o $@ $< $(LLAMA_LIBS)

ptq_qat_ternary_gguf: ptq_qat_ternary_gguf.cu
	$(NVCC) $(NVCCFLAGS) $(LLAMA_INC) -o $@ $< $(LLAMA_LIBS)

ptq_qat_trainer_gguf: ptq_qat_trainer_gguf.cu
	$(NVCC) $(NVCCFLAGS) $(LLAMA_INC) -o $@ $< $(LLAMA_LIBS)

clean:
	rm -f $(TARGETS)
