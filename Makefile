NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo
TARGETS := ptq_ternary_gguf hf_to_ternary_gguf

.PHONY: all clean

all: $(TARGETS)

ptq_ternary_gguf: ptq_ternary_gguf.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

hf_to_ternary_gguf: hf_to_ternary_gguf.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
