# Build the batched cuFFT signal denoiser.
#
#   make            build bin/fft_denoise
#   make run ARGS="--num_signals=8192"
#   make clean      remove build output and results

NVCC      ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -Wno-deprecated-gpu-targets
LDLIBS    := -lcufft

TARGET  := bin/fft_denoise
SRCS    := src/main.cu src/gpu_denoise.cu src/cpu_fft.cpp src/signals.cpp
HEADERS := $(wildcard src/*.h)

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $@ $(LDLIBS)

run: $(TARGET)
	./$(TARGET) $(ARGS)

clean:
	rm -rf bin output
