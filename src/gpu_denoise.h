// Copyright 2026 Divjot
//
// GPU implementation: batched cuFFT low-pass filter.

#ifndef SRC_GPU_DENOISE_H_
#define SRC_GPU_DENOISE_H_

#include <string>
#include <vector>

namespace denoise {

// Per-stage GPU timings in milliseconds, measured with CUDA events.
struct GpuTimings {
  float h2d_ms = 0.0f;     // Host to device copy.
  float fft_ms = 0.0f;     // Batched forward R2C FFT.
  float filter_ms = 0.0f;  // Low-pass mask kernel.
  float ifft_ms = 0.0f;    // Batched inverse C2R FFT.
  float d2h_ms = 0.0f;     // Device to host copy.
  float total_ms = 0.0f;   // Everything above, end to end.
};

// Creates the CUDA context so its start-up cost is not counted in timings,
// and returns the name of the GPU in use.
std::string InitGpu();

// Low-pass filters all signals in one batched cuFFT pass. Produces the same
// result as CpuLowPass (up to float rounding).
GpuTimings GpuLowPass(const std::vector<float>& input, int num_signals,
                      int length, int cutoff, std::vector<float>* output);

}  // namespace denoise

#endif  // SRC_GPU_DENOISE_H_
