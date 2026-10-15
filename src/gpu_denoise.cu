// Copyright 2026 Divjot

#include "gpu_denoise.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstdio>
#include <cstdlib>

namespace denoise {

namespace {

#define CUDA_CHECK(call)                                                 \
  do {                                                                   \
    cudaError_t err_ = (call);                                           \
    if (err_ != cudaSuccess) {                                           \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                   cudaGetErrorString(err_), __FILE__, __LINE__);        \
      std::exit(EXIT_FAILURE);                                           \
    }                                                                    \
  } while (0)

#define CUFFT_CHECK(call)                                                \
  do {                                                                   \
    cufftResult res_ = (call);                                           \
    if (res_ != CUFFT_SUCCESS) {                                         \
      std::fprintf(stderr, "cuFFT error %d at %s:%d\n",                  \
                   static_cast<int>(res_), __FILE__, __LINE__);          \
      std::exit(EXIT_FAILURE);                                           \
    }                                                                    \
  } while (0)

constexpr int kThreadsPerBlock = 256;
constexpr int kNumEvents = 6;

// One thread per complex frequency bin across the whole batch. Bins at or
// above the cutoff are zeroed; the rest are scaled by 1/N so the inverse FFT
// returns correctly normalized samples.
__global__ void LowPassKernel(cufftComplex* spectrum, int num_bins,
                              long long total_bins, int cutoff, float scale) {
  const long long idx =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_bins) return;
  const int bin = static_cast<int>(idx % num_bins);
  if (bin >= cutoff) {
    spectrum[idx].x = 0.0f;
    spectrum[idx].y = 0.0f;
  } else {
    spectrum[idx].x *= scale;
    spectrum[idx].y *= scale;
  }
}

}  // namespace

std::string InitGpu() {
  CUDA_CHECK(cudaFree(nullptr));  // Forces context creation.
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  return std::string(prop.name);
}

GpuTimings GpuLowPass(const std::vector<float>& input, int num_signals,
                      int length, int cutoff, std::vector<float>* output) {
  const int num_bins = length / 2 + 1;  // R2C output size per signal.
  const size_t real_bytes =
      sizeof(float) * static_cast<size_t>(num_signals) * length;
  const long long total_bins = static_cast<long long>(num_signals) * num_bins;
  const size_t complex_bytes = sizeof(cufftComplex) * total_bins;

  float* d_signal = nullptr;
  cufftComplex* d_spectrum = nullptr;
  CUDA_CHECK(cudaMalloc(&d_signal, real_bytes));
  CUDA_CHECK(cudaMalloc(&d_spectrum, complex_bytes));

  // One plan transforms every signal in the batch in a single call.
  int n[1] = {length};
  cufftHandle forward_plan;
  cufftHandle inverse_plan;
  CUFFT_CHECK(cufftPlanMany(&forward_plan, 1, n, nullptr, 1, length, nullptr,
                            1, num_bins, CUFFT_R2C, num_signals));
  CUFFT_CHECK(cufftPlanMany(&inverse_plan, 1, n, nullptr, 1, num_bins,
                            nullptr, 1, length, CUFFT_C2R, num_signals));

  cudaEvent_t events[kNumEvents];
  for (cudaEvent_t& e : events) CUDA_CHECK(cudaEventCreate(&e));

  CUDA_CHECK(cudaEventRecord(events[0]));
  CUDA_CHECK(cudaMemcpy(d_signal, input.data(), real_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(events[1]));

  CUFFT_CHECK(cufftExecR2C(forward_plan, d_signal, d_spectrum));
  CUDA_CHECK(cudaEventRecord(events[2]));

  const int blocks = static_cast<int>(
      (total_bins + kThreadsPerBlock - 1) / kThreadsPerBlock);
  LowPassKernel<<<blocks, kThreadsPerBlock>>>(
      d_spectrum, num_bins, total_bins, cutoff, 1.0f / length);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(events[3]));

  CUFFT_CHECK(cufftExecC2R(inverse_plan, d_spectrum, d_signal));
  CUDA_CHECK(cudaEventRecord(events[4]));

  output->resize(input.size());
  CUDA_CHECK(cudaMemcpy(output->data(), d_signal, real_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(events[5]));
  CUDA_CHECK(cudaEventSynchronize(events[5]));

  GpuTimings t;
  CUDA_CHECK(cudaEventElapsedTime(&t.h2d_ms, events[0], events[1]));
  CUDA_CHECK(cudaEventElapsedTime(&t.fft_ms, events[1], events[2]));
  CUDA_CHECK(cudaEventElapsedTime(&t.filter_ms, events[2], events[3]));
  CUDA_CHECK(cudaEventElapsedTime(&t.ifft_ms, events[3], events[4]));
  CUDA_CHECK(cudaEventElapsedTime(&t.d2h_ms, events[4], events[5]));
  CUDA_CHECK(cudaEventElapsedTime(&t.total_ms, events[0], events[5]));

  for (cudaEvent_t& e : events) CUDA_CHECK(cudaEventDestroy(e));
  CUFFT_CHECK(cufftDestroy(forward_plan));
  CUFFT_CHECK(cufftDestroy(inverse_plan));
  CUDA_CHECK(cudaFree(d_signal));
  CUDA_CHECK(cudaFree(d_spectrum));
  return t;
}

}  // namespace denoise
