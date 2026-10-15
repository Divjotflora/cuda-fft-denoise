// Copyright 2026 Divjot
//
// CPU reference implementation: radix-2 FFT and FFT low-pass filter.

#ifndef SRC_CPU_FFT_H_
#define SRC_CPU_FFT_H_

#include <complex>
#include <vector>

namespace denoise {

// In-place iterative radix-2 Cooley-Tukey FFT. data->size() must be a power
// of two. The inverse transform is unnormalized (no 1/N), matching cuFFT.
void Fft(std::vector<std::complex<float>>* data, bool inverse);

// Low-pass filters every signal in `input` on the CPU: forward FFT, zero all
// frequency bins k with min(k, length - k) >= cutoff, inverse FFT, scale 1/N.
void CpuLowPass(const std::vector<float>& input, int num_signals, int length,
                int cutoff, std::vector<float>* output);

}  // namespace denoise

#endif  // SRC_CPU_FFT_H_
