// Copyright 2026 Divjot

#include "cpu_fft.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace denoise {

namespace {
constexpr double kPi = 3.14159265358979323846;
}  // namespace

void Fft(std::vector<std::complex<float>>* data, bool inverse) {
  std::vector<std::complex<float>>& a = *data;
  const int n = static_cast<int>(a.size());

  // Bit-reversal permutation.
  for (int i = 1, j = 0; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }

  // Butterflies. cuFFT's forward transform uses exp(-i...), inverse exp(+i...).
  for (int len = 2; len <= n; len <<= 1) {
    const double angle = 2.0 * kPi / len * (inverse ? 1.0 : -1.0);
    const std::complex<float> w_len(static_cast<float>(std::cos(angle)),
                                    static_cast<float>(std::sin(angle)));
    for (int i = 0; i < n; i += len) {
      std::complex<float> w(1.0f, 0.0f);
      for (int k = 0; k < len / 2; ++k) {
        const std::complex<float> u = a[i + k];
        const std::complex<float> v = a[i + k + len / 2] * w;
        a[i + k] = u + v;
        a[i + k + len / 2] = u - v;
        w *= w_len;
      }
    }
  }
}

void CpuLowPass(const std::vector<float>& input, int num_signals, int length,
                int cutoff, std::vector<float>* output) {
  output->assign(input.size(), 0.0f);
  std::vector<std::complex<float>> buffer(length);
  const float scale = 1.0f / static_cast<float>(length);

  for (int s = 0; s < num_signals; ++s) {
    const size_t offset = static_cast<size_t>(s) * length;
    for (int n = 0; n < length; ++n) {
      buffer[n] = std::complex<float>(input[offset + n], 0.0f);
    }
    Fft(&buffer, /*inverse=*/false);
    for (int k = 0; k < length; ++k) {
      if (std::min(k, length - k) >= cutoff) buffer[k] = 0.0f;
    }
    Fft(&buffer, /*inverse=*/true);
    for (int n = 0; n < length; ++n) {
      (*output)[offset + n] = buffer[n].real() * scale;
    }
  }
}

}  // namespace denoise
