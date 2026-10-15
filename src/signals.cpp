// Copyright 2026 Divjot

#include "signals.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace denoise {

namespace {
constexpr int kTonesPerSignal = 3;
constexpr double kPi = 3.14159265358979323846;
}  // namespace

SignalBatch GenerateSignals(int num_signals, int length, int max_tone_bin,
                            float noise_std, unsigned int seed) {
  SignalBatch batch;
  const size_t total = static_cast<size_t>(num_signals) * length;
  batch.clean.assign(total, 0.0f);
  batch.noisy.assign(total, 0.0f);

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> bin_dist(1, max_tone_bin);
  std::uniform_real_distribution<float> amp_dist(0.5f, 1.5f);
  std::uniform_real_distribution<float> phase_dist(0.0f,
                                                   static_cast<float>(2 * kPi));
  std::normal_distribution<float> noise_dist(0.0f, noise_std);

  for (int s = 0; s < num_signals; ++s) {
    float* clean = batch.clean.data() + static_cast<size_t>(s) * length;
    float* noisy = batch.noisy.data() + static_cast<size_t>(s) * length;
    for (int t = 0; t < kTonesPerSignal; ++t) {
      const int bin = bin_dist(rng);
      const float amplitude = amp_dist(rng);
      const float phase = phase_dist(rng);
      for (int n = 0; n < length; ++n) {
        clean[n] += amplitude * static_cast<float>(std::sin(
                                    2.0 * kPi * bin * n / length + phase));
      }
    }
    for (int n = 0; n < length; ++n) {
      noisy[n] = clean[n] + noise_dist(rng);
    }
  }
  return batch;
}

double MeanSnrDb(const std::vector<float>& clean,
                 const std::vector<float>& test, int num_signals, int length) {
  double total_db = 0.0;
  for (int s = 0; s < num_signals; ++s) {
    const size_t offset = static_cast<size_t>(s) * length;
    double signal_power = 0.0;
    double error_power = 0.0;
    for (int n = 0; n < length; ++n) {
      const double c = clean[offset + n];
      const double e = test[offset + n] - c;
      signal_power += c * c;
      error_power += e * e;
    }
    error_power = std::max(error_power, 1e-20);
    total_db += 10.0 * std::log10(signal_power / error_power);
  }
  return total_db / num_signals;
}

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  float max_diff = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
  }
  return max_diff;
}

}  // namespace denoise
