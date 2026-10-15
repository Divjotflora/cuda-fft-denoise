// Copyright 2026 Divjot
//
// Synthetic signal generation and quality metrics.

#ifndef SRC_SIGNALS_H_
#define SRC_SIGNALS_H_

#include <vector>

namespace denoise {

// A batch of signals stored back to back: signal i occupies
// [i * length, (i + 1) * length).
struct SignalBatch {
  std::vector<float> clean;  // Ground truth: sum of sine tones.
  std::vector<float> noisy;  // clean + Gaussian noise.
};

// Generates num_signals signals of the given length. Each clean signal is a
// sum of three sine tones whose frequency bins lie in [1, max_tone_bin], so
// they survive a low-pass filter with cutoff > max_tone_bin.
SignalBatch GenerateSignals(int num_signals, int length, int max_tone_bin,
                            float noise_std, unsigned int seed);

// Mean signal-to-noise ratio in dB of `test` against `clean`, averaged over
// all signals in the batch.
double MeanSnrDb(const std::vector<float>& clean,
                 const std::vector<float>& test, int num_signals, int length);

// Largest absolute element-wise difference between two equal-size vectors.
float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b);

}  // namespace denoise

#endif  // SRC_SIGNALS_H_
