// Copyright 2026 Divjot
//
// fft_denoise: removes high-frequency noise from a large batch of 1-D
// signals with a batched cuFFT low-pass filter, checks the result against a
// CPU FFT implementation, and reports timings and signal quality.
//
// Usage:
//   fft_denoise [--num_signals=N] [--length=L] [--cutoff=C] [--noise=S]
//               [--seed=X] [--output_dir=DIR] [--save_examples=K]

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "cpu_fft.h"
#include "gpu_denoise.h"
#include "signals.h"

namespace denoise {
namespace {

// Largest acceptable GPU-vs-CPU difference for a run to count as verified.
constexpr float kTolerance = 1e-3f;

struct Options {
  int num_signals = 4096;
  int length = 1024;
  int cutoff = 32;
  float noise = 0.8f;
  unsigned int seed = 42;
  std::string output_dir = "output";
  int save_examples = 0;
};

void PrintUsage(const char* program) {
  std::printf(
      "Usage: %s [options]\n"
      "  --num_signals=N    signals in the batch (default 4096)\n"
      "  --length=L         samples per signal, power of two (default 1024)\n"
      "  --cutoff=C         keep frequency bins below C (default 32)\n"
      "  --noise=S          Gaussian noise std-dev (default 0.8)\n"
      "  --seed=X           random seed (default 42)\n"
      "  --output_dir=DIR   where CSV results go (default output)\n"
      "  --save_examples=K  write the first K signals to examples.csv\n"
      "  --help             show this message\n",
      program);
}

bool IsPowerOfTwo(int x) { return x > 0 && (x & (x - 1)) == 0; }

// Parses --key=value flags. Exits with a message on bad input.
Options ParseArgs(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    const size_t eq = arg.find('=');
    if (arg.rfind("--", 0) != 0 || eq == std::string::npos) {
      std::fprintf(stderr, "Unrecognized argument: %s\n", arg.c_str());
      PrintUsage(argv[0]);
      std::exit(EXIT_FAILURE);
    }
    const std::string key = arg.substr(2, eq - 2);
    const std::string value = arg.substr(eq + 1);
    if (key == "num_signals") {
      opt.num_signals = std::atoi(value.c_str());
    } else if (key == "length") {
      opt.length = std::atoi(value.c_str());
    } else if (key == "cutoff") {
      opt.cutoff = std::atoi(value.c_str());
    } else if (key == "noise") {
      opt.noise = static_cast<float>(std::atof(value.c_str()));
    } else if (key == "seed") {
      opt.seed = static_cast<unsigned int>(std::strtoul(value.c_str(), 0, 10));
    } else if (key == "output_dir") {
      opt.output_dir = value;
    } else if (key == "save_examples") {
      opt.save_examples = std::atoi(value.c_str());
    } else {
      std::fprintf(stderr, "Unknown option: --%s\n", key.c_str());
      PrintUsage(argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }

  if (opt.num_signals <= 0 || !IsPowerOfTwo(opt.length) || opt.length < 8 ||
      opt.cutoff < 2 || opt.cutoff > opt.length / 2 || opt.noise < 0.0f) {
    std::fprintf(stderr,
                 "Invalid options: need num_signals > 0, length a power of "
                 "two >= 8, 2 <= cutoff <= length/2, noise >= 0.\n");
    std::exit(EXIT_FAILURE);
  }
  if (opt.save_examples > opt.num_signals) opt.save_examples = opt.num_signals;
  return opt;
}

bool FileExists(const std::string& path) {
  struct stat info;
  return stat(path.c_str(), &info) == 0;
}

// Appends one result row to summary.csv, writing the header on first use.
void AppendSummary(const Options& opt, double cpu_ms, const GpuTimings& gpu,
                   double snr_in, double snr_out, float max_diff) {
  const std::string path = opt.output_dir + "/summary.csv";
  const bool write_header = !FileExists(path);
  std::ofstream out(path, std::ios::app);
  if (write_header) {
    out << "num_signals,length,cutoff,noise,cpu_ms,gpu_total_ms,gpu_fft_ms,"
           "gpu_filter_ms,gpu_ifft_ms,gpu_copy_ms,speedup,snr_in_db,"
           "snr_out_db,max_abs_diff\n";
  }
  out << opt.num_signals << ',' << opt.length << ',' << opt.cutoff << ','
      << opt.noise << ',' << cpu_ms << ',' << gpu.total_ms << ','
      << gpu.fft_ms << ',' << gpu.filter_ms << ',' << gpu.ifft_ms << ','
      << (gpu.h2d_ms + gpu.d2h_ms) << ',' << cpu_ms / gpu.total_ms << ','
      << snr_in << ',' << snr_out << ',' << max_diff << '\n';
}

// Writes the first opt.save_examples signals, sample by sample.
void WriteExamples(const Options& opt, const SignalBatch& batch,
                   const std::vector<float>& cpu_out,
                   const std::vector<float>& gpu_out) {
  const std::string path = opt.output_dir + "/examples.csv";
  std::ofstream out(path);
  out << "signal_id,sample,clean,noisy,cpu_denoised,gpu_denoised\n";
  for (int s = 0; s < opt.save_examples; ++s) {
    for (int n = 0; n < opt.length; ++n) {
      const size_t i = static_cast<size_t>(s) * opt.length + n;
      out << s << ',' << n << ',' << batch.clean[i] << ',' << batch.noisy[i]
          << ',' << cpu_out[i] << ',' << gpu_out[i] << '\n';
    }
  }
  std::printf("Wrote %d example signals to %s\n", opt.save_examples,
              path.c_str());
}

int Run(int argc, char** argv) {
  const Options opt = ParseArgs(argc, argv);
  mkdir(opt.output_dir.c_str(), 0755);  // Fine if it already exists.

  std::printf("=== fft_denoise ===\n");
  std::printf("GPU: %s\n", InitGpu().c_str());
  std::printf("Signals: %d x %d samples, cutoff bin %d, noise std %.2f\n",
              opt.num_signals, opt.length, opt.cutoff, opt.noise);

  // Tones stay well inside the pass band so the filter keeps all of them.
  const int max_tone_bin = opt.cutoff / 2;
  const SignalBatch batch = GenerateSignals(opt.num_signals, opt.length,
                                            max_tone_bin, opt.noise, opt.seed);

  std::vector<float> cpu_out;
  const auto cpu_start = std::chrono::steady_clock::now();
  CpuLowPass(batch.noisy, opt.num_signals, opt.length, opt.cutoff, &cpu_out);
  const auto cpu_end = std::chrono::steady_clock::now();
  const double cpu_ms =
      std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

  std::vector<float> gpu_out;
  const GpuTimings gpu = GpuLowPass(batch.noisy, opt.num_signals, opt.length,
                                    opt.cutoff, &gpu_out);

  const float max_diff = MaxAbsDiff(cpu_out, gpu_out);
  const double snr_in =
      MeanSnrDb(batch.clean, batch.noisy, opt.num_signals, opt.length);
  const double snr_out =
      MeanSnrDb(batch.clean, gpu_out, opt.num_signals, opt.length);

  std::printf("CPU time:  %10.3f ms\n", cpu_ms);
  std::printf("GPU time:  %10.3f ms  (copy in %.3f, FFT %.3f, filter %.3f, "
              "IFFT %.3f, copy out %.3f)\n",
              gpu.total_ms, gpu.h2d_ms, gpu.fft_ms, gpu.filter_ms,
              gpu.ifft_ms, gpu.d2h_ms);
  std::printf("Speedup:   %10.2fx\n", cpu_ms / gpu.total_ms);
  std::printf("SNR:       %6.2f dB -> %6.2f dB (gain %.2f dB)\n", snr_in,
              snr_out, snr_out - snr_in);
  std::printf("Max |GPU - CPU| = %.2e -> %s\n", max_diff,
              max_diff < kTolerance ? "VERIFIED" : "MISMATCH");

  AppendSummary(opt, cpu_ms, gpu, snr_in, snr_out, max_diff);
  if (opt.save_examples > 0) WriteExamples(opt, batch, cpu_out, gpu_out);
  std::printf("\n");
  return max_diff < kTolerance ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace
}  // namespace denoise

int main(int argc, char** argv) { return denoise::Run(argc, argv); }
