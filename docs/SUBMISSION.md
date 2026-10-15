# Coursera submission notes

Numbers below are from the run in `output/` (Google Colab, Tesla T4).

## Project title

Batched GPU Signal Denoising with cuFFT

## Code project description (paste into the form)

For this capstone I built a GPU program that denoises a large batch of 1-D
signals with an FFT low-pass filter, and compared it against a CPU
implementation of the same algorithm. I ran it on Google Colab with a Tesla T4.

Each signal is a sum of three sine tones plus Gaussian noise, so the clean
signal is known and the result can be measured. On the GPU, all signals are
copied over in one transfer and processed with a single batched real-to-complex
cuFFT plan (cufftPlanMany), a custom CUDA kernel that zeroes every frequency bin
above the cutoff and rescales the rest, and a batched inverse FFT. The CPU
version uses a radix-2 Cooley-Tukey FFT I wrote myself and processes signals one
at a time. Every run checks the GPU output against the CPU output (max
difference 2e-5, all six runs verified), and the filter raises the average
signal-to-noise ratio from 3.7 dB to 15.9 dB, a gain of about 12 dB.

I ran a sweep from 256 to 65,536 signals of 1,024 samples each, up to 64
million samples in one run. At 65,536 signals the CPU took 3.57 s and the GPU
took 258 ms end to end, about 14x faster, and the speedup stayed around 13-16x
at every batch size. The per-stage timings explained why: the actual GPU
compute (forward FFT, filter kernel, inverse FFT) took only 7 ms, roughly 500x
faster than the CPU, while copying the data to and from the GPU took 251 ms,
about 97% of the GPU time. This was the most interesting result for me: for a
cheap per-element operation like this, the GPU is limited by the PCIe transfer,
not by computation.

Lessons learned: cuFFT's inverse transform is not normalized, so the output
comes back scaled by N unless you divide it out; I folded the 1/N factor into
the filter kernel so it costs no extra pass over the data. Using R2C/C2R
transforms instead of complex-to-complex halves the spectrum size, but means
the kernel works with N/2 + 1 bins per signal and the CPU reference has to zero
the mirrored bins to match. I also kept plan creation and CUDA context start-up
out of the timed section so the timings reflect the real work. Because copies
dominate, the next steps would be pinned host memory and CUDA streams to
overlap transfers with compute, or keeping data on the GPU across several
processing stages so the transfer cost is paid once.

## Proof of execution artifacts

Upload `output.zip` (run_log.txt, summary.csv, examples.csv,
example_signals.png, timing.png).

## Presentation outline (5-10 minutes)

1. **Goal (1 min).** Denoise thousands of signals at once; why batching suits
   the GPU.
2. **Algorithm (2 min).** FFT, zero high-frequency bins, inverse FFT. Show
   `example_signals.png`.
3. **Code walkthrough (2-3 min).** `cufftPlanMany` setup, `LowPassKernel`,
   CUDA-event timing, CPU FFT for verification, CLI options.
4. **Live demo (1-2 min).** Run the notebook cells in Colab; point at
   `VERIFIED` and the speedup lines.
5. **Results (1-2 min).** `timing.png` plus the per-stage numbers: 14x end to
   end, ~500x for compute alone, copies are 97% of GPU time. SNR 3.7 to 15.9 dB.
6. **Lessons and next steps (1 min).** 1/N normalization, R2C layout, timing
   pitfalls; pinned memory and streams to fix the copy bottleneck.
