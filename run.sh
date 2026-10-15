#!/usr/bin/env bash
# Builds the project, runs it on batches of increasing size, and saves all
# results (CSV, log, plots) to output/.
set -euo pipefail
cd "$(dirname "$0")"

OUT=output
LOG="$OUT/run_log.txt"
LENGTH=1024
CUTOFF=32

make
mkdir -p "$OUT"
rm -f "$OUT/summary.csv"
: > "$LOG"

# Scaling sweep: from a few hundred to ~65k signals (64M samples).
for n in 256 1024 4096 16384 65536; do
  ./bin/fft_denoise --num_signals="$n" --length="$LENGTH" \
      --cutoff="$CUTOFF" --output_dir="$OUT" | tee -a "$LOG"
done

# One more run that also saves a few signals for the before/after plots.
./bin/fft_denoise --num_signals=4096 --length="$LENGTH" --cutoff="$CUTOFF" \
    --output_dir="$OUT" --save_examples=4 | tee -a "$LOG"

if python3 -c "import matplotlib" 2>/dev/null; then
  python3 scripts/plot_results.py "$OUT"
else
  echo "matplotlib not found; skipping plots (pip install matplotlib)."
fi

echo "Done. Results are in $OUT/"
