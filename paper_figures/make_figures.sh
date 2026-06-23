#!/usr/bin/env bash
# Regenerate every figure and table the LASO paper uses, from paper_figures/plotdata.xlsx.
# Outputs land in paper_figures/figures/*.{pdf,png} and paper_figures/tables/*.tex.
#
#   bash paper_figures/make_figures.sh        # from the repo root
#
# Needs: pandas, openpyxl, numpy, matplotlib (e.g. the `sembench` conda env).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-laso}"
PY="${PYTHON:-python3}"

if [[ ! -f "$HERE/plotdata.xlsx" ]]; then
  echo "ERROR: $HERE/plotdata.xlsx not found." >&2
  exit 1
fi

for s in \
  plot_system_f1.py \
  plot_system_latency.py \
  plot_time_breakdown.py \
  plot_hparam_sensitivity.py \
  plot_model_generalization.py \
  plot_tables.py
do
  echo ">>> $s"
  "$PY" "$HERE/$s"
done

echo "Done. Figures in $HERE/figures, tables in $HERE/tables."
