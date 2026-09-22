#!/bin/bash
# Merge one grid's per-job report.json files into a results table.
#
#   ./run_report.sh              # the main grid (pass2)
#   ./run_report.sh composer
#   ./run_report.sh v1 | ./run_report.sh topmagd | ./run_report.sh cipi_official
#
# Re-runnable at any time: it reads whatever has finished and says `--` for the
# rest, so a table can be built while half the grid is still queued and rebuilt
# later with more columns. It is also the only thing that turns results into
# numbers -- `report.format` is "json" in every recipe on purpose, because the
# library's markdown writer raises NotImplementedError.
#
# Prints the table and writes it next to the runs as CONSOLIDATED.md.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./env.sh

GRID="${1:-pass2}"
S="$PKG_ROOT/eval/benchmir"
PY="$(jr_python "$BENCHMIR_ENV")"

case "$GRID" in
  pass2)
    R="$RUN_ROOT/pass2"
    "$PY" "$S/scripts_ours/make_report_pass2.py" "$R" "$R/CONSOLIDATED.md" \
        --title 'Pre-trained arms, all tasks' ;;
  composer)
    R="$RUN_ROOT/composer"
    "$PY" "$S/scripts_ours/make_report_composer.py" "$R" "$R/CONSOLIDATED.md" \
        --jobs-dir "$JOBS_ROOT/composer" ;;
  v1)
    R="$RUN_ROOT/v1"
    "$PY" "$S/scripts_ours/make_report.py" "$R" "$R/CONSOLIDATED.md" ;;
  cipi_official)
    R="$RUN_ROOT/cipi_official"
    "$PY" "$S/scripts_ours/make_report_cipi_official.py" "$R/CONSOLIDATED.md" ;;
  pop909full)
    "$PY" "$S/scripts_ours/make_report_pop909full.py" ;;
  topmagd)
    # split_stats.py is the only producer of $WORK_ROOT/topmagd_split_stats.json,
    # which the report reads; run it first or the report has nothing to read.
    "$PY" "$S/scripts_topmagd/split_stats.py"
    "$PY" "$S/scripts_topmagd/make_report_topmagd_splits.py" ;;
  *)
    echo "usage: ./run_report.sh [pass2|v1|composer|cipi_official|pop909full|topmagd]" >&2
    exit 2 ;;
esac
