#!/bin/bash
# Concatenate report_sections/*.md into REPORT.md (edit the sections, then rerun).
cd "$(dirname "$0")/.."
S=report_sections
{
  for f in head changed verify_repro verify_confounds verify_confirm followup redesign testbed methods methodology literature next appendix_pilot; do
    [ -f "$S/$f.md" ] && { cat "$S/$f.md"; echo; }
  done
} > REPORT.md
echo "REPORT.md: $(wc -w < REPORT.md) words"
