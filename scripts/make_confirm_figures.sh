#!/bin/bash
# Figures for the confirmation runs (decayed logs hold the full trajectory: stable phase + decay).
cd "$(dirname "$0")/.."
mkdir -p figures/confirm
/opt/venv/bin/python scripts/analyze_confirm.py runs/confirm figures/confirm > figures/confirm/confirm_table.md
/opt/venv/bin/python scripts/plot_pilot.py 'runs/confirm/*_decay.json' figures/confirm --seed=10
/opt/venv/bin/python scripts/garg_eval.py figures/confirm/garg_eval_confirm.json runs/confirm/adamw_iso_s1[0-2]_decay.json runs/confirm/muon_iso_s1[0-2]_decay.json > /dev/null
/opt/venv/bin/python scripts/skewed_summary.py figures/confirm/garg_eval_confirm.json > figures/confirm/skewed_summary.txt
/opt/venv/bin/python scripts/heads_report.py 'runs/confirm/*_decay.json' figures/confirm > figures/confirm/heads.txt 2>&1
/opt/venv/bin/python -c "
import sys; sys.path.insert(0, 'scripts'); from plot_repro import fig_garg
fig_garg('figures/confirm/garg_eval_confirm.json', 'figures/confirm/garg_eval_confirm.png')"
echo done
