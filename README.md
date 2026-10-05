# AI651 Assignment 1 — Deep Learning for Space, Time and Graphs

## Layout

- `Question 1/` — Task 1 notebook (`Assignment1.ipynb`, executed with `PA1_PRESET=full`), supplied `harness/`, full-preset checkpoints, and exported outputs in `results/design/`. `Task1_Responses.md` holds the written responses.
- `Question 2 - Leaderboard/` — Task 2 Autoformer (`autoformer.py`), data, and results
  (ablation tables, figures, submission).
- `Report/` — LaTeX source (`report.tex`), figures, and compiled `report.pdf`.

## Setup

Python 3.11+:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r "Question 1/requirements.txt"
```

## Task 1

```bash
cd "Question 1"
PA1_PRESET=full jupyter nbconvert --to notebook --execute --inplace Assignment1.ipynb
```

Saved checkpoints are reused when present; delete `checkpoints/` to retrain.

## Task 2

```bash
cd "Question 2 - Leaderboard"
python autoformer.py ablate                          # 5 configs x seeds 0,1,2 -> results/ablation_*.csv
python autoformer.py final --ext future --epochs 4   # final model -> results/submission.txt
python autoformer.py figures                         # report figures -> results/fig_*.pdf
```

Declared leaderboard cost: **P = 22,081** trainable parameters, **E = 4** epochs.

The Autoformer is adapted from https://github.com/thuml/Autoformer (Wu et al., 2021).
