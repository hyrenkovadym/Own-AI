# Evaluation Guide

This document explains how to evaluate learning progress for the Snake module and how to compare training runs.

## 1. Run Training

Baseline training run:

```bash
python main.py train snake --episodes 100 --max-steps 250
```

Longer training runs:

```bash
python main.py train snake --episodes 500 --max-steps 250
python main.py train snake --episodes 3000 --max-steps 250
```

Optional metrics export:

```bash
python main.py train snake --episodes 500 --metrics-path runs/snake_training_metrics.csv
python main.py train snake --episodes 500 --metrics-path runs/snake_training_metrics.json
```

## 2. Run Evaluation (Play Mode)

```bash
python main.py play snake --episodes 20 --max-steps 250
```

Optional evaluation metrics export:

```bash
python main.py play snake --episodes 20 --metrics-path runs/snake_eval_metrics.csv
```

## 3. Metrics That Matter

Track these fields from training/evaluation output:
- `average score`
  - Training: `avg_score_last_100`
  - Evaluation: `avg_score`
- `best score`
- `wins`
- `total episodes`
- `q_states`
  - In this codebase, this is model size indicator (parameter count naming legacy).
- `epsilon start/end`
  - Training only (`epsilon_start`, `epsilon_end`)

## 4. Result Table Template

Do not fill with fake numbers. Use your real run outputs.

| Run | Episodes | Avg Score | Best Score | Wins | Total Episodes | q_states | Epsilon Start | Epsilon End | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Short | 100 | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | quick sanity check |
| Medium | 500 | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | expected early learning trend |
| Long | 3000 | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | stronger convergence signal |

## 5. Fresh Start vs Resumed Comparison

### Fresh start
```bash
python main.py train snake --episodes 500 --fresh-start --metrics-path runs/fresh_500.csv
python main.py play snake --episodes 20 --metrics-path runs/fresh_eval.csv
```

### Resumed
```bash
python main.py train snake --episodes 500 --metrics-path runs/resume_500.csv
python main.py play snake --episodes 20 --metrics-path runs/resume_eval.csv
```

Compare:
- evaluation average score
- best score
- wins
- stability across repeated eval runs

## 6. What Indicates Improvement

Strong indicators:
- higher average score over similar episode budgets
- best score increases
- wins appear more often
- resumed training outperforms fresh runs at equal or lower additional episodes

Potential warning signs:
- no average score improvement after many episodes
- wins remain zero in all runs
- performance collapses as epsilon decreases

## 7. Notes On Reproducibility
- Use fixed `--seed` when comparing experimental conditions.
- Keep board size (`--width`, `--height`) and step limits consistent between runs.
- Compare fresh and resumed runs on the same evaluation settings.
