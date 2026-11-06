# BUS-stop — PyTorch reimplementation

A clean PyTorch reimplementation of **"Early Stopping Based on Unlabeled
Samples in Text Classification"** (Choi et al., ACL 2022), used as the base
for a final-project reproduction + methodology improvement.

Original paper: <https://aclanthology.org/2022.acl-long.52/>
Original code:  <https://github.com/DMCB-GIST/BUS-stop>

## What this is

The original release is TensorFlow 2.2 / transformers 3.3.1 / Python 3.6,
which no longer runs on modern Colab (Python 3.12 + TF 2.20, Keras 3). This
folder reimplements the *method* in PyTorch (the framework the paper's idea is
agnostic to), keeping the hyperparameters and the two-stage algorithm.

## Layout

| File | Role |
|---|---|
| `criteria.py` | Framework-free math: calibration (Eq. 2), conf-sim/class-sim, downsampling, interleaved split. |
| `data.py` | Reads the paper's TSV files. |
| `model.py` | `BertClassifier` (BERT-base + head, dropout 0.2, word-embedding freeze). |
| `bus_stop.py` | Preliminary stage + main stage (Algorithm 2). |
| `run.py` | CLI reproducing the original `main.py` arguments. |
| `tests/test_criteria.py` | Unit tests for the framework-free math. |

## Quick start

```bash
pip install -r requirements.txt

# fast smoke test (reduced settings)
python run.py --data_dir ../bus-stop-keras/data --data_path bal/0 --seed 0 \
    --epochs 8 --T 1 --n_base 1

# full single-setting reproduction (SST-2 balanced, seed 0)
python run.py --data_dir ../bus-stop-keras/data --data_path bal/0 --seed 0
```

The data directory is the one shipped in the original repo
(`bus-stop-keras/data/SST-2/{bal,imbal}/<seed>/`).

## Notes / deviations

- We sort the unlabeled confidences before downsampling, following
  Algorithm 2 of the paper (the released Keras code does not).
- Exact decimals differ from the paper (different framework/hardware), but the
  method and hyperparameters match.
