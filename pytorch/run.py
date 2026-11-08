"""
Command line script to run the BUS-stop reproduction on SST-2.

This uses the same arguments as the original main.py.

Example (run it from inside the pytorch/ folder):
    python run.py --data_dir ../bus-stop-keras/data --data_path bal/0 --seed 0

Fast smoke test (fewer epochs, one preliminary run):
    python run.py --data_dir ../bus-stop-keras/data --data_path bal/0 --seed 0 \
        --epochs 8 --T 1 --n_base 1
"""

import argparse
import os

import numpy as np
import torch
from transformers import AutoTokenizer

from bus_stop import ModelFactory, get_device, main_stage, preliminary_stage
from data import load_labeled, load_unlabeled
from model import BertClassifier


def _bool(v):
    # Read a true/false value from the command line.
    return str(v).lower() in ("1", "true", "yes", "y")


def main():
    p = argparse.ArgumentParser()

    # Data and model settings.
    p.add_argument("--task", default="SST-2")
    p.add_argument("--data_dir", default="../bus-stop-keras/data")
    p.add_argument("--data_path", default="bal/0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--word_freeze", type=_bool, default=True)

    # Training settings.
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)

    # Preliminary stage settings.
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--val_ratio", type=float, default=0.5)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--n_base", type=int, default=5)

    # Main stage settings.
    p.add_argument("--n_que", type=int, default=5)
    p.add_argument("--smooth_alpha", type=float, default=1.0,
                   help="EMA smoothing of stopping signals (1.0 = off)")
    p.add_argument("--cali_acc_or_f1", default="f1", choices=["acc", "f1"])
    p.add_argument("--bias_lab_or_val", default="val", choices=["lab", "val"])

    p.add_argument("--device", default="auto")
    args = p.parse_args()

    # Make runs repeatable.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.device)

    # Load the three data files.
    base = os.path.join(args.data_dir, args.task, args.data_path)
    texts_lab, y_lab = load_labeled(os.path.join(base, "labeled.tsv"))
    texts_tst, y_tst = load_labeled(os.path.join(base, "test_with_gold.tsv"))
    texts_unl = load_unlabeled(os.path.join(base, "unlabeled.tsv"))
    num_labels = int(max(y_lab)) + 1

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    factory = ModelFactory(
        lambda: BertClassifier(args.model_name, num_labels, args.dropout, args.word_freeze),
        device,
    )

    print(f"=== BUS-stop | task={args.task} path={args.data_path} seed={args.seed} "
          f"device={device} ===")
    print(f"labeled={len(y_lab)} unlabeled={len(texts_unl)} test={len(y_tst)} "
          f"num_labels={num_labels}")

    # Stage 1: get the reference signals.
    print("[preliminary stage]")
    p_l_conf, c_u_cali, _ = preliminary_stage(
        factory, tokenizer, texts_lab, y_lab, texts_unl, device,
        args.max_len, args.epochs, args.patience, args.batch_size, args.lr,
        args.val_ratio, args.T, args.n_base, args.cali_acc_or_f1,
        args.bias_lab_or_val, args.seed, verbose=True)
    print(f"  c_u_cali={np.round(c_u_cali, 4)}")

    # Stage 2: train on all labeled data and stop with BUS-stop.
    print("[main stage]")
    model, final = main_stage(
        factory, tokenizer, texts_lab, y_lab, texts_unl, texts_tst, y_tst,
        p_l_conf, c_u_cali, device, args.max_len, args.epochs, args.batch_size,
        args.lr, args.n_que, args.seed, args.smooth_alpha, verbose=True)

    print("=" * 64)
    print(f"RESULT path={args.data_path} seed={args.seed} alpha={args.smooth_alpha} "
          f"stop_epoch={final['stop_epoch']} "
          f"tst_acc={final['acc']:.4f} tst_f1={final['f1']:.4f} tst_loss={final['loss']:.4f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
