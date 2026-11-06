"""
PyTorch version of BUS-stop (Choi et al., ACL 2022).

The idea in one line: instead of using a validation set to decide when to stop
training, watch the model's predictions on unlabeled data.

There are two stages:
1. Preliminary stage: train a few times with a validation split. From this we
   get two reference signals:
     - p_l_conf : the confidence profile of the labeled data
     - c_u_cali : the calibrated class distribution of the unlabeled data
2. Main stage: train on all labeled data. At every epoch we compute two
   similarity scores and stop using the rule from Algorithm 2.

Note: we sort the unlabeled confidences before downsampling, as the paper's
Algorithm 2 says (the released Keras code does not do this).
"""

from __future__ import annotations

import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from criteria import (
    accuracy,
    class_calibration,
    cosine_sim,
    downsample_indices,
    euclidean,
    evenly_shuffle_indices,
    macro_f1,
)


def get_device(device=None):
    # Use the best device we have: CUDA, then Apple MPS, then CPU.
    if device in ("cpu", "cuda", "mps"):
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode(tokenizer, texts, max_len, device):
    # Turn a list of texts into BERT inputs (ids and masks) on the device.
    enc = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def predict_probs(model, enc, batch_size=128):
    # Run the model on all inputs (in small batches) and return probabilities.
    model.eval()
    n = enc["input_ids"].size(0)
    out = []
    for i in range(0, n, batch_size):
        b = {k: v[i:i + batch_size] for k, v in enc.items()}
        logits = model(**b)
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def eval_metrics(model, enc, y, batch_size=128):
    # Compute accuracy, macro F1 and loss on a dataset.
    probs = predict_probs(model, enc, batch_size)
    preds = probs.argmax(1)
    y = np.asarray(y)
    ce = float(np.mean(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0))))
    return {"acc": accuracy(y, preds), "f1": macro_f1(y, preds), "loss": ce,
            "probs": probs, "preds": preds}


def train_one_epoch(model, enc, y, optimizer, loss_fn, batch_size, seed, epoch, device):
    # One pass over the training data, using shuffled mini-batches.
    model.train()
    rng = np.random.RandomState(seed + epoch)
    idx = rng.permutation(len(y))
    y = np.asarray(y)
    for i in range(0, len(y), batch_size):
        b = idx[i:i + batch_size]
        xb = {k: v[b] for k, v in enc.items()}
        yb = torch.tensor(y[b], dtype=torch.long, device=device)
        optimizer.zero_grad()
        loss = loss_fn(model(**xb), yb)
        loss.backward()
        optimizer.step()


def train_with_val_stop(model, enc_trn, y_trn, enc_val, y_val,
                        epochs, batch_size, lr, patience, seed, device):
    # Train and stop early when the validation loss stops improving.
    # We keep the weights from the best epoch.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, eps=1e-8)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, enc_trn, y_trn, optimizer, loss_fn,
                        batch_size, seed, epoch, device)
        val_loss = eval_metrics(model, enc_val, y_val)["loss"]

        if val_loss < best_val_loss:
            # New best: save these weights and reset the counter.
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            # Not better. Count bad epochs in a row.
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss


class ModelFactory:
    # Makes a fresh copy of the model quickly.
    # We build the model once, then deep-copy it whenever we need a new one.

    def __init__(self, model_fn, device):
        self.template = model_fn()
        self.device = device

    def build(self):
        return copy.deepcopy(self.template).to(self.device)


def preliminary_stage(factory, tokenizer, texts_lab, y_lab, texts_unl, device,
                      max_len, epochs, patience, batch_size, lr, val_ratio, T, n_base,
                      cali_acc_or_f1="f1", bias_lab_or_val="val", seed=0, verbose=True):
    """
    Stage 1: get the two reference signals (p_l_conf and c_u_cali).

    We retrain T times. Each time we split the labeled data into train/val,
    train n_base models, and keep the best one on validation.
    """
    y_lab = np.asarray(y_lab)
    lab_len = len(y_lab)
    num_labels = int(y_lab.max()) + 1

    # Shuffle, then reorder so the classes are mixed evenly.
    rng = np.random.RandomState(seed)
    perm = rng.permutation(lab_len)
    order = perm[evenly_shuffle_indices(y_lab[perm])]

    trn_len = int(round(lab_len * (1.0 - val_ratio)))
    t_unit = lab_len // T

    # Store one confidence value per labeled sample for each run.
    sample_conf_mat = np.zeros((T, lab_len))
    records = {"val_acc": [], "val_f1": [], "val_dist": [], "ulb_dist": []}
    records["lab_dist"] = np.bincount(y_lab, minlength=num_labels) / lab_len

    enc_unl = encode(tokenizer, texts_unl, max_len, device)

    for t in range(T):
        # A moving window of indices goes to train; the rest go to validation.
        trn_idx = [i % lab_len for i in range(t * t_unit, t * t_unit + trn_len)]
        trn_set = set(trn_idx)
        val_idx = [i for i in range(lab_len) if i not in trn_set]

        trn_sel = [order[i] for i in trn_idx]
        val_sel = [order[i] for i in val_idx]

        enc_trn = encode(tokenizer, [texts_lab[i] for i in trn_sel], max_len, device)
        enc_val = encode(tokenizer, [texts_lab[i] for i in val_sel], max_len, device)
        y_trn = y_lab[trn_sel]
        y_val = y_lab[val_sel]

        # Train n_base models and keep the one with the lowest validation loss.
        best_val_loss = float("inf")
        best_state = None
        for base_i in range(n_base):
            model = factory.build()
            model, vloss = train_with_val_stop(
                model, enc_trn, y_trn, enc_val, y_val,
                epochs, batch_size, lr, patience, seed + base_i, device)
            if vloss < best_val_loss:
                best_val_loss = vloss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # Load the best model and collect its predictions.
        model = factory.build()
        model.load_state_dict(best_state)

        val_probs = predict_probs(model, enc_val)
        val_preds = val_probs.argmax(1)
        val_acc = accuracy(y_val, val_preds)
        val_f1 = macro_f1(y_val, val_preds)
        val_confs = val_probs.max(1)
        unl_probs = predict_probs(model, enc_unl)

        # Save the confidence of each validation sample.
        for v_i, v_conf in zip(val_idx, val_confs):
            sample_conf_mat[t, v_i] = v_conf

        records["val_acc"].append(val_acc)
        records["val_f1"].append(val_f1)
        records["val_dist"].append(val_probs.mean(0))
        records["ulb_dist"].append(unl_probs.mean(0))
        if verbose:
            print(f"  preliminary t={t}: val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

    # Average the confidence per labeled sample (skip samples never in val).
    counts = (sample_conf_mat != 0.0).astype(float).sum(0)
    counts[counts == 0] = 1.0
    records["sample_confs"] = sample_conf_mat.sum(0) / counts

    # Sorted reference confidences (this is p_l_conf / P_l in the paper).
    p_l_conf = np.sort(records["sample_confs"])

    # Calibrate the unlabeled class distribution for each run, then average.
    cali_dists = []
    for t in range(T):
        ulb_dist = records["ulb_dist"][t]
        val_perf = records["val_acc"][t] if cali_acc_or_f1 == "acc" else records["val_f1"][t]
        smp_bias = records["lab_dist"] if bias_lab_or_val == "lab" else records["val_dist"][t]
        cali_dists.append(class_calibration(ulb_dist, smp_bias, val_perf, num_labels))
    c_u_cali = np.mean(cali_dists, axis=0)

    return p_l_conf, c_u_cali, records


def main_stage(factory, tokenizer, texts_lab, y_lab, texts_unl, texts_tst, y_tst,
               p_l_conf, c_u_cali, device, max_len, epochs, batch_size, lr, n_que,
               seed=0, verbose=True):
    """
    Stage 2: train on all labeled data and stop with the BUS-stop rule.

    At each epoch:
      - s_conf  = distance between unlabeled confidences and the reference
      - s_class = similarity between the class distribution and the calibrated one
    We stop when s_conf has not improved for n_que epochs. In that time window,
    we keep the model with the best s_class.
    """
    y_lab = np.asarray(y_lab)
    y_tst = np.asarray(y_tst)
    lab_len = len(y_lab)
    unl_len = len(texts_unl)

    enc_lab = encode(tokenizer, texts_lab, max_len, device)
    enc_unl = encode(tokenizer, texts_unl, max_len, device)
    enc_tst = encode(tokenizer, texts_tst, max_len, device)

    model = factory.build()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, eps=1e-8)
    loss_fn = nn.CrossEntropyLoss()

    # A queue that keeps the last n_que s_class values (starts with zeros).
    queue = deque([0.0] * n_que, maxlen=n_que)
    best_conf = float("inf")
    n_pat = 0
    best_state = None
    stop_epoch = None
    ids = downsample_indices(unl_len, lab_len)

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, enc_lab, y_lab, optimizer, loss_fn,
                        batch_size, seed, epoch, device)

        unl_probs = predict_probs(model, enc_unl)
        unl_confs = np.sort(unl_probs.max(1))   # sort, as in Algorithm 2
        unl_dist = unl_probs.mean(0)

        s_conf = euclidean(unl_confs[ids], p_l_conf)
        s_class = cosine_sim(unl_dist, c_u_cali)

        if verbose:
            tst = eval_metrics(model, enc_tst, y_tst)
            print(f"  epoch {epoch:2d}: s_conf={s_conf:.4f} s_class={s_class:.4f} "
                  f"tst_acc={tst['acc']:.4f} tst_loss={tst['loss']:.4f}")

        # Track the lowest s_conf so far (lower is better).
        if s_conf < best_conf:
            n_pat = 0
            queue = deque([0.0] * n_que, maxlen=n_que)
            best_conf = s_conf
        else:
            n_pat += 1

        # While s_conf is still improving, keep the model with the best s_class.
        if n_pat < n_que:
            if s_class > max(queue):
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stop_epoch = epoch
            queue.append(s_class)
        else:
            # s_conf did not improve for n_que epochs, so we stop.
            break

    # If nothing was ever saved (very short run), keep the last model.
    if best_state is None:
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        stop_epoch = stop_epoch or epochs
    model.load_state_dict(best_state)

    final = eval_metrics(model, enc_tst, y_tst)
    final["stop_epoch"] = stop_epoch
    final["best_conf"] = best_conf
    return model, final
