"""
Read the data files from the original BUS-stop code.

The files look like this:
- labeled.tsv        : two columns "label<TAB>text" (first line is a header)
- test_with_gold.tsv : two columns "label<TAB>text" (first line is a header)
- unlabeled.tsv      : one column  "text"           (first line is a header)
"""

from __future__ import annotations


def _read_lines(path):
    # Read the file as a list of lines (without the trailing newline).
    with open(path, "r", errors="ignore") as f:
        return [ln.rstrip("\n") for ln in f]


def load_labeled(path):
    """
    Read a "label<TAB>text" file.

    Returns two lists: texts and integer labels.
    We skip the first line because it is a header.
    """
    lines = _read_lines(path)
    texts, labels = [], []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        parts = ln.split("\t")
        labels.append(int(parts[0]))
        texts.append(parts[1])
    return texts, labels


def load_unlabeled(path):
    # Read a one-column "text" file. Skip the header and empty lines.
    lines = _read_lines(path)
    return [ln for ln in lines[1:] if ln.strip()]
