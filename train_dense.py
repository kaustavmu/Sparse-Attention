"""
Entry point to train the PTB model using the original dense NDArray backend.

This script mirrors train.py but forces NEEDLE_NDARRAY_IMPL=dense so we can
compare runtime/memory between dense and sparse implementations.
"""
import os
import sys

# Ensure the dense backend is selected before needle gets imported.
os.environ.setdefault("NEEDLE_NDARRAY_IMPL", "dense")

sys.path.append("./python")
import needle as ndl

sys.path.append("./apps")
from models import LanguageModel
from simple_ml import train_ptb, evaluate_ptb

from memory_utils import MemoryTracker


def _resolve_device():
    """Try CUDA first, otherwise fall back to CPU."""
    try:
        return ndl.cuda()
    except Exception:
        return ndl.cpu()


def main():
    tracker = MemoryTracker("Dense PTB")
    tracker.checkpoint("script start")

    device = _resolve_device()
    print(f"Using dense needle backend on device: {device}")
    tracker.checkpoint(f"device ready ({device})")

    corpus = ndl.data.Corpus("data/ptb")
    train_data = ndl.data.batchify(
        corpus.train, batch_size=32, device=device, dtype="float32"
    )
    tracker.checkpoint("data batchified")

    model = LanguageModel(
        20,
        len(corpus.dictionary),
        hidden_size=32,
        num_layers=1,
        seq_model="transformer",
        seq_len=20,
        device=device,
    )
    tracker.checkpoint("model initialized")

    train_ptb(
        model,
        train_data,
        seq_len=20,
        n_epochs=1,
        device=device,
        lr=0.003,
        optimizer=ndl.optim.Adam,
    )
    tracker.checkpoint("training finished")

    evaluate_ptb(model, train_data, seq_len=20, device=device)
    tracker.checkpoint("evaluation finished")

    tracker.report()


if __name__ == "__main__":
    main()

