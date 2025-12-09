import os
import sys

# Ensure we train with the sparse NDArray backend.
os.environ.setdefault("NEEDLE_NDARRAY_IMPL", "sparse")

sys.path.append("./python")
import needle as ndl

sys.path.append("./apps")
from models import LanguageModel
from simple_ml import train_ptb, evaluate_ptb

from memory_utils import MemoryTracker


def _resolve_device():
    """Prefer CUDA when available, otherwise fall back to CPU."""
    try:
        return ndl.cuda()
    except Exception:
        return ndl.cpu()


def main():
    tracker = MemoryTracker("Sparse PTB")
    tracker.checkpoint("script start")

    device = _resolve_device()
    tracker.checkpoint(f"device ready ({device})")

    corpus = ndl.data.Corpus("data/ptb")
    train_data = ndl.data.batchify(
        corpus.train, batch_size=8, device=device, dtype="float32"
    )
    tracker.checkpoint("data batchified")

    model = LanguageModel(
        20,
        len(corpus.dictionary),
        hidden_size=16,
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
