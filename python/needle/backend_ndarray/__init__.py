"""
Exports the selected NDArray implementation.

By default we expose the sparse backend. Setting the environment variable
NEEDLE_NDARRAY_IMPL=dense before importing needle switches back to the
original dense NDArray implementation.
"""
import os

_IMPL = os.environ.get("NEEDLE_NDARRAY_IMPL", "sparse").lower()

if _IMPL == "dense":
    from .ndarray import *

    _CSR_DISABLED_DEVICE = BackendDevice("csr", None)

    def csr():
        """Return a stub CSR device so sparse helpers can feature-detect support."""
        return _CSR_DISABLED_DEVICE
else:
    # Keep sparse as the default to avoid breaking existing behavior
    from .ndarray_sparse import *
