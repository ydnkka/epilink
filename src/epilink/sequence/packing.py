"""
Packed sequence storage and Hamming distance utilities.
"""

from __future__ import annotations

import textwrap
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
from numba import njit, prange

NDArrayInt8: TypeAlias = npt.NDArray[np.int8]
NDArrayUInt64: TypeAlias = npt.NDArray[np.uint64]
NDArrayInt32: TypeAlias = npt.NDArray[np.int32]


class SequencePacker64:
    """
    Utility engine for 64-bit packing and Hamming distance calculation.
    """

    @staticmethod
    @njit(parallel=True, cache=True, nogil=True)
    def pack_u64(arr: NDArrayInt8) -> NDArrayUInt64:
        """
        Pack 2-bit nucleotide arrays into 64-bit blocks.
        """

        N, L = arr.shape
        B = (L + 31) // 32
        out = np.zeros((N, B), dtype=np.uint64)

        for i in prange(N):
            for b in range(B):
                start = b * 32
                end = min(start + 32, L)
                word = np.uint64(0)
                shift = 62

                for k in range(start, end):
                    word |= np.uint64(arr[i, k]) << np.uint64(shift)
                    shift -= 2

                out[i, b] = word

        return out

    @staticmethod
    @njit(parallel=True, cache=True, nogil=True)
    def hamming64(packed: NDArrayUInt64) -> NDArrayInt32:
        """
        Compute pairwise Hamming distances for packed sequences.
        """

        M55 = np.uint64(0x5555555555555555)
        M33 = np.uint64(0x3333333333333333)
        M0F = np.uint64(0x0F0F0F0F0F0F0F0F)
        M01 = np.uint64(0x0101010101010101)

        N, B = packed.shape
        d = np.empty((N, N), dtype=np.int32)

        for i in prange(N):
            d[i, i] = 0

        for i in prange(N):
            pi = packed[i]
            for j in range(i + 1, N):
                pj = packed[j]
                total = 0

                for k in range(B):
                    x = pi[k] ^ pj[k]
                    diff = (x | (x >> 1)) & M55

                    c = diff
                    c = (c & M33) + ((c >> 2) & M33)
                    c = (c & M0F) + ((c >> 4) & M0F)
                    c = (c * M01) >> 56

                    total += c

                d[i, j] = np.int64(total)
                d[j, i] = np.int64(total)

        return d


def _ensure_pyfunc(func: Any) -> None:
    """Ensure a .py_func attribute for non-numba callables."""

    if not hasattr(func, "py_func"):
        func.py_func = func


_ensure_pyfunc(SequencePacker64.pack_u64)
_ensure_pyfunc(SequencePacker64.hamming64)


class PackedGenomicData:
    """
    Container for packed genomic sequences and lookup metadata.
    """

    def __init__(
        self,
        int8_matrix: NDArrayInt8,
        original_length: int,
        node_map: dict[str, int],
        base_map: dict[int, str],
    ):
        self.n_seqs: int
        self.original_length: int
        self.node_to_idx: dict[str, int]
        self.idx_to_node: dict[int, str]
        self.bases_map: dict[int, str]
        self.packed_u64: NDArrayUInt64

        if not np.issubdtype(int8_matrix.dtype, np.integer):
            raise TypeError("int8_matrix must be an integer array with values in {0, 1, 2, 3}.")
        if int8_matrix.size:
            min_val = int8_matrix.min()
            max_val = int8_matrix.max()
            if min_val < 0 or max_val > 3:
                raise ValueError("int8_matrix contains values outside {0, 1, 2, 3}.")
        if int8_matrix.dtype != np.int8 or not int8_matrix.flags["C_CONTIGUOUS"]:
            int8_matrix = np.ascontiguousarray(int8_matrix, dtype=np.int8)

        self.n_seqs, _ = int8_matrix.shape
        self.original_length = original_length
        self.node_to_idx = node_map
        self.idx_to_node = {v: k for k, v in node_map.items()}
        self.bases_map = base_map
        self.packed_u64 = SequencePacker64.pack_u64(int8_matrix)

    def compute_hamming_distances(self) -> NDArrayInt32:
        """
        Compute the pairwise Hamming distance matrix.
        """

        packed = (
            self.packed_u64
            if self.packed_u64.flags["C_CONTIGUOUS"]
            else np.ascontiguousarray(self.packed_u64)
        )
        return SequencePacker64.hamming64(packed)

    def write_fasta(self, filepath: str) -> None:
        """
        Write unpacked sequences to a FASTA file.
        """

        with open(filepath, "w") as f:
            for i in range(self.n_seqs):
                blocks = self.packed_u64[i]
                L = self.original_length
                unpacked = np.zeros(len(blocks) * 32, dtype=np.int8)

                idx = 0
                for w in blocks:
                    for shift in range(62, -1, -2):
                        unpacked[idx] = (w >> np.uint64(shift)) & np.uint64(3)
                        idx += 1

                seq = unpacked[:L]
                seq_str = "".join(self.bases_map[int(b)] for b in seq)

                header = f">{self.idx_to_node[i]}"
                body = textwrap.fill(seq_str, width=100)
                f.write(f"{header}\n{body}\n")


__all__ = ["PackedGenomicData", "SequencePacker64"]
