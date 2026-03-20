from __future__ import annotations

import textwrap
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

NDArrayInt8: TypeAlias = npt.NDArray[np.int8]
NDArrayUInt64: TypeAlias = npt.NDArray[np.uint64]
NDArrayInt32: TypeAlias = npt.NDArray[np.int32]

_PACK_SHIFTS = np.arange(62, -2, -2, dtype=np.uint64)
_UINT64_BYTES = np.dtype(np.uint64).itemsize
_DEFAULT_HAMMING_WORKING_SET_BYTES = 32 * 1024 * 1024
_XOR_BYTE_TO_HAMMING = np.array(
    [bin((byte | (byte >> 1)) & 0x55).count("1") for byte in range(256)],
    dtype=np.uint8,
)


class SequencePacker64:
    """
    Utility engine for 64-bit packing and Hamming distance calculation.
    """

    @staticmethod
    def _resolve_block_size(
        num_sequences: int,
        bytes_per_sequence: int,
        block_size: int | None,
    ) -> int:
        if block_size is not None:
            return max(1, min(num_sequences, int(block_size)))
        if num_sequences <= 1 or bytes_per_sequence == 0:
            return max(1, num_sequences)

        max_pairs = max(
            1,
            _DEFAULT_HAMMING_WORKING_SET_BYTES // max(1, 2 * bytes_per_sequence),
        )
        return max(1, min(num_sequences, int(np.sqrt(max_pairs))))

    @staticmethod
    def pack_u64(arr: NDArrayInt8) -> NDArrayUInt64:
        """
        Pack 2-bit nucleotide arrays into 64-bit blocks.
        """

        n_sequences, sequence_length = arr.shape
        n_blocks = (sequence_length + 31) // 32
        out = np.zeros((n_sequences, n_blocks), dtype=np.uint64)

        if n_blocks == 0:
            return out

        padded = np.zeros((n_sequences, n_blocks * 32), dtype=np.uint8)
        padded[:, :sequence_length] = arr
        grouped = padded.reshape(n_sequences, n_blocks, 32)

        for position, shift in enumerate(_PACK_SHIFTS):
            out |= grouped[:, :, position].astype(np.uint64) << shift

        return out

    @staticmethod
    def hamming64(
        packed: NDArrayUInt64,
        block_size: int | None = None,
    ) -> NDArrayInt32:
        """
        Compute pairwise Hamming distances for packed sequences.

        Parameters
        ----------
        packed : numpy.ndarray
            Packed 64-bit representation of the sequences.
        block_size : int, optional
            Number of sequences to compare per row/column chunk. Smaller values
            reduce peak memory for large pairwise calculations.
        """

        packed = np.ascontiguousarray(packed, dtype=np.uint64)
        n_sequences, n_blocks = packed.shape
        distances = np.zeros((n_sequences, n_sequences), dtype=np.int32)

        if n_sequences == 0 or n_blocks == 0:
            return distances

        packed_bytes = packed.view(np.uint8).reshape(n_sequences, n_blocks * _UINT64_BYTES)
        step = SequencePacker64._resolve_block_size(
            n_sequences,
            packed_bytes.shape[1],
            block_size,
        )

        for row_start in range(0, n_sequences, step):
            row_stop = min(row_start + step, n_sequences)
            left = packed_bytes[row_start:row_stop]

            for col_start in range(row_start, n_sequences, step):
                col_stop = min(col_start + step, n_sequences)
                right = packed_bytes[col_start:col_stop]
                xor_bytes = np.bitwise_xor(left[:, np.newaxis, :], right[np.newaxis, :, :])
                block = _XOR_BYTE_TO_HAMMING[xor_bytes].sum(axis=2, dtype=np.int32)

                distances[row_start:row_stop, col_start:col_stop] = block
                if col_start != row_start:
                    distances[col_start:col_stop, row_start:row_stop] = block.T

        return distances


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

    def compute_hamming_distances(
        self,
        block_size: int | None = None,
    ) -> NDArrayInt32:
        """
        Compute the pairwise Hamming distance matrix.

        Parameters
        ----------
        block_size : int, optional
            Number of sequences to compare per row/column chunk. Smaller values
            reduce peak memory for large pairwise calculations.
        """

        packed = (
            self.packed_u64
            if self.packed_u64.flags["C_CONTIGUOUS"]
            else np.ascontiguousarray(self.packed_u64)
        )
        return SequencePacker64.hamming64(packed, block_size=block_size)

    def write_fasta(self, filepath: str) -> None:
        """
        Write unpacked sequences to a FASTA file.
        """

        base_lookup = np.array([self.bases_map[idx] for idx in range(4)], dtype="<U1")

        with open(filepath, "w") as f:
            for i in range(self.n_seqs):
                blocks = self.packed_u64[i]
                L = self.original_length
                unpacked = np.empty((len(blocks), 32), dtype=np.int8)

                for position, shift in enumerate(_PACK_SHIFTS):
                    unpacked[:, position] = ((blocks >> shift) & np.uint64(3)).astype(np.int8)

                seq = unpacked.reshape(len(blocks) * 32)[:L]
                seq_str = "".join(base_lookup[seq])

                header = f">{self.idx_to_node[i]}"
                body = textwrap.fill(seq_str, width=100)
                f.write(f"{header}\n{body}\n")


__all__ = ["PackedGenomicData", "SequencePacker64"]
