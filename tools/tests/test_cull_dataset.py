"""Unit tests for cull_dataset's pure scorers (no Pillow/numpy, no filesystem)."""
import cull_dataset as C


def test_hamming():
    assert C.hamming(0b1010, 0b1010) == 0
    assert C.hamming(0b1010, 0b1000) == 1
    assert C.hamming(0b0000, 0b1111) == 4


def test_ahash_bits_and_length():
    # first 32 px below mean -> 0 bits, last 32 at/above mean -> 1 bits (MSB = first px)
    assert C.ahash([0] * 32 + [255] * 32) == 0xFFFFFFFF
    # uniform image -> every px >= mean -> all ones (64 bits set)
    assert C.ahash([100] * 64) == (1 << 64) - 1
    try:
        C.ahash([0, 1, 2])            # wrong length must raise
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_laplacian_var_sharp_beats_flat():
    flat = [[128] * 8 for _ in range(8)]
    checker = [[(255 if (x + y) % 2 else 0) for x in range(8)] for y in range(8)]
    assert C.laplacian_var(flat) == 0.0                       # no edges -> zero focus
    assert C.laplacian_var(checker) > C.laplacian_var(flat)   # hard edges -> high focus
    assert C.laplacian_var([[1, 2]]) == 0.0                   # too small -> 0, no crash


def test_find_duplicates_keeps_earliest():
    # idx0 and idx1 identical (dist 0); idx2 far (dist 4) -> drop only idx1
    drop = C.find_duplicates([0b0000, 0b0000, 0b1111], thresh=1)
    assert drop == {1}
    # within-threshold near-dup also dropped; earliest kept
    drop2 = C.find_duplicates([0b0000, 0b0001, 0b0011], thresh=1)
    assert 1 in drop2 and 0 not in drop2          # 1 is dist-1 from 0 -> dropped; 0 kept
    # nothing dedups when all far apart
    assert C.find_duplicates([0b0000, 0b1111], thresh=2) == set()
