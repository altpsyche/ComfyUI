"""Mechanical curation aid for a character dataset folder.

Flags the two tedious-by-hand culls on `output/dataset/<char>/`:
  - BLURRY frames (low focus measure — variance of a Laplacian on a downscaled grayscale), and
  - NEAR-DUPLICATE frames (average-hash within a small Hamming distance — keeps one per group).

It does NOT judge identity / off-model / anatomy — that's still your eyeball pass (the tool can't tell
*which* face is the right one). Think of it as the first sweep that clears the obvious junk so you
curate fewer frames by hand.

Non-destructive: by default it only REPORTS. Pass --apply to move the flagged frames into a
`_rejected/` subfolder (reversible — drag them back). `train_lora` counts only top-level images, so
`_rejected/` is ignored automatically.

Usage:
  python cull_dataset.py aria                 # report on output/dataset/aria/
  python cull_dataset.py aria --apply         # move blurry/dup frames to _rejected/
  python cull_dataset.py <path> --blur 80 --dup 6
  python cull_dataset.py aria --no-dup        # blur only

The pure scoring functions (ahash / hamming / laplacian_var / find_duplicates) take plain Python
lists/ints so they're unit-tested without Pillow or numpy.
"""
from __future__ import annotations
import argparse
import pathlib
import sys

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
HASH_SIDE = 8          # average-hash grid -> 64-bit hash
BLUR_GRAY = 64         # downscale edge for the focus measure (cheap + scale-stable)
DEFAULT_BLUR = 60.0    # focus measure below this = "blurry" (relative knob — calibrate per dataset)
DEFAULT_DUP = 5        # Hamming distance <= this between aHashes = "near-duplicate"


# ---- pure scoring (no Pillow/numpy; lists + ints) ---------------------------------------------------
def ahash(gray_flat: list[int], side: int = HASH_SIDE) -> int:
    """Average hash of a side*side flat grayscale list: bit set where pixel >= mean. MSB = first pixel."""
    n = side * side
    if len(gray_flat) != n:
        raise ValueError(f"ahash expects {n} values, got {len(gray_flat)}")
    mean = sum(gray_flat) / n
    bits = 0
    for px in gray_flat:
        bits = (bits << 1) | (1 if px >= mean else 0)
    return bits


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return bin(a ^ b).count("1")


def laplacian_var(gray2d: list[list[int]]) -> float:
    """Focus measure = variance of a 4-neighbour Laplacian over interior pixels. Higher = sharper."""
    h = len(gray2d)
    w = len(gray2d[0]) if h else 0
    if h < 3 or w < 3:
        return 0.0
    lap = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            lap.append(4 * gray2d[y][x] - gray2d[y - 1][x] - gray2d[y + 1][x]
                       - gray2d[y][x - 1] - gray2d[y][x + 1])
    m = sum(lap) / len(lap)
    return sum((v - m) ** 2 for v in lap) / len(lap)


def find_duplicates(hashes: list[int], thresh: int = DEFAULT_DUP) -> set[int]:
    """Indices to drop: for each near-group (Hamming <= thresh) keep the earliest, flag the rest."""
    drop: set[int] = set()
    for i in range(len(hashes)):
        if i in drop:
            continue
        for j in range(i + 1, len(hashes)):
            if j not in drop and hamming(hashes[i], hashes[j]) <= thresh:
                drop.add(j)
    return drop


# ---- I/O layer (Pillow, imported lazily so the module imports without it) ---------------------------
def _load_gray(path, side_hash=HASH_SIDE, side_blur=BLUR_GRAY):
    from PIL import Image
    im = Image.open(path).convert("L")
    hflat = list(im.resize((side_hash, side_hash), Image.BILINEAR).getdata())
    b = im.resize((side_blur, side_blur), Image.BILINEAR)
    bpx = list(b.getdata())
    gray2d = [bpx[r * side_blur:(r + 1) * side_blur] for r in range(side_blur)]
    return hflat, gray2d


def main(argv=None):
    ap = argparse.ArgumentParser(description="Flag blurry / near-duplicate frames in a dataset folder.")
    ap.add_argument("target", help="character name (-> output/dataset/<name>/) or a folder path")
    ap.add_argument("--apply", action="store_true", help="move flagged frames to _rejected/ (default: report only)")
    ap.add_argument("--blur", type=float, default=DEFAULT_BLUR, help=f"focus floor; below = blurry (default {DEFAULT_BLUR})")
    ap.add_argument("--dup", type=int, default=DEFAULT_DUP, help=f"max Hamming distance for near-dup (default {DEFAULT_DUP})")
    ap.add_argument("--no-blur", action="store_true", help="skip the blur check")
    ap.add_argument("--no-dup", action="store_true", help="skip the duplicate check")
    args = ap.parse_args(argv)

    folder = pathlib.Path(args.target)
    if not folder.exists():
        folder = pathlib.Path(__file__).resolve().parents[2] / "output" / "dataset" / args.target
    if not folder.is_dir():
        raise SystemExit(f"no dataset folder: {folder}")

    imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise SystemExit(f"no images in {folder}")

    hashes, reasons = [], {}
    for p in imgs:
        try:
            hflat, gray2d = _load_gray(p)
        except Exception as e:                       # unreadable image -> flag it
            hashes.append(0)
            reasons[p.name] = f"unreadable ({e})"
            continue
        hashes.append(ahash(hflat))
        if not args.no_blur:
            fm = laplacian_var(gray2d)
            if fm < args.blur:
                reasons[p.name] = f"blurry (focus {fm:.0f} < {args.blur:.0f})"

    if not args.no_dup:
        dups = find_duplicates(hashes, args.dup)
        for idx in dups:
            reasons.setdefault(imgs[idx].name, "near-duplicate")

    keep = [p for p in imgs if p.name not in reasons]
    print(f"{folder}: {len(imgs)} images -> keep {len(keep)}, flag {len(reasons)}")
    for name in sorted(reasons):
        print(f"  [flag] {name}: {reasons[name]}")

    if args.apply and reasons:
        rej = folder / "_rejected"
        rej.mkdir(exist_ok=True)
        for name in reasons:
            (folder / name).rename(rej / name)
        print(f"moved {len(reasons)} frame(s) to {rej}  (drag back to undo)")
    elif reasons:
        print("(report only — pass --apply to move these to _rejected/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
