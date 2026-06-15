"""Post-process WD14 caption .txt files for a character LoRA.

After running sd-scripts' tagger over the curated image folder, this:
  1. prepends a unique TRIGGER token (kept first; protected by keep_tokens=1), and
  2. PRUNES the identity tags you want the LoRA to *bake into* the trigger (hair, eyes, face)
     so they stop being optional prompt words and become part of the character itself.
Variable tags (pose, expression, framing, background, action) are kept so they stay promptable.

Usage:
  python prep_captions.py "C:/.../output/dataset/charA" --trigger ariacharA \
      --prune "brown hair,long hair,green eyes,freckles"
"""
from __future__ import annotations
import argparse
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder of images + WD14 .txt captions")
    ap.add_argument("--trigger", required=True, help="unique trigger token, e.g. ariacharA")
    ap.add_argument("--prune", default="",
                    help="comma-separated tags to remove (exact, case-insensitive match — so "
                         "pruning 'hair' won't also nuke 'hair ornament'). These bake into the trigger.")
    args = ap.parse_args()

    prune = {t.strip().lower() for t in args.prune.split(",") if t.strip()}
    trig = args.trigger.lower()
    folder = pathlib.Path(args.folder)
    txts = sorted(folder.glob("*.txt"))
    if not txts:
        raise SystemExit(f"no .txt captions in {folder} — run the WD14 tagger first")

    changed = 0
    for f in txts:
        tags = [t.strip() for t in f.read_text(encoding="utf-8").split(",") if t.strip()]
        tags = [t for t in tags if t.lower() not in prune and t.lower() != trig]
        tags = [args.trigger] + tags                    # trigger first (protected by keep_tokens=1)
        f.write_text(", ".join(tags), encoding="utf-8")
        changed += 1
    print(f"prepped {changed} captions in {folder} (trigger={args.trigger!r}, pruned {len(prune)} tags)")


if __name__ == "__main__":
    main()
