"""Post-process WD14 caption .txt files for a character LoRA.

After running sd-scripts' tagger over the curated image folder, this:
  1. prepends a unique TRIGGER token (kept first; protected by keep_tokens=1), and
  2. PRUNES the tags you want the LoRA to *bake into* the trigger so they stop being optional
     prompt words and become part of the character itself.

Pruning is AUTOMATIC for the outfit: pass --outfit (the same signature-outfit string from the
roster) and every caption tag whose head-noun matches an outfit garment is removed -- including
colour/style variants the tagger invents (e.g. outfit "teal and white tennis dress" locks the
tagger's "white dress", "tennis dress", "dress"). That folds the whole outfit into the trigger so
it renders identically in every scene, with no manual tag-hunting. Variable tags (pose, expression,
framing, background, action) are kept so they stay promptable.

You can still pass --prune for extra exact tags to bake (e.g. identity tags for a harder face lock),
and --keep to protect tags from being pruned.

Usage:
  python prep_captions.py "C:/.../output/dataset/aria" --trigger ariachar \
      --outfit "tennis uniform, teal and white tennis dress, white visor, white wristbands, white shoes"
"""
from __future__ import annotations
import argparse
import pathlib
import re
import sys


# garment head-nouns get baked; these expression/state/view tags are protected even if a head-noun
# would otherwise match -- they must stay promptable for comic panels.
PROTECT = {
    "closed eyes", "one eye closed", "half-closed eyes", "looking at viewer", "looking away",
    "looking back", "looking to the side", "looking up", "looking down",
}
# structural tags never derived into the lock set from the outfit string.
STRUCTURAL = {"1girl", "1boy", "1other", "2girls", "solo", "solo focus"}
# interchangeable garment head-nouns the tagger swaps in: if the outfit hits one, lock the whole
# cluster (so outfit "white shoes" also locks the tagger's "white footwear"/"sneakers"). normalized form.
SYNONYM_CLUSTERS = [
    {"shoe", "boot", "sneaker", "footwear", "heel", "sandal", "loafer"},
    {"short", "shorts"},
    {"top", "shirt", "tanktop", "camisole"},
    {"dress", "gown"},
    {"necklace", "jewelry", "choker", "pendant"},
]


def _norm(word: str) -> str:
    """lower + light plural fold so 'shoes'~'shoe', 'wristbands'~'wristband' match consistently."""
    w = word.lower().strip()
    return w[:-1] if (w.endswith("s") and len(w) > 3) else w


def _phrases(s: str):
    """split a roster id/outfit string into clean tag phrases (strip weights/parens)."""
    out = []
    for p in s.split(","):
        p = re.sub(r":\d+(\.\d+)?", "", p)          # drop ':1.1' weights
        p = p.replace("(", "").replace(")", "").strip().lower()
        if p and p not in STRUCTURAL:
            out.append(p)
    return out


def build_lock(outfit: str, extra_prune: str):
    """Return (exact_phrases, head_nouns) to remove, derived from the outfit + explicit extras."""
    phrases, nouns = set(), set()
    for p in _phrases(outfit):
        phrases.add(p)                              # exact phrase, e.g. 'teal and white tennis dress'
        nouns.add(_norm(p.split()[-1]))             # head noun, e.g. 'dress' -> matches any '... dress'
    for t in extra_prune.split(","):
        t = t.strip().lower()
        if t:
            phrases.add(t)
    for cluster in SYNONYM_CLUSTERS:               # if the outfit hits a cluster, lock its synonyms too
        ncluster = {_norm(c) for c in cluster}     # normalize so the plural-fold matches (dress~dres)
        if nouns & ncluster:
            nouns |= ncluster
    return phrases, nouns


def should_prune(tag: str, phrases: set, nouns: set, keep: set) -> bool:
    # WD14 writes underscored tags (crop_top, open_shirt, looking_at_viewer); the outfit / keep / PROTECT
    # sets are space-form. Normalize underscores -> spaces so the phrase + head-noun match actually fires
    # (otherwise NOTHING is baked and the outfit stays promptable instead of folding into the trigger).
    t = tag.lower().replace("_", " ").strip()
    if not t or t in keep or t in PROTECT:
        return False
    if t in phrases:
        return True
    return _norm(t.split()[-1]) in nouns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder of images + WD14 .txt captions")
    ap.add_argument("--trigger", required=True, help="unique trigger token, e.g. ariachar")
    ap.add_argument("--outfit", default="",
                    help="roster outfit string; its garment head-nouns are auto-pruned (colour/style "
                         "variants included) so the outfit bakes into the trigger.")
    ap.add_argument("--prune", default="",
                    help="extra comma-separated tags to bake (exact or head-noun match), e.g. identity "
                         "tags for a harder face lock.")
    ap.add_argument("--keep", default="",
                    help="comma-separated tags to protect from pruning (stay promptable).")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the tags that would be pruned per file, without writing anything.")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero (2) if --outfit/--prune was given but NOTHING matched -- i.e. the "
                         "outfit failed to bake (usually a tagger-vocabulary mismatch). train_lora passes "
                         "this so a zero-bake aborts the run instead of silently training a bad LoRA.")
    args = ap.parse_args()

    phrases, nouns = build_lock(args.outfit, args.prune)
    keep = {t.strip().lower() for t in args.keep.split(",") if t.strip()}
    trig = args.trigger.lower()
    folder = pathlib.Path(args.folder)
    txts = sorted(folder.glob("*.txt"))
    if not txts:
        raise SystemExit(f"no .txt captions in {folder} -- run the WD14 tagger first")

    changed = removed_total = 0
    for f in txts:
        tags = [t.strip() for t in f.read_text(encoding="utf-8").split(",") if t.strip()]
        pruned = [t for t in tags if t.lower() != trig and should_prune(t, phrases, nouns, keep)]
        kept = [t for t in tags if t.lower() != trig and not should_prune(t, phrases, nouns, keep)]
        removed_total += len(pruned)
        if args.dry_run:
            print(f"  {f.name}: prune {pruned}" if pruned else f"  {f.name}: (nothing to prune)")
        else:
            f.write_text(", ".join([args.trigger] + kept), encoding="utf-8")  # trigger first (keep_tokens=1)
        changed += 1
    verb, rverb = ("would prep", "would remove") if args.dry_run else ("prepped", "removed")
    print(f"{verb} {changed} captions in {folder} (trigger={args.trigger!r}, "
          f"lock nouns={sorted(nouns)}, {rverb} {removed_total} tag instances)")

    # Semantic guard: an outfit/prune was requested but NOTHING matched -> the outfit did NOT bake into
    # the trigger (it stays promptable). This is the silent failure mode (a green run, a bad LoRA), so
    # shout about it -- and under --strict, fail so the trainer aborts.
    if (args.outfit.strip() or args.prune.strip()) and removed_total == 0:
        print(f"[!] WARNING: --outfit/--prune given but NOTHING matched in {folder} -- the outfit did NOT "
              f"bake into trigger {args.trigger!r}; those garments stay PROMPTABLE, not locked. Most likely "
              f"the outfit words don't match the WD14 tagger's tags (e.g. 'stockings' vs 'thighhighs', "
              f"'thong underwear' vs 'panties'). Open a .txt to see the real tags. Lock nouns tried: "
              f"{sorted(nouns)}.", file=sys.stderr)
        if args.strict:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
