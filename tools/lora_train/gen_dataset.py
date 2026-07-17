"""Headless dataset generator: queue an IL_DatasetEdit graph N times with fresh seeds.

The `IL_DatasetEdit_<name>` graph makes ONE varied frame per queue (the wildcard instruction + edit seed
re-roll each run). In the UI you set batch-count ~40 and click once. This does the same unattended: it
POSTs the workflow to the ComfyUI API N times, bumping every seed each time so each frame differs, and
the images stream into output/dataset/<name>/ as usual.

One-time per character: open `IL_DatasetEdit_<name>` in ComfyUI and **File -> Export (API)**, saving it
next to the workflow as `IL_DatasetEdit_<name>.api.json` (the API/prompt format the /prompt endpoint
needs — the saved .json is the UI format, which this endpoint can't take). Then:

  python gen_dataset.py aria -n 40              # queue aria's graph 40x (random seeds) -> output/dataset/aria/
  python gen_dataset.py aria -n 40 --seed 1000  # reproducible: seeds 1000,1001,... per frame
  python gen_dataset.py --all -n 40             # every roster character that has an exported .api.json
  python gen_dataset.py path/to/wf.api.json -n 20

ComfyUI must be running (default http://127.0.0.1:8188). Frames queue and process sequentially.

`randomize_seeds` is pure (no network) so it's unit-tested.
"""
from __future__ import annotations
import argparse
import copy
import json
import pathlib
import random
import sys
from urllib import request, error

SEED_KEYS = ("seed", "noise_seed")            # widget names ComfyUI uses for RNG seeds
SEED_MAX = 2 ** 32 - 1
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKFLOWS = REPO / "user" / "default" / "workflows"
ROSTER = HERE / "roster.json"


def randomize_seeds(prompt: dict, rng: random.Random) -> int:
    """Set every node input named seed/noise_seed to a fresh value. Returns how many were changed."""
    n = 0
    for node in prompt.values():
        inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
        for k in SEED_KEYS:
            if k in inputs and not isinstance(inputs[k], list):   # skip linked inputs ([node, slot])
                inputs[k] = rng.randint(0, SEED_MAX)
                n += 1
    return n


def api_path_for(target: str) -> pathlib.Path:
    """Resolve a char name or a path to its API-format workflow json."""
    p = pathlib.Path(target)
    if p.suffix == ".json" and p.exists():
        return p
    cand = WORKFLOWS / f"IL_DatasetEdit_{target}.api.json"
    return cand


def _queue(prompt: dict, url: str):
    data = json.dumps({"prompt": prompt}).encode("utf-8")
    req = request.Request(f"{url}/prompt", data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_one(target: str, n: int, url: str, base_seed: int | None) -> bool:
    apifile = api_path_for(target)
    if not apifile.exists():
        print(f"[!] {target}: no API workflow at {apifile.name} -- in ComfyUI open IL_DatasetEdit_{target} "
              f"and 'File -> Export (API)', saving it as {apifile.name} in user/default/workflows/.",
              file=sys.stderr)
        return False
    base = json.loads(apifile.read_text(encoding="utf-8"))
    rng = random.Random(base_seed)                # base_seed=None -> nondeterministic
    queued = 0
    for i in range(n):
        prompt = copy.deepcopy(base)
        changed = randomize_seeds(prompt, rng)
        if changed == 0 and i == 0:
            print(f"[!] {target}: no seed inputs found to randomize -- frames may repeat.", file=sys.stderr)
        try:
            _queue(prompt, url)
            queued += 1
        except error.URLError as e:
            print(f"[x] {target}: cannot reach ComfyUI at {url} ({e.reason}). Is it running?", file=sys.stderr)
            return False
    print(f"[+] {target}: queued {queued}/{n} frames -> output/dataset/{target}/ (processing in ComfyUI)")
    return True


def _roster_names() -> list[str]:
    if not ROSTER.exists():
        raise SystemExit(f"roster not found ({ROSTER}); run python tools/build_il_graphs.py")
    return [e["name"] for e in json.loads(ROSTER.read_text(encoding="utf-8"))]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Queue an IL_DatasetEdit graph N times with fresh seeds.")
    ap.add_argument("target", nargs="?", help="character name, or a path to an exported .api.json")
    ap.add_argument("-n", "--count", type=int, default=40, help="frames to queue (default 40)")
    ap.add_argument("--all", action="store_true", help="run every roster character with an exported .api.json")
    ap.add_argument("--url", default="http://127.0.0.1:8188", help="ComfyUI base URL")
    ap.add_argument("--seed", type=int, default=None, help="base seed for reproducible runs (default random)")
    args = ap.parse_args(argv)

    if args.all:
        targets = _roster_names()
    elif args.target:
        targets = [args.target]
    else:
        ap.error("give a character name / .api.json path, or --all")

    ok = sum(run_one(t, args.count, args.url.rstrip("/"), args.seed) for t in targets)
    return 0 if ok == len(targets) else 1


if __name__ == "__main__":
    sys.exit(main())
