# Add a new character — full checklist

Follow this top to bottom and you won't miss a step. Terser loop: [QUICKSTART.md](QUICKSTART.md).
How the dataset engine works: [DATASET.md](DATASET.md). Every knob: [REFERENCE.md](REFERENCE.md).
Traps you shouldn't re-discover: [GOTCHAS.md](GOTCHAS.md). Which clothing mechanism to use
(outfit / keep / outfits / like): [CLOTHING_MODEL.md](CLOTHING_MODEL.md).

> **The 3 things people get wrong** (each one silently ruins the LoRA):
> 1. **Outfit must use the tagger's vocabulary** — write clothes the way the WD14 tagger does, e.g.
>    `pants` not `trousers`, `sneakers` not `trainers`. **Why:** baking works by *deleting the matching
>    tags from every caption* so they fuse into the trigger word. The tagger only ever writes danbooru
>    tags, so if you write `trousers` but each image was tagged `pants`, there's nothing for the matcher
>    to delete — the outfit never folds into the trigger; it stays a separate, still-promptable thing.
>    (The trainer now *aborts* on a total miss, but it can't catch a partial-vocab miss.)
> 2. **`id` is identity ONLY** — hair / eyes / face / body. **No clothes** in `id` (clothes go in `outfit`).
> 3. **Curate the dataset** — keep the on-model frames. 30 consistent images beat 200 drifting ones.

---

## 1. Define the character — `tools/il_graphs/characters.toml`

Add one `[table]` (table name = the character name you pass to `./dev train` = the dataset folder name):

```toml
[aria]
id = "1girl, solo, (long wavy auburn hair:1.1), (green eyes:1.1), freckles"   # identity ONLY, no clothes
outfit = "tennis uniform, teal and white tennis dress, white visor, white wristbands, white shoes"
```

- [ ] `id` — identity tags only (hair colour/length/style, eyes, face marks, body). Weight the
      face-defining ones, e.g. `(green eyes:1.1)`. **No outfit here.**
- [ ] `outfit` *(optional)* — the signature clothes, in **danbooru/WD14 tag words** (see §1a). Leave it
      off to let the checkpoint pick clothes (promptable, not locked).
- [ ] `trigger` *(optional)* — defaults to `<name>char` (e.g. `ariachar`). Use a rare word.
- [ ] Costume variant of an existing character? Use `like` (see §8) — don't rewrite `id`.

### 1a. Get the outfit vocabulary right (the #1 failure)

**How baking works (so the rule makes sense):** the WD14 tagger writes a list of danbooru tags onto
every image (`1girl, pants, sneakers, ...`). `prep_captions` then **deletes the tags that match your
`outfit` string** and prepends the trigger — so those garments stop being independent prompt words and
fuse into `<name>char`. Net effect: the trigger *carries* the outfit. But the match is literal against
the tagger's words. Write a word the tagger never emits and there's nothing to delete → that garment
stays a normal, separate tag → it does **not** travel with the trigger.

- Use danbooru tag names. Common mismatches (what you'd type → what the tagger writes):
  `trousers → pants`, `trainers / runners → sneakers`, `tights → pantyhose`, `t-shirt → shirt`,
  `jumper → sweater`, `stockings → thighhighs`.
- Multi-word tags are fine (`crop top`, `denim shorts`, `open shirt`) — the matcher handles the
  underscores the tagger uses.
- Keep it **simple and consistent**. A complex, contradictory outfit (e.g. crop top *and* open shirt
  *and* visible underwear) makes Qwen drift across frames, so even a perfect bake stays variable.
- Not sure of the exact tags? Generate the dataset first (§3), let it tag once, then **open any
  `output/dataset/<name>/*.txt`** and copy the real garment tags into your `outfit` string.

## 2. Regenerate the workflows

```sh
./dev build il_graphs
```
- [ ] Emits `IL_DatasetEdit_<name>` + rewrites `tools/lora_train/roster.json`, and **validates** every
      graph. If it prints a validation failure, fix it before continuing.
- [ ] **Re-open the workflow in ComfyUI** (it caches the old version).

## 3. Generate the dataset — in ComfyUI

Open **`IL_DatasetEdit_<name>`** (anatomy + tuning: [DATASET.md](DATASET.md)):
- [ ] **Stage 1:** reroll **Hero Seed** until the face in **HERO preview** is the one you want. Leave it
      fixed on that value.
- [ ] **Stage 2:** set the Edit-instruction seed to **randomize**, batch count **~40**, **Queue once** →
      fills `output/dataset/<name>/`.

## 4. Curate

- [ ] *(optional)* mechanical first pass — flag blurry / near-duplicate frames:
      `python tools/lora_train/cull_dataset.py <name>` (add `--apply` to move them to `_rejected/`).
- [ ] Delete off-model / bad-anatomy / wrong-outfit frames **in place** (the identity call is yours).
- [ ] Keep the best **25–40** (minimum **12**). Quality + variety beats count.

## 5. (Optional) Tune training

Defaults (dim 16 / alpha 8 / 1500 steps / prodigy) suit a ~30–40 image character. Change only if needed:
- [ ] Big set (~200 imgs): raise steps — `--steps 3000 --epochs 10`, or a `[train.<name>]` block in
      [`train.toml`](train.toml), or `--profile complex`.
- [ ] Visually complex character: `--profile complex` (dim 32 / alpha 16) or `--dim 32 --alpha 16`.
- [ ] Preview the resolved settings + exact command without training: `./dev train <name> --dry-run`.

Full matrix + precedence (CLI > `--profile` > `[train.<char>]` > defaults): [REFERENCE.md](REFERENCE.md).

## 6. Train

```bash
./dev train <name>
```
It auto-runs the WD14 tagger → bakes the outfit (`prep_captions`) → trains.

- [ ] **If it aborts with "outfit matched NO caption tags"** → the outfit vocabulary is wrong. Open a
      `output/dataset/<name>/*.txt`, copy the real garment tags into `outfit` in `characters.toml`,
      `python tools/build_il_graphs.py`, then retrain. (Or `--force` to train without the outfit baked.)
- [ ] Free ComfyUI's VRAM first (POST `/free` or close it) and **disable PC sleep** — a multi-hour run
      dies silently if the machine sleeps.

### 6a. Verify the bake actually landed

After training, **open a `output/dataset/<name>/*.txt`**:
- [ ] The **trigger is first** (`ariachar, ...`).
- [ ] The **outfit garments are GONE** from the tag list (they baked into the trigger). If a garment is
      still there, it didn't bake → fix its vocabulary (§1a) and retrain.

Output: `models/loras/<name>_v1.safetensors` + one checkpoint per epoch (`<name>_v1-000001…`), plus a
`<name>_v1.args.txt` recording the exact run.

## 7. Pick the best epoch — `IL_XYPlot`

The final epoch isn't always best. Compare them:
- [ ] Copy the epochs to compare into **`models/loras/_xyplot/`** (e.g. `<name>_v1.safetensors` + a few
      `-0000NN`). Thin to ~4 later epochs for speed.
- [ ] Open **`IL_XYPlot`**, add the **trigger word** to the Efficient Loader's positive prompt, keep the
      seed fixed, **Queue once** → `output/xyplot/`.
- [ ] Pick the `(epoch, strength)` cell with the best likeness that isn't over-cooked / over-saturated.

## 8. Use the LoRA

- [ ] In any IL_* graph, **LoRA bank** → toggle the chosen file **ON**, strength **~0.75**.
- [ ] Put the **trigger word** (`ariachar`) in the **Positive** prompt — without it the character/outfit
      won't reliably appear.

### Same character, different outfit (costume variant)

Add a `like` table — a separate LoRA, same face, you write only the new outfit:
```toml
[aria_gala]
like = "aria"                                    # inherits aria's id + hero_seed + prune (overridable)
outfit = "elegant emerald evening gown, long gloves, high heels"
```
Then §2 → §8 as normal. Trigger defaults to `aria_galachar`. **Pin `hero_seed`** on the parent (`aria`)
after you find a face you like, so variants reuse it (closest achievable face — not pixel-identical).

### Modular character — one LoRA, swappable outfits

Want several outfits **switchable at inference in one file** (instead of a separate `like` LoRA each)?
Use a `[<char>.outfits]` table. Trade-offs vs `like`: [CLOTHING_MODEL.md](CLOTHING_MODEL.md).

```toml
[mira]
id     = "1girl, solo, (short black bob:1.1), (red eyes:1.1), pale skin, mole under eye"
prune  = "black hair, bob cut, red eyes, mole under eye, pale skin"   # identity baked, always-on
[mira.outfits]                          # each -> a swappable token mira_<name>
hoodie = "hoodie, pleated skirt, black thighhighs, sneakers"
winter = "long coat, turtleneck sweater, scarf, knee boots"
[mira.keep]                             # optional, per-outfit: garments left promptable
winter = "coat, long coat"
```

- [ ] `outfits` is **mutually exclusive** with `outfit`/`like`. Outfit names must be `^[a-z0-9]+$`.
- [ ] §2 regenerate emits **one graph per outfit**: `IL_DatasetEdit_mira_hoodie`, `IL_DatasetEdit_mira_winter`.
- [ ] §3 generate **each** graph into its own subfolder `output/dataset/mira/<outfit>/` (~20–25 frames
      each; SDXL renders the character wearing that outfit, same `hero_seed`+id so faces match). Curate each (§4).
- [ ] §6 train **once**: `./dev train mira` — it captions every outfit subfolder with a two-token
      trigger (`mirachar, mira_<outfit>`), bakes per-outfit, and trains one balanced multi-subset LoRA.
- [ ] §6a per-outfit: each `.txt` starts `mirachar, mira_<outfit>, …` and that outfit's garments are gone.
- [ ] §8 inference: `mirachar, mira_hoodie` or `mirachar, mira_winter` — swap the outfit token to change
      costume. You can switch among **trained** outfits; you can't freely stack/mix garments (see
      [CLOTHING_MODEL.md](CLOTHING_MODEL.md)).

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `./dev train` aborts: "outfit matched NO caption tags" | Outfit vocabulary mismatch (§1a). Open a `.txt`, use the real tags, rebuild, retrain. |
| Trained, but the **outfit is different / inconsistent** | Outfit didn't bake (garments still in the `.txt` — §6a), or the dataset drifted (§4). Fix vocab + curate, retrain. |
| "no dataset at output/dataset/\<name\>" | You skipped §3 — generate the dataset first. |
| Character doesn't appear at inference | Trigger word missing from the prompt, or LoRA strength too low (§7). |
| Outfit looks right at high strength only | Use the XY plot (§7) to find the strength/epoch sweet spot. |
| OOM / too slow generating the dataset | `./dev models install il_graphs --variant quant=Q4_K_M`. More: [DATASET.md](DATASET.md). |
| Identity drifts across dataset frames | Lower the multiple-angles LoRA; pin a tight `id` + `hero_seed`. More: [DATASET.md](DATASET.md). |
| Modular: outfits bleed into each other / into identity | Hard to fully avoid with few outfits. Add more frames per outfit, make outfits more visually distinct, or add regularization images; retrain. [CLOTHING_MODEL.md](CLOTHING_MODEL.md). |
