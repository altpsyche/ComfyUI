# The clothing model — `outfit` vs `keep` vs `outfits` vs `like`

There are **four** ways to control a character's clothes. They are not competing — each answers a
different question. This page is the one place that reconciles them so you pick the right one.

## The one decision

> **How much runtime control over the costume do you want, and in how many LoRA files?**

| Mechanism | What it does | LoRAs | Runtime control | Use when |
|---|---|---|---|---|
| **`outfit`** | One signature costume, **baked** into the trigger (always-on). The garment tags are deleted from the captions and fold into `<char>char`. | **1** | none — the costume is the character | A character has a single canonical look. The default. |
| **`keep`** | A garment from `outfit` (or `outfits`) left **promptable** instead of baked — prompt the tag to wear it, omit/negative to drop it. | (modifier) | add/remove that one item | You want the coat removable but the rest of the outfit locked. Don't over-prune. |
| **`outfits`** *(modular)* | **Several** costumes in **one** LoRA: identity is always-on (`<char>char`), each outfit is a **swappable token** `<char>_<outfit>` baked from its own frames. | **1** | switch among the **trained** outfits by prompt | One character, a few distinct outfits, swapped at inference without juggling files. |
| **`like`** | A **separate** locked LoRA for the same identity in a different costume (inherits id + hero_seed + prune). | **N** (one per outfit) | none per file; you pick the file | Absolute per-outfit fidelity (comics), or outfits too different/numerous to disentangle in one LoRA. |

## Mental shortcuts

- **`outfit` + `keep`** = "one look, but this item is optional." Single LoRA, single costume.
- **`outfits`** = "a wardrobe in one file." Single LoRA, `mirachar, mira_winter` → swap `mira_winter`
  for `mira_hoodie` to change costume. Identity (`mirachar`) is always on.
- **`like`** = "the same person, a different file per costume." Maximum lock, zero runtime mixing.

## Honest limits of `outfits` (modular)

This is the standard community method for multi-outfit anime LoRAs and it works — but know the bounds:

- ✅ **Switching among trained outfits** is reliable; identity stays stable across them — each outfit's
  hero is rendered at the same `hero_seed` + identity, so faces are "recognizably the same person," and
  the always-on identity token averages them into one face at train time.
- ⚠️ **Clean separation is genuinely hard.** With only 2–3 outfits an outfit can leak into identity or
  another outfit. Mitigations baked in: a distinct token per outfit, identity pruned on every frame,
  `keep_tokens=2`, and balanced `num_repeats` (each outfit gets an equal step-share). If a trained LoRA
  still leaks, add more frames per outfit and/or regularization images, then retrain.
- ❌ **You cannot freely stack/mix garments** across outfits (e.g. one outfit's coat over another's
  dress). Separate tokens exist to keep the costumes *separate*; each token recalls its **whole** trained
  outfit. Garment-level layering is an inference-time job (regional prompt / inpaint), not training. For
  absolute lock, use `like`.

## How each is configured (in `characters.toml`)

```toml
# 1) outfit (+ keep): one baked costume, hoodie left promptable
[aria]
id     = "1girl, solo, (long wavy auburn hair:1.1), (green eyes:1.1), freckles"
outfit = "tennis uniform, teal and white tennis dress, white visor, white shoes"
keep   = "visor"                       # promptable; the rest of the outfit bakes

# 2) outfits (modular): one LoRA, swappable tokens mira_hoodie / mira_winter
[mira]
id     = "1girl, solo, (short black bob:1.1), (red eyes:1.1), pale skin, mole under eye"
prune  = "black hair, bob cut, red eyes, mole under eye, pale skin"   # identity baked, always-on
[mira.outfits]
hoodie = "hoodie, pleated skirt, black thighhighs, sneakers"
winter = "long coat, turtleneck sweater, scarf, knee boots"
[mira.keep]                            # optional, per-outfit promptable garments
winter = "coat, long coat"

# 3) like: a separate locked LoRA per costume, same identity
[aria_gala]
like   = "aria"
outfit = "elegant emerald evening gown, long gloves, silver necklace, high heels"
```

`outfits` is **mutually exclusive** with `outfit`/`like` (a character is single-outfit, modular, or a
like-variant — not a mix). Outfit names must match `^[a-z0-9]+$`. Inference for the modular example:
`mirachar, mira_hoodie` (identity + hoodie) or `mirachar, mira_winter` (identity + coat).

See [ADD_CHARACTER.md](ADD_CHARACTER.md) for the step-by-step, and the field docs at the top of
[`../il_graphs/characters.toml`](../il_graphs/characters.toml).
