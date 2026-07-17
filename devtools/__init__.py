"""ComfyUI dev toolkit — cross-platform provisioning, workflow generation, and LoRA training.

Invoked as `python -m devtools <command>` (see the ./dev launcher). The dispatcher and the
`setup` path import ONLY the standard library so they run on a bare system python before any
venv exists; everything that needs torch/deps re-execs into the appropriate venv.
"""
