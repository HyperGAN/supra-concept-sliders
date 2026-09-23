#!/usr/bin/env python3
"""Product train entry for Supra2-IMG.

Moved here from HyperGAN/particle-sliders ``conceptmod/textsliders/train_lora_supra.py``.
The particle game is ``particle_sliders.winning_formulation()`` in
particle-sliders-core. This script does not vendor that math.

``conceptmod/textsliders/train_lora_supra.py`` is still the research UNI
trainer in particle-sliders. It is not this entry. See docs/shared-core.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supra.train import main


if __name__ == "__main__":
    main()
