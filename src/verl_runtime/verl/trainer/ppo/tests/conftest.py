# Copyright 2026 CPR contributors
# Licensed under the Apache License, Version 2.0
"""Pytest config for CPR test suite.

Adds ``/workspace/src`` to ``sys.path`` so standalone helpers such as
``cpr_alpha_fit`` (sibling to ``run_sirl_verl.py`` and the ``sirl/`` package)
are importable under the same ``pytest`` invocation that already picks up
``verl`` from ``/workspace/src/verl_runtime``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[5]  # .../src/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
