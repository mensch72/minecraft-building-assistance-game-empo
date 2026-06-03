#!/usr/bin/env python3
"""Generate the presentation figures from the REAL vignette code.

Imports `empo_real/vignettes.py`, runs both vignettes, and renders the key
isometric voxel states to high-resolution PNGs for the Beamer deck:

  per vignette:
    * the possible goals (a strip, one PNG per goal)
    * per arm (no assistant / goal-inference / power):
        - "after assistant"  (the world the assistant leaves)
        - "human builds true goal" (final frame of pi_h toward the true goal)

Run from the repo root with the venv:
    .venv/bin/python empo_real/presentation/make_figures.py
"""

import os
import sys

import cairosvg

# Make `empo_real` importable regardless of cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
EMPO_REAL = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(EMPO_REAL)
sys.path.insert(0, REPO_ROOT)

from empo_real.vignettes import (  # noqa: E402
    render_state,
    run_vignette,
    vignette_dont_foreclose,
    vignette_open_options,
)

FIG_DIR = os.path.join(HERE, "figures")
SCALE = 3.0  # render at 3x for crisp slides


def svg_to_png(svg, path):
    cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        write_to=path,
        scale=SCALE,
        background_color="white",
    )
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)}")


def arm_key(arm):
    if arm.startswith("power"):
        return "power"
    if arm.startswith("goal"):
        return "goalinf"
    return "none"


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    builders = {
        "v1": vignette_open_options,
        "v2": vignette_dont_foreclose,
    }
    for tag, builder in builders.items():
        print(f"[{tag}] running vignette + rendering...")
        S = run_vignette(builder())
        ws = S["ws"]

        # possible goals
        for i, g in enumerate(S["goal_set"]):
            svg_to_png(render_state(g, ws), os.path.join(FIG_DIR, f"{tag}_goal{i}.png"))

        # per-arm frames
        for arm, r in S["results"].items():
            k = arm_key(arm)
            svg_to_png(
                render_state(r["state1"]["current_blocks"], ws),
                os.path.join(FIG_DIR, f"{tag}_{k}_after.png"),
            )
            svg_to_png(
                render_state(r["frames"][-1], ws),
                os.path.join(FIG_DIR, f"{tag}_{k}_human.png"),
            )
        print()

    print(f"All figures written to {os.path.relpath(FIG_DIR, REPO_ROOT)}/")


if __name__ == "__main__":
    main()
