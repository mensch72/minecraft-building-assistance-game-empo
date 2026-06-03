# Beamer deck: power maximization as an alternative assistant objective in MBAG

A short, professional talk aimed at the **AssistanceZero authors**. It shows that
the power-maximization objective (Heitzig & Potham, arXiv:2508.00159) integrates
faithfully into *their* MBAG framework as an alternative assistant objective, and
contrasts the two objectives behaviorally on two vignettes.

All numbers and figures are generated from the real experiment code in
[`../vignettes.py`](../vignettes.py) — nothing is faked or hand-drawn.

## Contents

- `empo_mbag.tex` — the Beamer source (metropolis theme, with a graceful fallback
  to a standard theme if metropolis is unavailable). 16:9, 12 slides.
- `empo_mbag.pdf` — the compiled deck.
- `make_figures.py` — imports [`../vignettes.py`](../vignettes.py), runs both
  vignettes, and renders the key isometric voxel states to high-res PNGs
  (`scale=3`) via `cairosvg`.
- `figures/` — the generated PNGs (possible goals + per-arm "after assistant" /
  "human builds true goal" frames for each vignette).
- `build.sh` — regenerate figures (optional) and compile to PDF.

## Build

From the repo root (the figure script needs the project `.venv`):

```bash
# regenerate figures from the real code, then compile twice:
empo_real/presentation/build.sh --figures

# or just recompile (figures already present):
empo_real/presentation/build.sh
```

Or manually:

```bash
.venv/bin/python empo_real/presentation/make_figures.py   # -> figures/*.png
cd empo_real/presentation
pdflatex empo_mbag.tex && pdflatex empo_mbag.tex          # run twice for refs
```

**Requirements**: a LaTeX toolchain with Beamer (`pdflatex`; the `metropolis`
theme is used if installed, otherwise a clean fallback theme), and the project
`.venv` with `cairosvg` for the figures.

> Note: `metropolis` prints a warning suggesting XeLaTeX/LuaLaTeX for its Fira
> fonts; `pdflatex` compiles fine with substitute fonts.

## Slide list

1. Title
2. Where this sits in your framework (assistance games → goal inference → power)
3. Two objectives, one environment (the goal-inference vs. power objective)
4. The power objective: `Veh → Xh → Ur` (the equations, `zeta/xi/eta`)
5. How it plugs into MBAG (their env, `pi_h` = their human model, their planner,
   the fork's `goal_agnostic` / bedrock / clutter)
6. Experimental design (controlled objective-swap, no training)
7. V1 "open the options" (figures + numbers + takeaway)
8. V2 "don't foreclose" (figures + numbers + disempowerment takeaway)
9. Results summary table
10. Why it matters (empowerment / robustness / avoid irreversible disempowerment)
11. Honest scope & next steps (trained head-to-head, the two-phase algorithm)
12. Conclusion
