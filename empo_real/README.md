# EMPO on the real MBAG environment — orientation

This folder implements **power maximization (Heitzig & Potham, arXiv:2508.00159)
as an alternative assistant objective inside the AssistanceZero / MBAG framework**,
using the *actual* repo code (real `MbagEnv`, the fork's bedrock asymmetry, the
real trained checkpoint) — not a toy. It is built for a pitch to the AssistanceZero
authors: a **controlled objective-swap** where everything is held fixed (their
environment, model-based planning, the human prior `πh`, the goal set) and only the
assistant's *objective* changes.

This README is the map. Read it, then change things.

---

## 1. Quickstart

The package needs **Python 3.8–3.10** (the system Python is 3.12, which will NOT
work). Set up once with `uv`:

```bash
# from the repo root
uv venv --python 3.10 .venv
uv pip install -e '.[rllib]'        # core + ray 2.7.1 + torch; cairosvg for figure preview
```

Run everything with the venv interpreter:

```bash
.venv/bin/python empo_real/vignettes.py        # the headline; writes empo_real/out/pitch.html
.venv/bin/python empo_real/power_on_mbag.py    # the bare power metric on the real env
.venv/bin/python empo_real/load_checkpoint.py  # loads the real trained policies
```

Outputs land in `empo_real/out/`. The slide deck is in `empo_real/presentation/`
(build with `presentation/build.sh`).

---

## 2. File map

| File | What it is | Where to look |
|---|---|---|
| `vignettes.py` ⭐ | The headline experiment + isometric SVG renderer + `pitch.html` writer | start here |
| `power_on_mbag.py` | The minimal `Veh→Xh→Ur` metric on the real env (the kernel `vignettes.py` generalizes) | simplest entry point |
| `load_checkpoint.py` | Loads the real trained `human` + `assistant` policies from `data/assistancezero_assistant/checkpoint_002000` | the "πh = your model" bridge |
| `out/` | Generated `pitch.html`, `pitch.png`, `metrics.svg` | gitignore-able artifacts |
| `presentation/` | Beamer deck (`empo_mbag.tex`/`.pdf`), figure generator, `build.sh` | the talk |

---

## 3. How `vignettes.py` works (the mental model)

Everything is exact and tiny — no neural nets, no training. The loop per vignette:

1. **A goal set `Gh`** = a handful of target block structures (`MinecraftBlocks`
   grids), built with `with_blocks(...)`.
2. **A human prior `πh`** = a goal-directed builder (`human_action`): each step it
   reduces the world→goal mismatch by one cell (place a needed block / break a
   stray one). It **cannot break bedrock** — the real env enforces that for humans,
   and that asymmetry is the whole point of V1.
3. **`Veh(state, g)`** (`run_human`) = the fraction of the in-scope mismatch the
   human closes when building goal `g` from a given world state. This is the
   paper's effective goal-attainment.
4. **Power** = `human_power_X` (`Xh = Σ_g Veh^ζ`) and `U_r` (`Ur = -(Xh^{-ξ})^η`,
   single human).
5. **Two assistants**, each a model-based planner that picks the candidate "help"
   plan maximizing its objective, scored by rolling `πh` forward in the real env:
   - `goal_inference_assistant` → maximizes `Veh` for one *guessed* goal.
   - `power_assistant` → maximizes `Ur` over *all* goals (goal-agnostic).
6. **`run_vignette`** runs three arms (none / goal-inference / power), prints the
   table, and records state frames; `render_state` + `write_pitch_html` turn them
   into the isometric `pitch.html`.

A "plan" is just `{cell: block_id}` the assistant applies to the world before the
human acts (`apply_plan`). `BEDROCK`→`AIR` clears an obstacle; `AIR`→`STONE` builds.

### One subtlety worth knowing: `Veh` scope
`scope_cells` decides which cells count toward "did the human attain goal `g`":
- `"structure"` — only cells the goal adds vs. the bare floor. Used by **V1** so
  leftover bedrock elsewhere doesn't penalize a goal (otherwise *both* assistants
  would clear everything and the contrast vanishes).
- `"changes_union"` — goal cells **plus any cell the assistant altered**, so the
  human is also charged for *undoing* stray blocks. Used by **V2** to make
  over-committing visibly costly.

This per-vignette scope is the trick that makes each story come out cleanly; if you
add a vignette, pick the scope that matches what you're measuring.

---

## 4. How to change things

**Objective parameters** — edit the constants at the top of `vignettes.py`:
`ZETA, XI, ETA` (`ζ=2, ξ=1, η=1.1`, the paper's gridworld values). Raising the
per-action cost in a vignette's plans reproduces the paper's "optimal menu size."

**Add or modify a vignette** — copy `vignette_open_options` / `vignette_dont_foreclose`.
A vignette builder returns the tuple
`(name, world_size, horizon, start_blocks, bare, goal_set, plans, guess, true_idx, scope)`:
- `world_size`, `horizon` — keep small; horizon must be *tight* if you want
  over-commitment to hurt (V2 uses 3).
- `goal_set` — list of `MinecraftBlocks`; build with `with_blocks(bare, {cell: id})`.
- `plans` — the assistant's candidate actions, `{name: {cell: block_id}}`. Both
  assistants choose among the *same* plans; the objective decides which they pick.
- `guess` / `true_idx` — which goal the goal-inference assistant assumes vs. the
  real hidden goal (pick `true_idx != guess` for the contrast).
- `scope` — `"structure"` or `"changes_union"` (see §3).
Then add the builder to the loop in `main`.

**Change the human prior `πh`** — edit `human_action`. It currently builds
deterministically in cell order. You could make it bounded-rational (sample the
next cell), prioritize bottom-up, etc. To use a repo agent instead, the
`mbag/agents/heuristic_agents.py` builders (`LayerBuilderAgent`, …) are drop-in
goal-directed humans.

**Change what `Veh` measures** — edit `scope_cells` and/or `run_human` (e.g. add
discounting, a completion threshold, or partial credit).

**Change the visuals** — `render_state` is a self-contained isometric SVG renderer;
tune `TW/TH/VH` (tile sizes) and `SHADES` (per-block colors). `write_pitch_html`
controls the page layout.

**Regenerate the deck figures** — `presentation/make_figures.py` imports this
module and renders states to PNG; `presentation/build.sh --figures` regenerates
figures and recompiles the PDF.

---

## 5. Current results (re-run to refresh)

| | no assistant | goal-inference | power |
|---|---|---|---|
| **V1 open the options** (bedrock only the assistant can clear) | Xh 0, fails | clears only its guess → Xh 1, **human fails true goal** | clears all → Xh 3, **human succeeds** |
| **V2 don't foreclose** (committing to a guess can disempower) | Xh 1.25, ok | builds toward guess → Xh 1.18, **human only 43% of true goal** | builds shared foundation → Xh 1.36, **human succeeds** |

V2 is the sharp one: the goal-inference assistant leaves the human **worse off than
no help at all** — the disempowerment-when-wrong the power objective avoids.

---

## 6. Real-env facts you'll need

- **`MbagEnv` API** (see `mbag/environment/mbag_env.py`): `env.get_state()` /
  `env.set_state(state)` snapshot/inject an exact world (we use this to place
  obstacles and goals); `env.step([action_tuple])` where an action is
  `(action_type, ravel(cell), block_id)`; `info["goal_percentage"]` = completion.
- **Bedrock asymmetry** (the fork's modification, `mbag/environment/blocks.py` +
  `action_distributions`): humans cannot break bedrock; the robot assistant can.
  This is what makes "open the options" meaningful.
- **Abilities** default to teleportation + flying + infinite blocks, so the human
  is only ever limited by what it *can't* do (break bedrock) — not by reach or
  inventory.
- **Block ids**: `AIR=0`, `BEDROCK=1`, `dirt=2`, `stone=7` (via
  `MinecraftBlocks.NAME2ID`). World layers: `y=0` bedrock floor, `y=1` dirt,
  `y≥2` build space.

---

## 7. The real checkpoint bridge

`load_checkpoint.py` loads the published trained policies (`human` and `assistant`,
~5.2M params each, 11×10×10 craftassist) from
`data/assistancezero_assistant/checkpoint_002000`. It proves the `πh` slot in the
power planner can be filled by *their* learned human model — the planner doesn't
care whether `πh` is the heuristic builder used here or the trained network.

---

## 8. Scope and the next phase

This **is** a faithful, no-training integration of the power objective into the
authors' framework, showing the behavioral difference with their components. It is
**not** a trained head-to-head at full 11×10×10 scale. That next phase: amortize
the planner's `Veh` (over a sampled/clustered goal set) into a learned value head,
and train a power-maximizing assistant via the repo's AlphaZero loop
(`mbag/rllib/alpha_zero/`), replacing the goal-completion reward with `ΔUr` at the
single reward touch-point in `mbag/rllib/alpha_zero/planning.py`, then compare to
the AssistanceZero checkpoint at scale.
