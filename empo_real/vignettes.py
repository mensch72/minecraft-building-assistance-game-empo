#!/usr/bin/env python3
"""
Power-maximization as an alternative assistant objective, INSIDE the MBAG /
AssistanceZero framework.

Pitch framing (audience = the AssistanceZero authors): we keep *your* environment
(`MbagEnv`), *your* style of model-based planning (the AlphaZero idea: plan over
the known simulator), and a goal-directed human prior pi_h. We change ONE thing:
the assistant's objective.

  - GOAL-INFERENCE assistant (the AssistanceZero objective): infer the human's goal
    and maximize completion of THAT goal.
  - POWER assistant (Heitzig & Potham, arXiv:2508.00159): maximize aggregate human
    power Ur = -(sum_g Veh^zeta)^(-xi*eta) over the whole space of possible goals,
    without inferring which one is intended (goal-agnostic).

Same env, same planner, same human, same goal set -> the behavioral difference is
purely the objective. Two vignettes:

  V1 "open the options"  : bedrock obstacles only the assistant can clear.
  V2 "don't foreclose"   : helping a guessed goal can foreclose the others.

No training: each assistant plans by evaluating candidate helpful actions against
its objective, scored by rolling the human prior forward in the real env.

Run:  .venv/bin/python empo_real/vignettes.py
"""

import copy
import os

import numpy as np

from mbag.environment.actions import MbagAction
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.config import DEFAULT_CONFIG
from mbag.environment.mbag_env import MbagEnv

STONE = MinecraftBlocks.NAME2ID["stone"]
AIR, BEDROCK = MinecraftBlocks.AIR, MinecraftBlocks.BEDROCK

# Power-objective parameters (paper's gridworld, Table 5).
ZETA, XI, ETA = 2.0, 1.0, 1.1


# --------------------------------------------------------------------------- #
# Env + state helpers
# --------------------------------------------------------------------------- #
def make_env(world_size, horizon):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["world_size"] = world_size
    cfg["num_players"] = 1
    cfg["horizon"] = horizon
    cfg["num_clutter_blocks"] = 0
    cfg["terminate_on_goal_completion"] = False
    return MbagEnv(cfg)


def ravel(cell, world_size):
    return int(np.ravel_multi_index(cell, world_size))


def with_blocks(base, changes):
    b = base.copy()
    for cell, bid in changes.items():
        b.blocks[cell] = bid
        b.block_states[cell] = 0
    return b


def nonfloor_cells(world_size):
    W, H, D = world_size
    return [(x, y, z) for x in range(W) for y in range(2, H) for z in range(D)]


# --------------------------------------------------------------------------- #
# What cells count toward "did the human attain goal g?" (the goal event).
#   scope="structure"     -> cells the goal adds vs the bare world (V1).
#   scope="changes_union" -> goal's cells PLUS any cell the assistant altered, so
#                            the human is also charged for undoing stray blocks (V2).
# --------------------------------------------------------------------------- #
def scope_cells(scope, goal, state1_blocks, bare, world_size):
    out = []
    for c in nonfloor_cells(world_size):
        in_goal = int(goal.blocks[c]) != int(bare.blocks[c])
        in_state = int(state1_blocks.blocks[c]) != int(bare.blocks[c])
        if in_goal or (scope == "changes_union" and in_state):
            out.append(c)
    return out


def diff_over(cur, goal, cells):
    return sum(int(cur.blocks[c]) != int(goal.blocks[c]) for c in cells)


# --------------------------------------------------------------------------- #
# Human prior pi_h: goal-directed builder. Reduces the world->goal mismatch one
# cell per step -- placing a needed block or breaking a stray/wrong one. It
# CANNOT break bedrock (the env enforces this for humans).
# --------------------------------------------------------------------------- #
def human_action(cur, goal, world_size):
    for c in nonfloor_cells(world_size):
        cb, gb = int(cur.blocks[c]), int(goal.blocks[c])
        if cb == gb or cb == BEDROCK:
            continue
        if cb == AIR and gb != AIR:
            return (MbagAction.PLACE_BLOCK, ravel(c, world_size), gb)
        if cb != AIR:
            return (MbagAction.BREAK_BLOCK, ravel(c, world_size), 0)
    return (MbagAction.NOOP, 0, 0)


def run_human(env, state1, goal, horizon, scope, bare, record=False):
    """Run pi_h toward `goal` from post-assistant state `state1`.
    Returns (Veh, frames). Veh = fraction of the in-scope mismatch the human closes."""
    ws = env.config["world_size"]
    cells = scope_cells(scope, goal, state1["current_blocks"], bare, ws)
    st = copy.deepcopy(state1)
    st["goal_blocks"] = goal.copy()
    env.set_state(st)
    init_diff = diff_over(env.current_blocks, goal, cells)
    frames = [env.current_blocks.copy()] if record else None
    for _ in range(horizon):
        if diff_over(env.current_blocks, goal, cells) == 0:
            break
        env.step([human_action(env.current_blocks, goal, ws)])
        if record:
            frames.append(env.current_blocks.copy())
    final_diff = diff_over(env.current_blocks, goal, cells)
    veh = 1.0 if init_diff == 0 else (init_diff - final_diff) / init_diff
    return max(0.0, min(1.0, veh)), frames


# --------------------------------------------------------------------------- #
# Power metric
# --------------------------------------------------------------------------- #
def human_power_X(vehs):
    return sum(v ** ZETA for v in vehs)


def U_r(vehs):
    x = max(human_power_X(vehs), 1e-9)
    return -((x ** (-XI)) ** ETA)


def capability(env, state1, goal_set, horizon, scope, bare):
    return [run_human(env, state1, g, horizon, scope, bare)[0] for g in goal_set]


# --------------------------------------------------------------------------- #
# Assistants = model-based planners choosing a candidate "help" plan ({cell:
# block_id} applied to the world) to maximize their objective.
# --------------------------------------------------------------------------- #
def apply_plan(base_state, plan):
    st = copy.deepcopy(base_state)
    st["current_blocks"] = with_blocks(base_state["current_blocks"], plan)
    return st


def power_assistant(env, base_state, goal_set, plans, horizon, scope, bare):
    best, best_obj = None, -np.inf
    for name, plan in plans.items():
        st = apply_plan(base_state, plan)
        obj = U_r(capability(env, st, goal_set, horizon, scope, bare))   # goal-AGNOSTIC
        if obj > best_obj:
            best, best_obj = name, obj
    return best


def goal_inference_assistant(env, base_state, goal_set, plans, horizon, scope, bare, guess):
    best, best_obj = None, -np.inf
    for name, plan in plans.items():
        st = apply_plan(base_state, plan)
        obj = run_human(env, st, goal_set[guess], horizon, scope, bare)[0]   # only the GUESS
        if obj > best_obj:
            best, best_obj = name, obj
    return best


# --------------------------------------------------------------------------- #
# Vignette runner
# --------------------------------------------------------------------------- #
def run_vignette(spec):
    (name, ws, horizon, start_blocks, bare, goal_set, plans, guess, true_idx,
     scope) = spec
    env = make_env(ws, horizon)
    env.reset()
    base_state = env.get_state()
    base_state["current_blocks"] = start_blocks.copy()

    arms = {
        "no assistant": "noop",
        f"goal-inference (guess G{guess})":
            goal_inference_assistant(env, base_state, goal_set, plans, horizon, scope, bare, guess),
        "power (goal-agnostic)":
            power_assistant(env, base_state, goal_set, plans, horizon, scope, bare),
    }

    print("=" * 76)
    print(f"VIGNETTE: {name}")
    print("=" * 76)
    print(f"{'assistant objective':<30}{'chose plan':<18}{'Xh':<7}{'attainable':<11}"
          f"{'true-goal Veh':<13}")

    results = {}
    K = len(goal_set)
    for arm, plan_name in arms.items():
        st1 = apply_plan(base_state, plans[plan_name])
        vehs = capability(env, st1, goal_set, horizon, scope, bare)
        xh = human_power_X(vehs)
        attainable = sum(1 for v in vehs if v > 0.75)
        true_veh, frames = run_human(env, st1, goal_set[true_idx], horizon, scope, bare, record=True)
        results[arm] = dict(plan=plan_name, vehs=vehs, xh=xh, attainable=attainable,
                            true_veh=true_veh, state1=st1, frames=frames)
        print(f"{arm:<30}{plan_name:<18}{xh:<7.2f}{attainable:<11}{true_veh:<13.2f}")

    print(f"\nTrue (hidden) goal = G{true_idx}.  zeta={ZETA} xi={XI} eta={ETA}, K={K}")
    for arm in arms:
        v = results[arm]["true_veh"]
        verdict = "completes it" if v > 0.99 else ("partial" if v > 0.5 else "fails")
        print(f"  {arm:<30} -> human {verdict} (Veh={v:.2f})")
    return dict(name=name, ws=ws, results=results, goal_set=goal_set,
                true_idx=true_idx, base_state=base_state)


# --------------------------------------------------------------------------- #
# Concrete vignettes
# --------------------------------------------------------------------------- #
def vignette_open_options():
    ws, horizon = (5, 4, 5), 30
    env = make_env(ws, horizon); env.reset()
    bare = env.current_blocks.copy()                      # bare floor
    cells = [(1, 2, 1), (2, 2, 2), (3, 2, 3)]
    goal_set = [with_blocks(bare, {c: STONE}) for c in cells]
    start = with_blocks(bare, {c: BEDROCK for c in cells})  # every goal blocked
    plans = {
        "noop": {},
        "clear G1": {cells[1]: AIR},                      # serve only the guess
        "clear all": {c: AIR for c in cells},             # open every option
    }
    return ("open the options (bedrock only the assistant can clear)",
            ws, horizon, start, bare, goal_set, plans, 1, 2, "structure")


def vignette_dont_foreclose():
    ws, horizon = (7, 4, 5), 3
    env = make_env(ws, horizon); env.reset()
    bare = env.current_blocks.copy()
    F = (3, 2, 2)                                         # shared foundation
    A_cells = [(1, 2, 2), (1, 3, 2), (1, 2, 3), (1, 3, 3), (1, 2, 1)]  # goal A (5)
    B_cells = [(5, 2, 2), (5, 3, 2)]                                   # goal B (2)
    goalA = with_blocks(bare, {F: STONE, **{c: STONE for c in A_cells}})
    goalB = with_blocks(bare, {F: STONE, **{c: STONE for c in B_cells}})
    goal_set = [goalA, goalB]
    plans = {
        "noop": {},
        "build foundation": {F: STONE},                        # helps BOTH
        "build toward A": {F: STONE, **{c: STONE for c in A_cells}},  # commit to A
    }
    return ("don't foreclose (committing to the guess can disempower)",
            ws, horizon, bare, bare, goal_set, plans, 0, 1, "changes_union")


# --------------------------------------------------------------------------- #
# Isometric voxel renderer (pure SVG, no deps). Renders an MBAG block grid.
# --------------------------------------------------------------------------- #
TW, TH, VH = 26, 14, 20          # iso tile width/height, cube vertical height
SHADES = {                       # block_id -> (top, left, right)
    BEDROCK: ("#5a5a5a", "#444444", "#333333"),
    MinecraftBlocks.NAME2ID["dirt"]: ("#9c7a52", "#7d6141", "#654f34"),
    STONE: ("#cfcfcf", "#a9a9a9", "#8f8f8f"),
}
DEFAULT_SHADE = ("#7da6c9", "#5f86a8", "#4d6f8c")


def _proj(x, y, z, ox, oy):
    return (ox + (x - z) * (TW / 2), oy + (x + z) * (TH / 2) - y * VH)


def render_state(blocks, ws, *, draw_floor_y=1):
    """Isometric SVG for a block grid. Draws floor from draw_floor_y up."""
    W, H, D = ws
    width = (W + D) * (TW / 2) + 20
    height = (W + D) * (TH / 2) + H * VH + 20
    ox, oy = D * (TW / 2) + 10, H * VH + 5
    cubes = []
    for x in range(W):
        for y in range(draw_floor_y, H):
            for z in range(D):
                b = int(blocks.blocks[x, y, z])
                if b == AIR:
                    continue
                cubes.append((x + z, y, x, x, y, z, b))
    cubes.sort()                                   # painter: back (small x+z,y) first
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
         f'height="{height:.0f}">']
    for _, _, _, x, y, z, b in cubes:
        top, left, right = SHADES.get(b, DEFAULT_SHADE)
        c = {(dx, dy, dz): _proj(x + dx, y + dy, z + dz, ox, oy)
             for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)}
        def poly(pts, fill):
            d = " ".join(f"{c[p][0]:.1f},{c[p][1]:.1f}" for p in pts)
            return f'<polygon points="{d}" fill="{fill}" stroke="#2b2b2b" stroke-width="0.6"/>'
        s.append(poly([(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)], top))     # top
        s.append(poly([(1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)], right))   # +x
        s.append(poly([(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], left))    # +z
    s.append("</svg>")
    return "".join(s)


def write_pitch_html(summaries, path):
    css = """
 body{font-family:Helvetica,Arial,sans-serif;max-width:1040px;margin:36px auto;
      color:#1c1c1c;line-height:1.5;padding:0 16px}
 h1{font-size:25px} h2{font-size:19px;margin-top:38px}
 .sub{color:#666;font-size:14px}
 .arm{display:flex;align-items:center;gap:18px;border:1px solid #e4e4e4;
      border-radius:10px;padding:12px 16px;margin:12px 0}
 .arm .label{width:210px;flex:none}
 .arm .frames{display:flex;gap:8px;align-items:center}
 .tag{font-size:12px;color:#fff;border-radius:5px;padding:2px 8px;display:inline-block}
 .nums{font-size:13px;color:#333}
 .arrow{color:#999;font-size:22px}
 .good{color:#1a7a36;font-weight:bold} .bad{color:#c0392b;font-weight:bold}
 .cap{font-size:12px;color:#777;text-align:center}
"""
    H = [f"<!doctype html><html><head><meta charset='utf-8'>"
         f"<title>Power-maximizing assistant in MBAG</title><style>{css}</style></head><body>"]
    H.append("<h1>Power maximization as an alternative assistant objective in MBAG</h1>")
    H.append("<p class='sub'>Same environment (<code>MbagEnv</code>), same model-based "
             "planning, same goal-directed human prior &pi;<sub>h</sub>, same goal set. "
             "Only the assistant's <b>objective</b> changes: maximize completion of the "
             "<i>inferred</i> goal (AssistanceZero) vs. maximize aggregate human power "
             "<code>U<sub>r</sub></code> over all possible goals, goal-agnostically "
             "(Heitzig &amp; Potham). No training &mdash; the assistant plans against its "
             "objective by rolling &pi;<sub>h</sub> forward in the real simulator.</p>")

    arm_color = {"no assistant": "#888", "goal-inference": "#d35400", "power": "#1a7a36"}

    for S in summaries:
        ws = S["ws"]
        H.append(f"<h2>{S['name']}</h2>")
        # possible goals strip
        goals_svg = "".join(
            f"<div><div class='cap'>G{i}{' (true)' if i==S['true_idx'] else ''}</div>{render_state(g, ws)}</div>"
            for i, g in enumerate(S["goal_set"]))
        H.append(f"<div class='sub'>Possible goals (hidden from the robot):</div>"
                 f"<div style='display:flex;gap:10px'>{goals_svg}</div>")
        for arm, r in S["results"].items():
            key = ("power" if arm.startswith("power")
                   else "goal-inference" if arm.startswith("goal") else "no assistant")
            v = r["true_veh"]
            cls = "good" if v > 0.99 else "bad"
            outcome = ("completes true goal" if v > 0.99
                       else f"only {int(v*100)}% of true goal")
            H.append(
                f"<div class='arm'><div class='label'>"
                f"<span class='tag' style='background:{arm_color[key]}'>{arm}</span><br>"
                f"<div class='nums'>plan: <b>{r['plan']}</b><br>"
                f"power left X<sub>h</sub>={r['xh']:.2f} &middot; "
                f"goals attainable {r['attainable']}/{len(S['goal_set'])}<br>"
                f"<span class='{cls}'>{outcome}</span></div></div>"
                f"<div class='frames'>"
                f"<div><div class='cap'>after assistant</div>{render_state(r['state1']['current_blocks'], ws)}</div>"
                f"<span class='arrow'>&rarr;</span>"
                f"<div><div class='cap'>human builds true goal</div>{render_state(r['frames'][-1], ws)}</div>"
                f"</div></div>")
    H.append("<p class='sub'>Bedrock = obstacle only the assistant can clear "
             "(the fork's MBAG modification). Gray = stone the human builds; "
             "dark gray = bedrock; brown = dirt floor.</p>")
    H.append("</body></html>")
    with open(path, "w") as f:
        f.write("".join(H))


def main():
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    summaries = []
    for builder in (vignette_open_options, vignette_dont_foreclose):
        summaries.append(run_vignette(builder()))
        print()
    write_pitch_html(summaries, os.path.join(out, "pitch.html"))
    print(f"Wrote {out}/pitch.html")
    return summaries


if __name__ == "__main__":
    main()
