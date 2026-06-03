#!/usr/bin/env python3
"""
Milestone 4(a): the human-power metric on the REAL MbagEnv.

This is the standalone MVP idea (empo_mvp/demo.py) grounded in the actual
Minecraft Building Assistance Game environment from this repo:

  - real `MbagEnv` dynamics, blocks, and actions,
  - the fork's real bedrock asymmetry (a human cannot break bedrock; the robot
    assistant can),
  - a goal set Gh of hand-set target structures injected via `set_state`,
  - a goal-directed human driven step-by-step through `env.step(...)`.

For each robot behavior we compute the paper's quantities
(Heitzig & Potham, arXiv:2508.00159):
    Veh(s, g) = how much of goal g the human attains          (eq. 6)
    Xh(s)     = sum_g Veh(s,g)^zeta                            (eq. 7)
    Ur(s)     = -(Xh^{-xi})^eta   (single human)               (eq. 8)

Story: each goal needs a stone block at a cell that starts as BEDROCK. The human
cannot clear bedrock, so it can only build a goal whose bedrock the robot removed.
  - POWER robot          -> clears ALL bedrock  -> human can attain ANY goal.
  - GOAL-INFERENCE robot -> guesses one goal, clears only that -> others blocked.

Run:  .venv/bin/python empo_real/power_on_mbag.py
"""

import copy
import os

import numpy as np

from mbag.environment.actions import MbagAction
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.config import DEFAULT_CONFIG
from mbag.environment.mbag_env import MbagEnv

STONE = MinecraftBlocks.NAME2ID["stone"]
AIR = MinecraftBlocks.AIR
BEDROCK = MinecraftBlocks.BEDROCK

# Paper parameters (the gridworld experiment, Table 5).
ZETA, XI, ETA = 2.0, 1.0, 1.1

WORLD = (5, 4, 5)                       # small, fast
# One structure cell per goal (x, y, z); y=2 sits in the air above the dirt floor.
GOAL_CELLS = [(1, 2, 1), (2, 2, 2), (3, 2, 3)]
HORIZON = 40


def make_env():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["world_size"] = WORLD
    cfg["num_players"] = 1
    cfg["horizon"] = HORIZON
    cfg["num_clutter_blocks"] = 0           # we place obstacles by hand, controlled
    cfg["terminate_on_goal_completion"] = False
    return MbagEnv(cfg)


def goal_for(base_blocks, cell):
    """A goal = the base floor plus a single stone block at `cell`."""
    g = base_blocks.copy()
    g.blocks[cell] = STONE
    g.block_states[cell] = 0
    return g


def world_with_bedrock(base_blocks, bedrock_cells):
    w = base_blocks.copy()
    for c in bedrock_cells:
        w.blocks[c] = BEDROCK
        w.block_states[c] = 0
    return w


def ravel(cell):
    return int(np.ravel_multi_index(cell, WORLD))


def human_action_toward(env, goal_cell):
    """Goal-directed human: get one stone block onto goal_cell.
    It can place into air, and break a NON-bedrock block in the way, but it
    cannot break bedrock (the env enforces this for humans)."""
    cur = env.current_blocks.blocks[goal_cell]
    if cur == STONE:
        return (MbagAction.NOOP, 0, 0)
    if cur == AIR:
        return (MbagAction.PLACE_BLOCK, ravel(goal_cell), int(STONE))
    if cur == BEDROCK:
        return (MbagAction.NOOP, 0, 0)          # cannot fix -> stuck
    return (MbagAction.BREAK_BLOCK, ravel(goal_cell), 0)


def veh(env, base_state, world_blocks, goal_cell):
    """Run the human in `world_blocks` toward the goal at `goal_cell`; return the
    attained value in [0,1] (1.0 iff the stone block ends up placed)."""
    state = copy.deepcopy(base_state)
    state["current_blocks"] = world_blocks.copy()
    state["goal_blocks"] = goal_for(base_state["current_blocks"], goal_cell)
    env.set_state(state)
    for _ in range(HORIZON):
        if env.current_blocks.blocks[goal_cell] == STONE:
            return 1.0
        env.step([human_action_toward(env, goal_cell)])
    return 1.0 if env.current_blocks.blocks[goal_cell] == STONE else 0.0


def human_power_X(vehs):
    return sum(v ** ZETA for v in vehs)


def U_r(vehs):
    x = max(human_power_X(vehs), 1e-9)
    return -((x ** (-XI)) ** ETA)


def main():
    env = make_env()
    env.reset()
    base_state = env.get_state()
    base_blocks = base_state["current_blocks"].copy()  # floor only, no obstacles
    K = len(GOAL_CELLS)

    # Robot behaviors -> which bedrock obstructions remain (uncleared).
    guess = 1  # goal-inference robot guesses goal G1
    behaviors = {
        "none":            set(range(K)),            # nothing cleared
        f"goal-infer(G{guess})": set(range(K)) - {guess},  # clears only guessed
        "power":           set(),                    # clears all
    }

    print("=" * 70)
    print("Human power on the REAL MbagEnv  (bedrock = obstacle only the robot")
    print(f"can clear).  zeta={ZETA} xi={XI} eta={ETA}   goals K={K}")
    print("=" * 70)
    print(f"{'robot behavior':<18}{'Veh per goal':<24}{'Xh':<8}{'Ur':<10}{'attainable':<10}")

    results = {}
    for name, uncleared in behaviors.items():
        bedrock_cells = [GOAL_CELLS[i] for i in sorted(uncleared)]
        world = world_with_bedrock(base_blocks, bedrock_cells)
        vehs = [veh(env, base_state, world, c) for c in GOAL_CELLS]
        xh, ur = human_power_X(vehs), U_r(vehs)
        attainable = sum(1 for v in vehs if v > 0.5)
        results[name] = dict(vehs=vehs, xh=xh, ur=ur, attainable=attainable)
        ur_s = "-inf" if xh < 1e-6 else f"{ur:.3f}"
        veh_s = "[" + ", ".join(f"{v:.2f}" for v in vehs) + "]"
        print(f"{name:<18}{veh_s:<24}{xh:<8.3f}{ur_s:<10}{attainable:<10}")

    print()
    print("True hidden goal = G2 (a goal the goal-inference robot did NOT guess):")
    for name in behaviors:
        v = results[name]["vehs"][2]
        print(f"  under {name:<18} -> human {'ATTAINS G2' if v > 0.5 else 'BLOCKED on G2'}")

    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    write_metrics_svg(os.path.join(out, "metrics.svg"), results, K)
    print(f"\nWrote {out}/metrics.svg")


def write_metrics_svg(path, results, K):
    order = list(results.keys())
    reach = [results[n]["attainable"] for n in order]
    succ = [r / K for r in reach]
    W, H, bw, gap, base_y, max_h = 760, 360, 70, 40, 280, 200
    cols = ["#bbbbbb", "#d95f0e", "#31a354"]
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'font-family="Helvetica,Arial,sans-serif">']
    s.append(f'<text x="{W/2}" y="26" text-anchor="middle" font-size="17" '
             f'font-weight="bold">Human power on the real MbagEnv '
             f'(robot does not know the goal)</text>')

    def chart(x0, title, vals, vmax, fmt):
        s.append(f'<text x="{x0+150}" y="62" text-anchor="middle" font-size="14" '
                 f'font-weight="bold">{title}</text>')
        s.append(f'<line x1="{x0}" y1="{base_y}" x2="{x0+300}" y2="{base_y}" stroke="#999"/>')
        for i, name in enumerate(order):
            v = vals[i]
            bh = (v / vmax) * max_h if vmax else 0
            bx = x0 + 30 + i * (bw + gap)
            s.append(f'<rect x="{bx}" y="{base_y-bh}" width="{bw}" height="{bh}" fill="{cols[i]}"/>')
            s.append(f'<text x="{bx+bw/2}" y="{base_y-bh-8}" text-anchor="middle" '
                     f'font-size="13" font-weight="bold">{fmt(v)}</text>')
            s.append(f'<text x="{bx+bw/2}" y="{base_y+18}" text-anchor="middle" '
                     f'font-size="11">{name}</text>')

    chart(20, f"Goals attainable (of {K})", reach, K, lambda v: f"{v}")
    chart(420, "Success over unknown goal", succ, 1.0, lambda v: f"{int(round(v*100))}%")
    s.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(s))


if __name__ == "__main__":
    main()
