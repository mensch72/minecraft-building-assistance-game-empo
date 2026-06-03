#!/usr/bin/env python3
"""Verify we can load the REAL trained policies from the shipped checkpoint.

This proves the bridge to the actual published AssistanceZero assistant
(data/assistancezero_assistant/checkpoint_002000), which contains a `human`
policy and an `assistant` policy trained with MbagAlphaZero on an 11x10x10
craftassist world.
"""

import ray

# Importing these registers the custom models / trainable that the checkpoint
# references (ModelCatalog + register_trainable side effects).
import mbag.rllib.torch_models  # noqa: F401
import mbag.rllib.alpha_zero.alpha_zero  # noqa: F401
from mbag.rllib.os_utils import configure_ray_environment
from mbag.rllib.training_utils import load_policy

CKPT = "data/assistancezero_assistant/checkpoint_002000"


def main():
    configure_ray_environment()
    ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False)
    for pid in ("human", "assistant"):
        policy = load_policy(CKPT, pid)
        model = getattr(policy, "model", None)
        print(f"[{pid}] loaded: {type(policy).__name__}")
        print(f"     obs_space   = {policy.observation_space}")
        print(f"     act_space   = {policy.action_space}")
        if model is not None:
            n = sum(p.numel() for p in model.parameters())
            print(f"     model       = {type(model).__name__}  ({n:,} params)")
    print("OK: real trained policies load.")


if __name__ == "__main__":
    main()
