import pickle

import pytest

try:
    import torch

    from mbag.rllib.training_utils import load_policies_from_checkpoint, load_trainer
except ImportError:
    pass


@pytest.mark.uses_rllib
def test_load_alpha_zero_assistant_checkpoint(tmp_path):
    load_trainer(
        "data/testing/checkpoints/alpha_zero_assistant/checkpoint_000100",
        "MbagAlphaZero",
        config_updates={"num_workers": 0, "num_gpus": 0, "gpus_per_worker": 0},
    )


@pytest.mark.uses_rllib
def test_load_policies_from_checkpoint_skips_mismatched_weights(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    policy_dir = checkpoint_dir / "policies" / "human"
    policy_dir.mkdir(parents=True)

    with open(policy_dir / "policy_state.pkl", "wb") as policy_state_file:
        pickle.dump(
            {
                "weights": {
                    "position_embedding": torch.zeros((11, 10, 10, 18)),
                    "shared_linear.weight": torch.ones((2, 3)),
                }
            },
            policy_state_file,
        )

    class FakePolicy:
        def __init__(self):
            self.current_weights = {
                "position_embedding": torch.full((33, 10, 10, 18), 7.0),
                "shared_linear.weight": torch.full((2, 3), 5.0),
            }
            self.applied_weights = None

        def get_weights(self):
            return self.current_weights

        def set_weights(self, weights):
            self.applied_weights = weights

    class FakeLocalWorker:
        def __init__(self, policy):
            self.policy = policy

        def foreach_policy(self, func):
            return [func(self.policy, "human")]

    class FakeWorkers:
        def __init__(self, policy):
            self.policy = policy
            self.local = FakeLocalWorker(policy)

        def local_worker(self):
            return self.local

        def foreach_policy(self, func):
            func(self.policy, "human")

    class FakeTrainer:
        def __init__(self, policy):
            self.policy = policy
            self.workers = FakeWorkers(policy)

        def get_policy(self, policy_id):
            assert policy_id == "human"
            return self.policy

    policy = FakePolicy()
    trainer = FakeTrainer(policy)
    load_policies_from_checkpoint(str(checkpoint_dir), trainer)

    assert policy.applied_weights is not None
    assert torch.equal(
        policy.applied_weights["position_embedding"],
        policy.current_weights["position_embedding"],
    )
    assert torch.equal(
        policy.applied_weights["shared_linear.weight"],
        torch.ones((2, 3)),
    )
