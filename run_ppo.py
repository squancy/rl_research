from il.bc import BC
from models.merton import create_merton_model
from policies.analytic import TimeDependentMertonPolicy
from policies.learnable import NNPolicy
from ppo.compare import compare_ppo_vs_il_ppo
from utils.consts import PPOConfig

if __name__ == "__main__":
    config = PPOConfig(
        total_timesteps=500_000,
        n_envs=16,
        rollout_steps=1250,
        eval_interval=20_000,
        state_type="default",
    )

    # 1) Train BC to get a pre-trained policy
    print("=" * 60)
    print("Phase 1: Pre-training with Behavior Cloning")
    print("=" * 60)
    fin_model = create_merton_model(policy_class=TimeDependentMertonPolicy)
    trajectories = fin_model.generate_data(m=100, state_type="default")
    il_policy = NNPolicy(in_dim=3, pi_scale=20.0)
    bc = BC(
        D=trajectories,
        policy=il_policy,
        lr=1e-3,
        epochs=10,
        batch_size=16,
        traj_dataset=False,
        optimizer="adam",
    )
    bc.train()

    il_mean = bc.dataset.mean
    il_std = bc.dataset.std

    # 2) Compare PPO (random) vs PPO (IL pre-trained)
    print("\n" + "=" * 60)
    print("Phase 2: PPO Training")
    print("=" * 60)
    ppo_scratch, ppo_pretrained = compare_ppo_vs_il_ppo(
        il_policy=il_policy,
        il_mean=il_mean,
        il_std=il_std,
        config=config,
        savepath="plots/ppo_vs_il_ppo.png",
    )
