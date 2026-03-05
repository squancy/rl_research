from il.bc import BC
from il.dagger import DAgger
from models.jump_diffusion import create_jump_diffusion_model
from models.merton import create_merton_model
from policies.analytic import (
    JumpDiffusionPolicy,
    MertonPolicy,
    TimeDependentJumpDiffusionPolicy,
    TimeDependentMertonPolicy,
)
from policies.learnable import LinearPolicy, NNPolicy, RNNPolicy

if __name__ == "__main__":
    # Evaluate BC using the standard Merton model on 10 trajectories
    # This is enough for BC to learn the optimal policy
    # Adding more trajectories to the training set stabilizes the learning a little
    # more but does not have an effect on performance
    policy = LinearPolicy(in_features=3)
    merton_model = create_merton_model(policy_class=MertonPolicy)
    expert_dataset = merton_model.generate_data(m=10)
    bc = BC(D=expert_dataset, policy=LinearPolicy(in_features=3), epochs=5)
    bc.train()
    bc.compare_to_financial_model(
        financial_model=merton_model,
        m=10,
        savepath="plots/bc_vs_constant_merton.png",
        dpi=600,
    )

    # Weights are close to zero and the bias term is approximately
    # the optimal policy, as expected
    for name, param in policy.named_parameters():
        print(name, param.shape)
        print(param)

    # Evaluate BC on a distribution shift using 4 trajectories and 5 epochs
    merton_model = create_merton_model(policy_class=MertonPolicy)
    expert_dataset = merton_model.generate_data(m=4)
    bc = BC(D=expert_dataset, policy=LinearPolicy(in_features=3), epochs=4)
    bc.train()
    merton_model.params.sigma = 0.4

    bc.compare_to_financial_model(
        financial_model=merton_model,
        m=10,
        plots_to_show=["action_time", "terminal_wealth", "expected_utility"],
        dpi=600,
        savepath="plots/bc_vs_constant_merton_distr_shift.png",
    )

    # Evaluate BC against a time-dependent Merton policy in a POMDP setting
    # States only contain returns and wealths, so BC has to infer the distribution
    # of the mean and volatility of returns
    merton_model = create_merton_model(policy_class=TimeDependentMertonPolicy)
    expert_dataset = merton_model.generate_data(m=100, state_type="pomdp")
    bc = BC(D=expert_dataset, policy=NNPolicy(in_dim=1), epochs=10)
    bc.train()
    bc.compare_to_financial_model(
        financial_model=merton_model,
        savepath="plots/bc_vs_timedep_merton_1year.png",
        dpi=600,
        m=10,
        plots_to_show=["action_time"],
        n_action_time_trajectories=3,
        state_type="pomdp",
    )

    # Evaluate BC against a time-dependent merton policy but the states now
    # contain the actual mean and volatility for each trajectory
    merton_model = create_merton_model(policy_class=TimeDependentMertonPolicy)
    expert_dataset = merton_model.generate_data(m=100, state_type="full")
    bc = BC(D=expert_dataset, policy=NNPolicy(in_dim=3), epochs=10)
    bc.train()
    bc.compare_to_financial_model(
        financial_model=merton_model,
        savepath="plots/bc_vs_timedep_merton_full_obs.png",
        dpi=600,
        m=10,
        state_type="full",
        n_action_time_trajectories=3,
    )

    # Evaluate DAgger against a time-dependent Merton policy using 1-year
    # trajectories in a POMDP setting
    dagger = DAgger(
        expert_policy="time_dep_merton", policy=NNPolicy(in_dim=1), state_type="pomdp"
    )
    dagger.train()
    merton_model = create_merton_model(policy_class=TimeDependentMertonPolicy)
    dagger.bc.compare_to_financial_model(
        financial_model=merton_model,
        m=10,
        savepath="plots/dagger_vs_time_dep_merton_1year.png",
        dpi=600,
        plots_to_show="action_time",
        is_bc=False,
        n_action_time_trajectories=3,
        state_type="pomdp",
    )

    # Evaluate behavior cloning using the time-dependent Merton model
    # Using longer trajectories (15-20 years of trading or more) significantly
    # improves the performance
    # Now, states also contain the empirical mean and variance of the returns up
    # to time t
    # NOTE: change T in consts to 15-20 and uncomment the code below
    """
    merton_model = create_merton_model(policy_class=TimeDependentMertonPolicy)
    expert_dataset = merton_model.generate_data(m=100, state_type="statistic")
    bc = BC(D=expert_dataset, policy=NNPolicy(in_dim=3), epochs=10)
    bc.train()
    bc.compare_to_financial_model(
        financial_model=merton_model,
        savepath="plots/bc_vs_timedep_merton_20year.png",
        dpi=600,
        m=10,
        plots_to_show=["action_time"],
        state_type="statistic",
        n_action_time_trajectories=3,
    )
    """
    jd_model = create_jump_diffusion_model(policy_class=JumpDiffusionPolicy)
    expert_dataset = jd_model.generate_data(m=100)
    bc = BC(D=expert_dataset, policy=NNPolicy(in_dim=3), epochs=10)
    bc.train()
    bc.compare_to_financial_model(
        financial_model=jd_model,
        m=10,
        savepath="plots/bc_vs_jd.png",
        dpi=600,
        plots_to_show=["action_time", "rollout_drift"],
    )

    dagger = DAgger(policy=NNPolicy(in_dim=3), expert_policy="jump_diffusion")
    dagger.train()
    dagger.bc.compare_to_financial_model(
        financial_model=jd_model,
        savepath="plots/dagger_vs_jd.png",
        plots_to_show=["action_time", "rollout_drift"],
    )

    model = create_jump_diffusion_model(policy_class=TimeDependentJumpDiffusionPolicy)

    policy = RNNPolicy(input_dim=1, hidden_dim=128, probabilistic=True)
    D = model.generate_trajectories(m=5000, state_type="pomdp")
    bc = BC(
        D=D,
        policy=policy,
        lr=1e-3,
        epochs=10,
        batch_size=16,
        traj_dataset=True,
        optimizer="adam",
    )
    bc.train()
    bc.diagnose_rnn(financial_model=model, savepath="plots/bc_vs_jd_rnn.png", dpi=600)

    model = create_jump_diffusion_model(policy_class=TimeDependentJumpDiffusionPolicy)
    policy = RNNPolicy(input_dim=1, hidden_dim=128, probabilistic=True)
    dagger = DAgger(
        expert_policy="time_dep_jump_diffusion",
        policy=policy,
        traj_dataset=True,
        state_type="pomdp",
        m=1000,
        bc_optimizer="adam",
        bc_epochs=10,
        bc_batch_size=16,
        base_lr=1e-3,
    )
    dagger.train()
    dagger.bc.diagnose_rnn(
        financial_model=model, savepath="plots/dagger_vs_jd_rnn.png", dpi=600
    )

    model = create_merton_model(policy_class=TimeDependentMertonPolicy)

    policy = RNNPolicy(input_dim=1, hidden_dim=128, probabilistic=True)
    D = model.generate_trajectories(m=5000, state_type="pomdp")
    bc = BC(
        D=D,
        policy=policy,
        lr=1e-3,
        epochs=10,
        batch_size=16,
        traj_dataset=True,
        optimizer="adam",
    )
    bc.train()
    bc.diagnose_rnn(
        financial_model=model, savepath="plots/bc_vs_merton_rnn.png", dpi=600
    )

    model = create_merton_model(policy_class=TimeDependentMertonPolicy)
    policy = RNNPolicy(input_dim=1, hidden_dim=128, probabilistic=True)
    dagger = DAgger(
        expert_policy="time_dep_merton",
        policy=policy,
        traj_dataset=True,
        state_type="pomdp",
        m=1000,
        bc_optimizer="adam",
        bc_epochs=10,
        bc_batch_size=16,
        base_lr=1e-3,
    )
    dagger.train()
    dagger.bc.diagnose_rnn(
        financial_model=model, savepath="plots/dagger_vs_merton_rnn.png", dpi=600
    )
