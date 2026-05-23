# dynamic_MPE

Replication notebooks for **Estimating Dynamic Marginal Policy Effects under Sequential Unconfoundedness**.

This repository contains simulation code for estimating **dynamic marginal policy effects (MPEs)**: the local change in discounted welfare induced by a small perturbation of a baseline dynamic policy. The implemented experiments focus on continuous-action **location shifts**, where actions or prices are nudged upward.

## Contents

| File | Description |
|---|---|
| `T=2 .ipynb` | Hidden-state benchmark with horizon `T = 2`. Note the space before `.ipynb`. |
| `T=3.ipynb` | Hidden-state benchmark with horizon `T = 3`. |
| `T=5.ipynb` | Hidden-state benchmark with horizon `T = 5`. |
| `T=10.ipynb` | Hidden-state benchmark with horizon `T = 10`. |
| `simulator.ipynb` | Dynamic pricing simulator with hidden customer heterogeneity and latent reference-price effects. |

The `T=...` notebooks reproduce the hidden-state simulation experiment. `simulator.ipynb` reproduces the dynamic pricing experiment.

## Setup

```bash
git clone https://github.com/IanLai0924/dynamic_MPE.git
cd dynamic_MPE

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn torch matplotlib joblib threadpoolctl jupyterlab
```

Then start Jupyter:

```bash
jupyter lab
```

The notebooks run on CPU. A CUDA-enabled PyTorch installation is recommended for larger Monte Carlo runs.

## Quick start

Open one notebook and run it cell by cell. For example:

```bash
jupyter lab "T=2 .ipynb"
```

or:

```bash
jupyter lab simulator.ipynb
```

For a first smoke test, reduce the Monte Carlo and training settings inside the notebook:

```python
learner = LearnerConfig(
    epochs_q=5,
    epochs_h=5,
    batch_size_q=256,
    batch_size_h=256,
)

mc = MonteCarloConfig(
    N=500,
    R=2,
    n_folds=2,
    oracle_mc=10_000,
    n_jobs=1,
)
```

After the smoke test succeeds, increase `N`, `R`, `oracle_mc`, `n_folds`, `epochs_q`, and `epochs_h` for paper-scale runs.

## Hidden-state benchmark

Use the horizon-specific notebooks:

```text
T=2 .ipynb
T=3.ipynb
T=5.ipynb
T=10.ipynb
```

These notebooks simulate a partially observed dynamic system with observed histories, continuous actions, rewards, and an unobserved latent regime. The feasible estimators only use the observed histories, actions, and rewards.

Main objects:

| Object | Purpose |
|---|---|
| `POMDPConfig` | Data-generating process, horizon, state dimension, discount factor, policy, rewards, and transitions. |
| `LearnerConfig` | Neural-network architecture and optimization settings for the value and score learners. |
| `MonteCarloConfig` | Sample size, replications, folds, oracle finite-difference settings, and parallelism. |

Main functions:

| Function | Purpose |
|---|---|
| `generate_pomdp_benchmark(...)` | Simulates trajectories under the baseline or shifted policy. |
| `estimate_oracle_theta_fd(...)` | Approximates the true MPE by central finite differences. |
| `estimate_mpe_crossfit(...)` | Computes Direct, SRW, ASRW, and oracle-score ASRW using cross-fitting. |

Typical outputs are bias, RMSE, standard errors, and confidence-interval coverage for each estimator.

## Dynamic pricing simulator

Use:

```text
simulator.ipynb
```

This notebook simulates a platform that posts bounded prices over time. Demand depends on current price, seasonality, hidden willingness to pay, a hidden reference price, and a persistent latent taste shock.

The baseline policy is a clipped Gaussian pricing rule. The target MPE is the effect of locally shifting the pricing rule upward before clipping.

Main objects:

| Object | Purpose |
|---|---|
| `PricingDGPConfig` | Pricing DGP, price bounds, latent heterogeneity, demand model, reference-price dynamics, and discounting. |
| `LearnerConfig` | Neural-network and optimizer settings. |
| `MonteCarloConfig` | Sample size, replications, folds, oracle sample size, and parallelism. |

Typical outputs are an estimator comparison table and sampling-distribution plots.

## Estimators

For each period `t`, the notebooks estimate:

```text
q_t(S_t, A_t) = E[Gamma_t | S_t, A_t]
```

where `Gamma_t` is the discounted outcome from period `t` onward. For the continuous location-shift experiments, the direct operator is the action derivative:

```text
L_t q_t(S_t, A_t) = partial q_t(S_t, A_t) / partial A_t.
```

The implemented estimators are:

```text
Direct = mean_i sum_t gamma^(t-1) * d q_hat_t(S_it, A_it) / d A_it

SRW    = mean_i sum_t gamma^(t-1) * H_hat_t(S_it, A_it) * Gamma_it

ASRW   = Direct
         + mean_i sum_t gamma^(t-1) * H_hat_t(S_it, A_it)
           * {Gamma_it - q_hat_t(S_it, A_it)}
```

`ASRW` is the main feasible doubly robust estimator. `ASRW (oracle score)` replaces the learned score with the known analytic score in the simulated designs and is included only as a benchmark.

## Reproducing the paper experiments

For the hidden-state benchmark, run the four horizon notebooks over the desired combinations of:

```text
N in {1000, 2000, 5000}
p in {5, 10}
```

For the pricing experiment, run `simulator.ipynb` with the default pricing DGP and `N = 5000`.

Full replication is computationally expensive because each Monte Carlo replication trains neural networks inside cross-fitting folds. Start small, verify the workflow, then scale up.

## Troubleshooting

**`T=2 .ipynb` does not open from the shell.** The filename contains a space before `.ipynb`; quote it:

```bash
jupyter lab "T=2 .ipynb"
```

**The run is slow.** Reduce `N`, `R`, `oracle_mc`, `n_folds`, `epochs_q`, and `epochs_h` while debugging.

**Memory usage is high.** Reduce `N`, batch sizes, network widths, or `n_jobs`.

**The score learner is unstable.** Use a smooth activation such as `silu` or `gelu`; derivative-based score learning is less stable with ReLU.

## Citation

```text
Lai, I-han, and Stefan Wager. Estimating Dynamic Marginal Policy Effects under Sequential Unconfoundedness. 2026.
```
