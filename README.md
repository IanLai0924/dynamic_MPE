# Dynamic marginal policy effects

Replication code for [Estimating Dynamic Marginal Policy Effects under Sequential Unconfoundedness](https://arxiv.org/abs/2604.05639), by I-han Lai and Stefan Wager.

The notebooks study how small changes to a dynamic policy affect long-term welfare. They cover a hidden-state benchmark and a dynamic pricing experiment with unobserved customer heterogeneity and reference-price effects.

## Notebooks

- [T = 2](T%3D2%20.ipynb): hidden-state benchmark with two periods.
- [T = 3](T%3D3.ipynb): hidden-state benchmark with three periods.
- [T = 5](T%3D5.ipynb): hidden-state benchmark with five periods.
- [T = 10](T%3D10.ipynb): hidden-state benchmark with ten periods.
- [Dynamic pricing](simulator.ipynb): a bounded-price experiment with eight periods.

Each notebook is self-contained and generates its own simulated data. The filename `T=2 .ipynb` includes a space before `.ipynb`; keep the quotes when opening it from the command line.

## Setup

The notebooks were saved with Python 3.11.5. To create an environment and install the required packages:

```bash
git clone https://github.com/IanLai0924/dynamic_MPE.git
cd dynamic_MPE

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install numpy scipy scikit-learn torch matplotlib joblib threadpoolctl jupyterlab

jupyter lab "T=2 .ipynb"
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Running the experiments

### Start with a small run

In `T=2 .ipynb`, run the **first code cell only** to load the definitions. Then add a new cell with the following example:

```python
dgp = POMDPConfig(T=2, p=5)
learner = LearnerConfig(
    hidden_dim_q=32,
    hidden_dim_h=32,
    epochs_q=5,
    epochs_h=5,
)
mc = MonteCarloConfig(
    N=500,
    R=2,
    n_folds=2,
    oracle_mc=10_000,
    n_jobs=1,
    make_histograms=False,
)

results = run_monte_carlo(dgp=dgp, learner=learner, mc=mc)
```

This is a setup check, not a configuration for comparing estimator performance. The remaining cells launch larger experiments, so use **Run All** only when you intend to run every configuration.

### Run the full experiments

**Hidden-state benchmark.** After the definitions cell, each horizon notebook contains six experiment cells covering sample sizes `N = 1000, 2000, 5000` and state dimensions `p = 5, 10`. These cells use 1,000 replications and five cross-fitting folds. Run the configurations you need.

**Dynamic pricing.** Open `simulator.ipynb`. Its first cell defines the simulation and estimators; the second runs the experiment and plots the results. The supplied run uses `T = 8`, `N = 5000`, 500 replications, and five folds. Reduce the settings in that cell for an initial check.

Both experiments report bias, RMSE, and confidence-interval coverage. The hidden-state notebooks also produce histograms; the pricing notebook produces histograms and boxplots.

### Adjust the configuration

- `POMDPConfig` or `PricingDGPConfig` controls the data-generating process.
- `LearnerConfig` controls the neural networks and their training settings.
- `MonteCarloConfig` controls sample size, replications, cross-fitting, oracle approximation, and parallelism.

Full runs train neural networks within each replication and fold. To reduce runtime or memory use, start with smaller `N`, `R`, `oracle_mc`, network widths, and epoch counts, and set `n_jobs=1`.

The code supports CPU execution. The hidden-state notebooks select CUDA when available. In the pricing notebook, CUDA is used only when available, `use_cuda_when_possible=True`, and `n_jobs=1`; parallel runs use CPU workers.

## Estimators

The experiments compare four estimators:

- **Direct:** a plug-in estimator based on estimated continuation values.
- **SRW:** score reweighting.
- **ASRW:** augmented score reweighting, the feasible doubly robust estimator.
- **ASRW with oracle score:** a benchmark using the known policy score in the simulated design.

The oracle target is approximated by finite differences. See the [paper](https://arxiv.org/abs/2604.05639) for the identification results, assumptions, and estimator definitions.

## Citation

```bibtex
@misc{lai2026dynamicmpe,
  title         = {Estimating Dynamic Marginal Policy Effects under Sequential Unconfoundedness},
  author        = {Lai, I-han and Wager, Stefan},
  year          = {2026},
  eprint        = {2604.05639},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2604.05639}
}
```
