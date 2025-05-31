## Overview

**Rust_MCS_Sharpe_Ratio** is an example on optimizing a portfolio’s Sharpe ratio under real-world constraints using:

1. **[Rust_MCS minimizer](https://github.com/SergeiGL/Rust_MCS)** (a Rust port of the Multilevel Coordinate Search algorithm)
2. **SciPy’s SLSQP-based `minimize`** (Python)

The goal is to construct a portfolio of 206 stocks (daily return data sourced from MOEX, adjusted for dividends and splits) that maximizes the Sharpe
ratio under practical weight constraints:

- **Total weight constraint:** All asset weights must sum to 100%.
- **Individual weight constraints:** Each selected stock must have a weight between 1% and 10%, or be zero (unselected).

#### Because directly enforcing a 1% minimum is hard, `Python Approach` (naive) is:

1. **First pass (0%–10% bounds):**
    - Use SciPy’s SLSQP (`scipy.optimize.minimize`) to solve for continuous weights with bounds of [0, 0.1].
    - Drop any stocks whose optimal weight is below 1%.
    - Capture the partial portfolio (weights ≥ 1%) and the total weight allocated so far.

2. **Second pass (1%–10% bounds):**
    - Restrict the optimization to the subset of stocks from the first pass.
    - Treat the previously computed “large‐weight” positions as a baseline, and optimize only the small adjustments needed to lift all weights up to
      at least 1%.
    - Enforce that each remaining stock’s final weight ∈ [0.01, 0.1], while the dropped stocks remain at zero.

#### Rust Approach (using `Rust_MCS`)

1. **Define the objective function** (Sharpe ratio) given a vector of weights (each ∈ [0.01, 0.1]):

   Before computing the Sharpe ratio, weights are adjusted as follows:
    - **Threshold**: Set any weight below `MIN_NON_ZERO_WEIGHT` to zero.
    - **Normalize**: Scale the remaining (non-zero) weights so that they sum to 1.
    - **Cap and Redistribute**:
        1. Identify any weight that exceeds `MAX_WEIGHT`.
        2. Cap it at `MAX_WEIGHT`.
        3. Redistribute the excess proportionally among the other non-zero weights.
    - **Repeat**: If any weight still falls outside the range [0.01, 0.1], repeat the process.

2. **Execute** the `Rust_MCS` algorithm to optimize the (adjusted) Sharpe ratio.

## Results

The `Rust_MCS` optimizer outperformed the Python's approach in this task. Specifically:

* **Rust\_MCS Sharpe ratio**: ~0.1256
* **Python (SciPy) Sharpe ratio**: ~0.1253

`Rust_MCS` not only achieved a slightly higher Sharpe ratio,
but also provided sparse solutions that better respected the desired weight constraints (clean 0%, 1%–10% bounds).

This highlights the potential of Rust-based numerical methods in quantitative finance and portfolio optimization.