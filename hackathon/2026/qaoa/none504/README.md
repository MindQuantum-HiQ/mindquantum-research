# NONE504 - 2026 QAOA Track Submission

This directory contains team **NONE504**'s submission for the 2026 quantum
computing hackathon QAOA track.

## Overview

The project solves a multi-objective Ising optimization benchmark with two
entry points:

- `main1`: quantum-circuit sampling for small 20-qubit, five-objective
  instances under a fixed 100,000-shot budget. It combines a structured
  scalarization-weight QAOA sweep with finite-depth gate-based digitized
  annealing. The two circuit families produce complementary sampling
  distributions; the returned samples are evaluated together as one Pareto
  candidate set.
- `main2`: exact large-instance post-processing for 200,000 random samples. It
  preserves the serial PCG64 random stream, evaluates Ising energies in chunks,
  extracts local first fronts, and performs exact cross-dominance merging.

The large-instance path changes the computation order, not the definition of
dominance or hypervolume. The validation target is the same samples, objective
vectors, non-dominated frontier, non-dominated count, and hypervolume as the
baseline.

## Directory layout

```text
none504/
├── README.md
├── requirements.txt
└── src/
    ├── answer.py
    ├── baseline.py
    ├── run.py
    ├── transfer_data.csv
    ├── utils.py
    └── data/
        └── w_pool_k5_n1000_seed2026.json
```

`answer.py` contains the submitted solver. The remaining files are the support
code and fixed transfer/weight data needed to reproduce the implementation in
the organizer's evaluation layout.

## Environment

The final submission was tested with:

- Python 3.11.15
- MindQuantum 0.12.0
- NumPy 2.4.3
- pygmo 2.19.8

Create a clean environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Solver interfaces

The judge-facing interfaces are:

```python
from answer import main1, main2

small_result = main1(
    problem_input,
    sample_budget=100000,
    rng_seed=2026,
)

large_result = main2(
    problem_input,
    shots=200000,
    rng_seed=2026,
    chunk_size=4096,
)
```

`problem_input` may be an organizer `.npz` path, an `IsingMOOProblem`, or the
dictionary representation accepted by `src/utils.py`.

## Reproduction

The organizer datasets are not redistributed in this directory. Place the
official data in the following layout:

```text
src/data/public/
src/data/large/
```

Then run from `src/`:

```bash
python run.py --split public --max-cases 0 --large-shots 200000 \
  --out results/none504_public.json
```

For a quick source check that does not require the benchmark data:

```bash
python -m py_compile answer.py utils.py baseline.py run.py
```

## Reported public results

The final defense report records the following results on the organizer's
public benchmark:

| Task | Result |
| --- | --- |
| Small instances | Positive hypervolume increment on 10/10 instances |
| Mean small-instance hypervolume increment | 0.002299 |
| Large-instance average time | 34.31 s baseline to 6.93 s proposed |
| Large-instance frontier consistency | 10/10 |
| Maximum absolute hypervolume difference | 0.00e+00 |

These values describe the fixed public instances and evaluation settings; they
are not presented as a multi-seed statistical significance claim.

## Reproducibility and submission constraints

- The sampling budget is fixed at 100,000 shots for `main1`.
- Random seeds are deterministic and derived from the supplied seed and problem
  coefficients.
- Returned `main1` samples come directly from quantum-circuit measurements.
- `main2` preserves the baseline PCG64 stream using `PCG64.advance`.
- No account credentials, tokens, hidden test data, caches, or generated logs
  are included.

## License

This contribution is submitted under the repository's Apache-2.0 license.
