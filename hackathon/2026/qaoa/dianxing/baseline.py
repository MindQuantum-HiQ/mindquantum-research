from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Union

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from mindquantum.simulator import Simulator

from utils import (
    HV_REF,
    IsingMOOProblem,
    build_qaoa_circuit_from_projected_ising,
    load_transfer_params_csv,
    large_random_frontier_hv,
    problem_from_npz,
    load_weight_pool,
    sampling_result_to_unique_spins,
)
# =========================
# Official baseline budgets
# =========================
NUM_WEIGHTS = 100
SHOTS_PER_WEIGHT = 1000
BASE_SAMPLE_BUDGET = NUM_WEIGHTS * SHOTS_PER_WEIGHT 

# Use transfer angles provided with the repo (p=P_LAYERS).
P_LAYERS = 3
TRANSFER_CSV_PATH = Path(__file__).resolve().parent / "transfer_data.csv"
_TRANSFER_TABLE = load_transfer_params_csv(str(TRANSFER_CSV_PATH), q_target=2, p_list=(P_LAYERS,))
if P_LAYERS not in _TRANSFER_TABLE:
    raise ValueError(f"Missing transfer parameters for p={P_LAYERS} in {TRANSFER_CSV_PATH}.")

def _to_problem(x: Union[str, IsingMOOProblem, dict]) -> IsingMOOProblem:
    if isinstance(x, IsingMOOProblem):
        return x
    if isinstance(x, str):
        return problem_from_npz(x)
    if isinstance(x, dict):
        return IsingMOOProblem(
            name=str(x.get("name", "inline_problem")),
            a=int(x["a"]),
            b=int(x["b"]),
            k=int(x["k"]),
            edges=np.asarray(x["edges"], dtype=np.int32),
            weights=np.asarray(x["weights"], dtype=np.float64),
            h=np.asarray(x["h"], dtype=np.float64),
        )
    raise TypeError(f"Unsupported problem_input type: {type(x)}")


def main1(
    problem_input: Union[str, IsingMOOProblem, dict],
    sample_budget: int = BASE_SAMPLE_BUDGET,
    rng_seed: int | None = None,
) -> Dict[str, object]:
    """Baseline main1: single-round QAOA with fixed 100000-shot budget."""
    problem = _to_problem(problem_input)
    betas, gammas = _TRANSFER_TABLE[P_LAYERS]
    if int(sample_budget) != BASE_SAMPLE_BUDGET:
        raise ValueError(
            f"sample_budget must equal {BASE_SAMPLE_BUDGET}, got {sample_budget}."
        )

    seed = 2026 if rng_seed is None else int(rng_seed)
    # Fair comparison: load a pre-generated lambda pool (1000) shared by baseline/answer.
    lambdas = load_weight_pool(int(problem.k), n=1000, seed=2026)[:NUM_WEIGHTS].astype(np.float64)
    projected_j_pool = np.asarray(lambdas @ problem.weights, dtype=np.float64)
    projected_h_pool = np.asarray(lambdas @ problem.h, dtype=np.float64)

    sim = Simulator("mqvector", int(problem.n), seed=int(seed))

    blocks = np.empty((NUM_WEIGHTS, SHOTS_PER_WEIGHT, int(problem.n)), dtype=np.int8)
    for i in range(NUM_WEIGHTS):
        circ = build_qaoa_circuit_from_projected_ising(
            problem,
            projected_j_pool[i],
            projected_h_pool[i],
            betas=betas,
            gammas=gammas,
            warm_bits01=None,
        )
        sim.reset()
        # Use a deterministic per-weight sampling seed.
        res = sim.sampling(circ, shots=SHOTS_PER_WEIGHT, seed=int(seed + i))
        unique_spins, counts = sampling_result_to_unique_spins(res, int(problem.n))
        blocks[i] = np.repeat(unique_spins, counts.astype(np.int32), axis=0)

    sample_spins = blocks.reshape((-1, int(problem.n)))
    return {"sample_used": int(sample_spins.shape[0]), "sample_spins": sample_spins}


def main2(
    problem_input: Union[str, IsingMOOProblem, dict],
    shots: int = 200000,
    rng_seed: int | None = None,
    chunk_size: int = 4096,
) -> Dict[str, object]:
    """Baseline main2: fast accurate random frontier+HV (README baseline)."""
    problem = _to_problem(problem_input)
    seed = 2026 if rng_seed is None else int(rng_seed)
    return large_random_frontier_hv(problem, shots=int(shots), chunk_size=int(chunk_size), rng_seed=seed, ref=HV_REF)


__all__ = [
    "NUM_WEIGHTS",
    "SHOTS_PER_WEIGHT",
    "BASE_SAMPLE_BUDGET",
    "main1",
    "main2",
]
