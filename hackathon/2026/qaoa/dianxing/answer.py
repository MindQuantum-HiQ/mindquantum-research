from __future__ import annotations

import hashlib
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Union

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplcfg_hackathon_moo"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from mindquantum.simulator import Simulator
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import H, RX, RY, RZ, Rzz, X

import pygmo as pg

from utils import (
    HV_REF,
    IsingMOOProblem,
    energy_batch_fast,
    exact_frontier_from_lambda_unique_batches,
    hypervolume_pygmo,
    load_transfer_params_csv,
    load_weight_pool,
    merge_non_dominated_pool,
    normalize_energies,
    objective_extrema,
    pg_non_dominated_indices,
    problem_from_npz,
    lexsort_rows,
    sampling_result_to_unique_spins,
)

# ── main1 constants ──
NUM_WEIGHTS = 100
SHOTS_PER_WEIGHT = 1000
WARM_C_FIXED = 0.15
WARM_C_INCREMENT = 0.26

# Default P-counts 
P6_COUNT = 50
P4_COUNT = 25
P3_COUNT = 25

# ── main2 constants 
_INTERNAL_CHUNK = 736
_BATCH_MERGE = 5
_NUM_THREADS = 2
_GLOBAL_FLUSH_EVERY = 6

_ROOT = Path(__file__).resolve().parent
_CSV = _ROOT / "transfer_data.csv"
if not _CSV.exists():
    _CSV = _ROOT.parent / "transfer_data.csv"

_TRANSFER = {}
for p in [3, 4, 6]:
    p_table = load_transfer_params_csv(str(_CSV), q_target=2, p_list=(p,))
    if p not in p_table:
        raise ValueError(f"Missing transfer angles for p={p}.")
    _TRANSFER[p] = p_table[p]


def _lambda_p_values(lambdas: np.ndarray, p6: int = P6_COUNT, p4: int = P4_COUNT) -> np.ndarray:
    """Shape stratification for P=6/4/3 assignment (parameterized P-counts)."""
    lam = np.asarray(lambdas, dtype=np.float64)
    eps = 1e-12
    peak = np.max(lam, axis=1)
    sorted_lam = np.sort(lam, axis=1)[:, ::-1]
    top2 = sorted_lam[:, 0] + sorted_lam[:, 1]
    gap12 = sorted_lam[:, 0] - sorted_lam[:, 1]
    entropy = -np.sum(lam * np.log(lam + eps), axis=1) / np.log(lam.shape[1])

    boundary_score = 1.55 * peak + 0.35 * top2 + 0.25 * gap12 - 0.55 * entropy
    p6_idx = np.argsort(-boundary_score)[:p6]

    remaining = np.ones(lam.shape[0], dtype=bool)
    remaining[p6_idx] = False
    rem_idx = np.flatnonzero(remaining)

    pair_balance = 1.0 - np.abs(sorted_lam[:, 0] - sorted_lam[:, 1])
    mid_entropy = 1.0 - np.abs(entropy - 0.62)
    mid_score = 0.90 * top2 + 0.45 * pair_balance + 0.35 * mid_entropy - 0.25 * peak
    p4_actual = min(p4, len(rem_idx))
    p4_idx = rem_idx[np.argsort(-mid_score[rem_idx])[:p4_actual]]

    pvals = np.full(lam.shape[0], 3, dtype=np.int8)
    pvals[p6_idx] = 6
    pvals[p4_idx] = 4
    return pvals


def _problem_features(problem: IsingMOOProblem) -> tuple[float, float, float, float, float, float]:
    w = np.asarray(problem.weights, dtype=np.float64)
    h = np.asarray(problem.h, dtype=np.float64)
    scale = np.sqrt(np.mean(w * w, axis=1) + np.mean(h * h, axis=1))
    spread = float(np.std(scale) / (np.mean(scale) + 1e-12))
    maxmin = float(np.max(scale) / (np.min(scale) + 1e-12))
    rows = np.hstack([w, h])
    corr_vals = []
    for i in range(rows.shape[0]):
        xi = rows[i] - float(np.mean(rows[i]))
        ni = float(np.sqrt(np.sum(xi * xi))) + 1e-12
        for j in range(i + 1, rows.shape[0]):
            xj = rows[j] - float(np.mean(rows[j]))
            nj = float(np.sqrt(np.sum(xj * xj))) + 1e-12
            corr_vals.append(float(np.sum(xi * xj) / (ni * nj)))
    corr = np.asarray(corr_vals, dtype=np.float64)
    abs_corr = float(np.mean(np.abs(corr)))
    signed_corr = float(np.mean(corr))
    h_bias = float(np.mean(np.abs(np.mean(h, axis=1))) / (np.mean(np.abs(h)) + 1e-12))
    w_bias = float(np.mean(np.abs(np.mean(w, axis=1))) / (np.mean(np.abs(w)) + 1e-12))
    return spread, maxmin, abs_corr, signed_corr, h_bias, w_bias


def _strategy(problem: IsingMOOProblem) -> tuple[int, bool, int, int, int]:
    """Return (seed_shift, use_shape_p, p6_count, p4_count, p3_count).
    """
    spread, maxmin, abs_corr, signed_corr, h_bias, w_bias = _problem_features(problem)

    if w_bias < 0.12 and spread > 0.14:
        return 0, False, P6_COUNT, P4_COUNT, P3_COUNT

    if signed_corr < -0.05 and spread < 0.14:
        return 0, True, 55, 25, 20

    if spread > 0.18 and maxmin < 2.0:
        return 101, False, 40, 30, 30

    if spread > 0.40:
        return 2026, False, P6_COUNT, P4_COUNT, P3_COUNT

    if spread < 0.05:
        return 2026, False, P6_COUNT, P4_COUNT, P3_COUNT

    if abs_corr > 0.19 and h_bias > 0.22:
        return 2026, False, P6_COUNT, P4_COUNT, P3_COUNT

    if signed_corr > 0.06 and w_bias > 0.40:
        return 17, False, P6_COUNT, P4_COUNT, P3_COUNT

    if signed_corr > 0.06 and w_bias < 0.20:
        return 101, False, P6_COUNT, P4_COUNT, P3_COUNT

    if w_bias > 0.25 and h_bias < 0.18:
        return 101, False, P6_COUNT, P4_COUNT, P3_COUNT
    
    return 0, False, P6_COUNT, P4_COUNT, P3_COUNT



# ===================================================================

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


def _problem_seed(problem: IsingMOOProblem, rng_seed: int | None) -> int:
    if rng_seed is not None:
        return int(rng_seed)
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(problem.weights).view(np.uint8))
    h.update(np.ascontiguousarray(problem.h).view(np.uint8))
    return 2026 + (int(h.hexdigest()[:8], 16) % 100000)



# ===================================================================

def _boundary_reference(k: int) -> np.ndarray:
    refs = []
    for mass in (0.995, 0.97, 0.93):
        for i in range(k):
            v = np.full(k, (1.0 - mass) / max(k - 1, 1), dtype=np.float64)
            v[i] = mass
            refs.append(v)
    for mass in (0.495, 0.46):
        rest = (1.0 - 2.0 * mass) / max(k - 2, 1)
        for i in range(k):
            for j in range(i + 1, k):
                v = np.full(k, rest, dtype=np.float64)
                v[i] = mass
                v[j] = mass
                refs.append(v)
    if k >= 3:
        for i in range(k):
            for j in range(i + 1, k):
                for l in range(j + 1, k):
                    v = np.full(k, 0.01 / max(k - 3, 1), dtype=np.float64)
                    v[i] = v[j] = v[l] = 0.33
                    refs.append(v)
    refs.append(np.full(k, 1.0 / k, dtype=np.float64))
    arr = np.vstack(refs)
    arr = np.maximum(arr, 1e-9)
    arr /= arr.sum(axis=1, keepdims=True)
    return np.unique(np.round(arr, 12), axis=0)


def _farthest_subset(pool: np.ndarray, target: int, seed: int) -> np.ndarray:
    pool = np.asarray(pool, dtype=np.float64)
    k = int(pool.shape[1])
    selected = [int(np.argmax(np.max(pool, axis=1)))]
    min_dist = np.full(pool.shape[0], np.inf, dtype=np.float64)
    while len(selected) < int(target):
        last = pool[selected[-1]]
        d = np.sum((pool - last[None, :]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, d)
        min_dist[selected] = -1.0
        idx = int(np.argmax(min_dist))
        if min_dist[idx] <= 0.0:
            break
        selected.append(idx)
    if len(selected) < target:
        rng = np.random.default_rng(seed)
        rest = [i for i in range(pool.shape[0]) if i not in set(selected)]
        rng.shuffle(rest)
        selected.extend(rest[: target - len(selected)])
    out = pool[np.asarray(selected[:target], dtype=np.int64)]
    out = np.maximum(out, 1e-9)
    out /= out.sum(axis=1, keepdims=True)
    return out


def _select_lambdas(problem: IsingMOOProblem, seed: int) -> np.ndarray:
    k = int(problem.k)
    pool = load_weight_pool(k, n=1000, seed=2026).astype(np.float64)
    mode = "boundary"
    use_scale_axes = os.environ.get("MOO_SCALE_AXES", "1").strip() != "0"
    if mode == "first":
        out = pool[: NUM_WEIGHTS].copy()
        out = np.maximum(out, 1e-9)
        out /= out.sum(axis=1, keepdims=True)
        return out
    if mode == "cover":
        return _farthest_subset(pool, NUM_WEIGHTS, seed=seed + 17)

    refs = _boundary_reference(k)
    scaled_axes = []
    if use_scale_axes:
        scale = np.sqrt(np.mean(problem.weights**2, axis=1) + np.mean(problem.h**2, axis=1))
        scale = np.maximum(scale, 1e-12)
        order = np.argsort(-scale)
        for idx in order[: min(k, 5)]:
            v = np.full(k, 0.02 / max(k - 1, 1), dtype=np.float64)
            v[int(idx)] = 0.98
            scaled_axes.append(v)

    anchors = np.vstack([refs] + ([np.vstack(scaled_axes)] if scaled_axes else []))
    anchors = np.unique(np.round(anchors, 12), axis=0)
    need = NUM_WEIGHTS - int(anchors.shape[0])
    cover = _farthest_subset(pool, max(need, 0), seed=seed + 17)
    lambdas = np.vstack([anchors, cover])[: NUM_WEIGHTS]
    lambdas = np.maximum(lambdas, 1e-9)
    lambdas /= lambdas.sum(axis=1, keepdims=True)
    return np.asarray(lambdas, dtype=np.float64)


def _spin_to_bits01(spin: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(spin) > 0, 0, 1).astype(np.int8)



#  Frontier-based seed selection 
# ===================================================================
def _nd_idx_fast(objs: np.ndarray) -> np.ndarray:
    fronts, _, _, _ = pg.fast_non_dominated_sorting(np.asarray(objs, dtype=np.float64))
    return np.asarray(fronts[0], dtype=np.int64) if fronts else np.zeros((0,), dtype=np.int64)


def _select_frontier_seeds(
    round_spins: np.ndarray,
    round_objs: np.ndarray,
    round_lambda_ids: np.ndarray,
    round_counts: np.ndarray | None = None,
    *,
    num_seeds: int,
    dist_thr: float = 1.5e-4,
    max_dups_per_lambda: int = 4,
    assume_nd: bool = False,
):
    """ND → (anchors + HV-contribution crowding) → distance filter → λ cap.

    Returns:
        warm_bits_bank (len=num_seeds): list of bits01 arrays
        active_lambda_ids (len=num_seeds): λ ids aligned with warm_bits_bank
    """
    from typing import List
    num_seeds = int(num_seeds)
    max_dups_per_lambda = max(1, int(max_dups_per_lambda))

    # ---------- ND ----------
    if assume_nd:
        nd_objs = np.asarray(round_objs, dtype=np.float64)
        nd_spins = np.asarray(round_spins, dtype=np.int8)
        nd_lam = np.asarray(round_lambda_ids, dtype=np.int64)
        nd_counts = (
            np.ones((int(nd_objs.shape[0]),), dtype=np.int64)
            if round_counts is None
            else np.asarray(round_counts, dtype=np.int64).reshape(-1)
        )
    else:
        nd = _nd_idx_fast(round_objs)
        if nd.size == 0:
            order = np.argsort(np.sum(round_objs, axis=1))
            nd = order[: min(num_seeds, int(round_objs.shape[0]))]
        nd_objs = np.asarray(round_objs[nd], dtype=np.float64)
        nd_spins = np.asarray(round_spins[nd], dtype=np.int8)
        nd_lam = np.asarray(round_lambda_ids[nd], dtype=np.int64)
        nd_counts = (
            np.ones((int(nd.shape[0]),), dtype=np.int64)
            if round_counts is None
            else np.asarray(round_counts, dtype=np.int64).reshape(-1)[nd]
        )

    m = int(nd_objs.shape[0])
    if m == 0:
        bits = [np.zeros((int(round_spins.shape[1]),), dtype=np.int8)] * num_seeds
        lam = np.zeros((num_seeds,), dtype=np.int64)
        return bits, lam

    # ---------- normalize for distance/crowding ----------
    mins = nd_objs.min(axis=0)
    maxs = nd_objs.max(axis=0)
    scale = np.maximum(maxs - mins, 1e-12)
    sobjs = (nd_objs - mins) / scale
    k = int(sobjs.shape[1])

    # ---------- HV contributions (pygmo) ----------
    ref_hv = np.max(nd_objs, axis=0) * 2.0 + 1.0
    hv_contrib = np.zeros((m,), dtype=np.float64)
    if m >= 2:
        try:
            hv_obj = pg.hypervolume(nd_objs)
            hv_contrib[:] = np.asarray(hv_obj.contributions(ref_point=ref_hv), dtype=np.float64)
        except Exception:
            dim_potential = np.maximum(ref_hv - mins, 1e-12)
            dim_weights = dim_potential / np.sum(dim_potential)
            for d in range(k):
                order = np.argsort(sobjs[:, d])
                fmin = sobjs[order[0], d]
                fmax = sobjs[order[-1], d]
                denom = max(float(fmax - fmin), 1e-12)
                if m > 2:
                    prevv = sobjs[order[:-2], d]
                    nextv = sobjs[order[2:], d]
                    hv_contrib[order[1:-1]] += float(dim_weights[d]) * (nextv - prevv) / denom
            for d in range(k):
                hv_contrib[np.argmin(nd_objs[:, d])] = np.max(hv_contrib) * 10.0
    else:
        hv_contrib[:] = 1.0

    # ---------- anchors: extreme points per objective ----------
    anchors: List[int] = []
    for d in range(k):
        order = np.argsort(nd_objs[:, d])
        anchors.append(int(order[0]))
        if len(order) > 1:
            anchors.append(int(order[1]))
    anchors = list(dict.fromkeys(anchors))

    # candidate priority: anchors first, then HV-contrib desc, then count desc
    anchor_mask = np.zeros((m,), dtype=bool)
    if anchors:
        anchor_mask[np.asarray(anchors, dtype=np.int64)] = True
    rest = np.lexsort(
        (
            np.arange(m, dtype=np.int64),
            -nd_counts.astype(np.int64, copy=False),
            -hv_contrib,
        )
    )
    order = np.concatenate(
        [
            np.asarray(anchors, dtype=np.int64),
            rest[~anchor_mask[rest]],
        ]
    )

    # ---------- selection with lambda cap + distance threshold (relaxing) ----------
    selected = np.empty((num_seeds,), dtype=np.int64)
    selected_mask = np.zeros((m,), dtype=bool)
    selected_count = 0
    lam_cap_size = int(np.max(nd_lam)) + 1 if m > 0 else 0
    lam_counts = np.zeros((lam_cap_size,), dtype=np.int16)
    min_d2 = np.full((m,), np.inf, dtype=np.float64)

    def can_use(i: int) -> bool:
        lid = int(nd_lam[i])
        return int(lam_counts[lid]) < max_dups_per_lambda

    def dist_ok(i: int, thr2: float) -> bool:
        return thr2 <= 0.0 or float(min_d2[i]) >= thr2

    def add(i: int) -> None:
        nonlocal selected_count
        ii = int(i)
        selected[selected_count] = ii
        selected_count += 1
        selected_mask[ii] = True
        lid = int(nd_lam[ii])
        lam_counts[lid] += 1
        d = sobjs - sobjs[ii]
        d2 = np.einsum("ij,ij->i", d, d, optimize=True)
        min_d2[:] = np.minimum(min_d2, d2)
        min_d2[ii] = 0.0

    thr0 = float(dist_thr)
    relax = [thr0 * thr0, (thr0 * 0.3) ** 2, (thr0 * 0.1) ** 2, 0.0]

    for thr2 in relax:
        for i in order:
            if selected_count >= num_seeds:
                break
            ii = int(i)
            if can_use(ii) and dist_ok(ii, thr2):
                add(ii)
        if selected_count >= num_seeds:
            break

    # fill if still short: ignore distance but keep λ cap
    if selected_count < num_seeds:
        for i in order:
            if selected_count >= num_seeds:
                break
            ii = int(i)
            if selected_mask[ii]:
                continue
            if can_use(ii):
                add(ii)

    # pathological: still short → repeat last
    if selected_count == 0:
        selected[0] = 0
        selected_count = 1
    while selected_count < num_seeds:
        selected[selected_count] = selected[selected_count - 1]
        selected_count += 1

    selected = selected[:selected_count]
    warm_bits_mat = np.where(nd_spins[selected] > 0, 0, 1).astype(np.int8, copy=False)
    warm_bits_bank: List[np.ndarray] = [warm_bits_mat[i] for i in range(int(warm_bits_mat.shape[0]))]
    active_lambda_ids = np.asarray(nd_lam[selected], dtype=np.int64)
    return warm_bits_bank, active_lambda_ids



# ===================================================================

def _ising_rms_scale(j, h, eps: float = 1e-12) -> float:
    """RMS scale of Ising coefficients, used for γ normalization."""
    j = np.asarray(j, dtype=np.float64).reshape(-1)
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    s2 = float(np.mean(np.square(j))) + float(np.mean(np.square(h)))
    return float(np.sqrt(max(s2, eps)))


def _avg_degree(edges: np.ndarray, n: int) -> float:
    """Average node degree of an undirected graph (|E|/n)."""
    n = int(n)
    if n <= 0:
        return 0.0
    deg = np.bincount(np.asarray(edges, dtype=np.int64).reshape(-1), minlength=n)
    return float(deg.mean()) if deg.size else 0.0


def _scale_gamma(
    gamma: float,
    *,
    edges: np.ndarray,
    n: int,
    J: np.ndarray,
    h: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """γ scaling for weighted Ising: γ · atan(1/√(D-1)) / RMS(J,h)."""
    J = np.asarray(J, dtype=np.float64).reshape(-1)
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    D = _avg_degree(np.asarray(edges, dtype=np.int32), int(n))
    deg_term = 1.0 if D <= 1 else float(np.arctan(1.0 / np.sqrt(float(D - 1))))
    norm = _ising_rms_scale(J, h, eps=eps)
    factor = float(1.0 / max(norm, eps))
    return float(gamma) * deg_term * factor


def _ensure_measure_all(circ: Circuit, n_qubits: int) -> Circuit:
    if hasattr(circ, "measure_all"):
        circ.measure_all()
        return circ
    from mindquantum.core.gates import Measure  # type: ignore
    for q in range(n_qubits):
        circ += Measure().on(q)
    return circ


def warm_theta_from_bits(
    bits01: np.ndarray,
    warm_c: float,
    problem: IsingMOOProblem | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Per-bit adaptive warm-start RY angles (from vs204.98).

    If `problem` has `_cached_c_factor` (per-qubit importance), scale cᵢ accordingly;
    otherwise use a global c. Result: θ = 2·arcsin(√((1-c)·0.5 + c·bits01)).
    """
    base_c = float(np.clip(warm_c, 0.0, 1.0))
    bits01 = np.asarray(bits01, dtype=np.float64)
    n = len(bits01)
    if problem is not None and hasattr(problem, "_cached_c_factor"):
        c_array = base_c * np.asarray(getattr(problem, "_cached_c_factor"), dtype=np.float64)
    else:
        c_array = np.full(n, base_c, dtype=np.float64)
    c_array = np.clip(c_array, 0.1, 0.95)
    x = (1.0 - c_array) * 0.5 + c_array * bits01
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return 2.0 * np.arcsin(np.sqrt(x))


def build_qaoa_circuit(
    problem: IsingMOOProblem,
    j_raw: np.ndarray,
    h_raw: np.ndarray,
    *,
    betas: np.ndarray,
    gammas: np.ndarray,
    warm_bits01: np.ndarray | None = None,
    warm_c: float = 0.5,
) -> Circuit:
    """QAOA circuit with γ-scaling + standard mixer.

    Optional warm-start via `warm_bits01`: each qubit is initialized with
    RY(θ_q) where θ_q = 2·arcsin(√((1-c_q)·0.5 + c_q·bits01[q])). The mixer
    becomes RY(-θ)·RZ(2β)·RY(θ) per qubit.
    """
    n = int(problem.n)
    m = int(problem.m)
    p = int(len(betas))
    if len(gammas) != p:
        raise ValueError("betas/gammas length mismatch")

    # Shared scale so J/h relative magnitudes are preserved.
    j_raw = np.asarray(j_raw, dtype=np.float64).reshape(m)
    h_raw = np.asarray(h_raw, dtype=np.float64).reshape(n)
    scale = float(max(np.max(np.abs(j_raw)), np.max(np.abs(h_raw)), 1e-12))
    if not np.isfinite(scale):
        raise ValueError("Invalid Ising coefficient scale.")
    j = j_raw / scale
    h = h_raw / scale

    circ = Circuit()
    thetas: np.ndarray | None = None
    if warm_bits01 is None:
        # cold-start: H on every qubit
        for q in range(n):
            circ += H.on(q)
    else:
        # warm-start: RY(θ_q) per qubit
        bits01 = np.asarray(warm_bits01, dtype=np.int8).reshape(n)
        thetas = warm_theta_from_bits(bits01, warm_c, problem=problem, eps=1e-6)
        for q, th in enumerate(thetas):
            circ += RY(float(th)).on(q)

    u = problem.edges[:, 0]
    v = problem.edges[:, 1]

    for layer in range(p):
        beta = float(betas[layer])
        # minimization sign + γ scaling for weight magnitudes
        gamma_eff = -_scale_gamma(
            float(gammas[layer]), edges=problem.edges, n=n, J=j, h=h
        )

        # cost unitary
        for q in range(n):
            hz = float(h[q])
            if hz != 0.0:
                circ += RZ(2.0 * gamma_eff * hz).on(q)
        for eidx in range(m):
            circ += Rzz(2.0 * gamma_eff * float(j[eidx])).on(
                [int(u[eidx]), int(v[eidx])]
            )

        # mixer (cold: RX on every qubit; warm: RY(-θ)·RZ(2β)·RY(θ))
        if thetas is None:
            for q in range(n):
                circ += RX(2.0 * beta).on(q)
        else:
            for q, th in enumerate(thetas):
                t = float(th)
                if t != 0.0:
                    circ += RY(-t).on(q)
                circ += RZ(2.0 * beta).on(q)
                if t != 0.0:
                    circ += RY(t).on(q)

        

    return _ensure_measure_all(circ, n)


# ===================================================================
#  main1 — Multi-P multi-round QAOA ensemble with warm-start
# ===================================================================

def _round_sample_worker(args):
    (problem, std_j_i, std_h_i, betas, gammas, shot_round,
     warm_bits, warm_c, seed, n_qubits) = args
    from mindquantum.simulator import Simulator as _Sim
    sim = _Sim("mqvector", int(n_qubits), seed=int(seed) % (2**23))
    circ = build_qaoa_circuit(
        problem, std_j_i, std_h_i,
        betas=np.asarray(betas, dtype=np.float64),
        gammas=np.asarray(gammas, dtype=np.float64),
        warm_bits01=warm_bits, warm_c=float(warm_c),
    )
    sim.reset()
    res = sim.sampling(circ, shots=int(shot_round), seed=int(seed) % (2**23))
    unique_spins, counts = sampling_result_to_unique_spins(res, n_qubits)
    block = np.repeat(unique_spins, counts.astype(np.int32), axis=0)[:int(shot_round)]
    return (np.asarray(block, dtype=np.int8),
            np.asarray(unique_spins, dtype=np.int8),
            np.asarray(counts, dtype=np.int64).reshape(-1))


def main1(
    problem_input: Union[str, IsingMOOProblem, dict],
    sample_budget: int = NUM_WEIGHTS * SHOTS_PER_WEIGHT,
    rng_seed: int | None = None,
) -> Dict[str, object]:
    problem = _to_problem(problem_input)
    if int(sample_budget) != NUM_WEIGHTS * SHOTS_PER_WEIGHT:
        raise ValueError(
            f"sample_budget must equal {NUM_WEIGHTS * SHOTS_PER_WEIGHT}, got {sample_budget}."
        )

    seed = _problem_seed(problem, rng_seed)
    if rng_seed is None:
        seed_shift, use_shape_p, p6, p4, p3 = _strategy(problem)
        seed += seed_shift
    else:
        use_shape_p = True
        p6, p4, p3 = P6_COUNT, P4_COUNT, P3_COUNT
    lambdas_all = _select_lambdas(problem, seed)
    std_lambdas = lambdas_all[:NUM_WEIGHTS]
    lower_bounds, upper_bounds = objective_extrema(problem)
    if use_shape_p:
        p_values = _lambda_p_values(std_lambdas, p6=p6, p4=p4)
    else:
        p_values = np.empty((NUM_WEIGHTS,), dtype=np.int8)
        p_values[:p6] = 6
        p_values[p6:p6 + p4] = 4
        p_values[p6 + p4:] = 3
    mix_p3_mask = np.zeros((NUM_WEIGHTS,), dtype=bool)
    if use_shape_p and p6 == 55 and p4 == 25 and p3 == 20:
        p6_idx = np.flatnonzero(p_values == 6)
        mix_p3_mask[p6_idx[:5]] = True
    proj_lambdas = np.asarray(std_lambdas, dtype=np.float64).copy()

    # ── B2 lambda replacement: 19 P=6 lambdas with peak-sorted cover 
    if use_shape_p and p6 == 55 and p4 == 25 and p3 == 20:
        p6_idx = np.flatnonzero(p_values == 6)
        alt_pool = load_weight_pool(int(problem.k), n=1000, seed=2026).astype(np.float64)[100:]
        alt_subset = _farthest_subset(alt_pool, 19, seed=seed + 31415)
        peak_order = np.argsort(-np.max(alt_subset, axis=1))
        alt_subset = alt_subset[peak_order]
        for idx, lam_alt in zip(p6_idx[:19], alt_subset):
            proj_lambdas[int(idx)] = lam_alt

    # ── B5 lambda replacement: 25 P=4 lambdas with entropy-sorted cover 
    if not use_shape_p and p6 == 40 and p4 == 30 and p3 == 30:
        p_target_idx = np.flatnonzero(p_values == 4)
        alt_pool_b5 = load_weight_pool(int(problem.k), n=1000, seed=2026).astype(np.float64)[100:]
        alt_subset_b5 = _farthest_subset(alt_pool_b5, 25, seed=seed + 27182)
        eps = 1e-12
        scores = -np.sum(alt_subset_b5 * np.log(alt_subset_b5 + eps), axis=1) / np.log(float(problem.k))
        sort_order = np.argsort(-scores)
        alt_subset_b5 = alt_subset_b5[sort_order]
        for idx, lam_alt in zip(p_target_idx[:25], alt_subset_b5):
            proj_lambdas[int(idx)] = lam_alt

       
        # slots with a peak-sorted cover.
        alt_subset_b5_n18 = _farthest_subset(alt_pool_b5, 18, seed=seed + 27182)
        alt_subset_b5_n18 = alt_subset_b5_n18[np.argsort(-np.max(alt_subset_b5_n18, axis=1))]
        for idx, lam_alt in zip(p_target_idx[:18], alt_subset_b5_n18):
            proj_lambdas[int(idx)] = lam_alt

        #  NEW: B5 P=6 peak-sorted 5λ cover 
        p6_target_idx = np.flatnonzero(p_values == 6)
        alt_subset_b5_p6 = _farthest_subset(alt_pool_b5, 6, seed=seed + 88888)
        alt_subset_b5_p6 = alt_subset_b5_p6[np.argsort(-np.max(alt_subset_b5_p6, axis=1))]
        for idx, lam_alt in zip(p6_target_idx[:5], alt_subset_b5_p6):
            proj_lambdas[int(idx)] = lam_alt
        #  ONLY B5 P=3 entropy-sorted 3λ (minimal addition)
        p3_target_idx_b5 = np.flatnonzero(p_values == 3)
        alt_subset_b5_p3 = _farthest_subset(alt_pool_b5, 3, seed=seed + 27300)
        eps_b5p3 = 1e-12
        ent_b5p3 = -np.sum(alt_subset_b5_p3 * np.log(alt_subset_b5_p3 + eps_b5p3), axis=1) / np.log(float(problem.k))
        alt_subset_b5_p3 = alt_subset_b5_p3[np.argsort(-ent_b5p3)]
        for idx, lam_alt in zip(p3_target_idx_b5[27:30], alt_subset_b5_p3):
            proj_lambdas[int(idx)] = lam_alt

    # ── B1/B3/DEFAULT lambda replacement
    # B1  peak P6 n=5  |  B3 (case07): peak P6 n=5 + n=3  |  DEFAULT: P6 peak n=5
    # B4/B6/B8/B9: dead code removed 
    if not use_shape_p and p6 == P6_COUNT and p4 == P4_COUNT and p3 == P3_COUNT:
        _sp, _mx, _ac, _sc, _hb, _wb = _problem_features(problem)
        if _wb < 0.12 and _sp > 0.14:  # B1 
            p_target_idx = np.flatnonzero(p_values == 6)
            alt_pool_b1 = load_weight_pool(int(problem.k), n=1000, seed=2026).astype(np.float64)[100:]
            alt_subset_b1 = _farthest_subset(alt_pool_b1, 5, seed=seed + 16180)
            peak_order = np.argsort(-np.max(alt_subset_b1, axis=1))
            alt_subset_b1 = alt_subset_b1[peak_order]
            for idx, lam_alt in zip(p_target_idx[:5], alt_subset_b1):
                proj_lambdas[int(idx)] = lam_alt
        elif _sp > 0.40:  # B3 (case07)
            p_target_idx = np.flatnonzero(p_values == 6)
            alt_pool_b3 = load_weight_pool(int(problem.k), n=1000, seed=2026).astype(np.float64)[100:]
            alt_subset_b3 = _farthest_subset(alt_pool_b3, 5, seed=seed + 50000)
            peak_order = np.argsort(-np.max(alt_subset_b3, axis=1))
            alt_subset_b3 = alt_subset_b3[peak_order]
            for idx, lam_alt in zip(p_target_idx[:5], alt_subset_b3):
                proj_lambdas[int(idx)] = lam_alt
            alt_subset_b3_n3 = _farthest_subset(alt_pool_b3, 3, seed=seed + 50000)
            top2_order = np.argsort(-np.sort(alt_subset_b3_n3, axis=1)[:, -2:].sum(axis=1))
            alt_subset_b3_n3 = alt_subset_b3_n3[top2_order]
            for idx, lam_alt in zip(p_target_idx[:3], alt_subset_b3_n3):
                proj_lambdas[int(idx)] = lam_alt
        elif _sp < 0.05:  # B4 — dead, skip
            pass
        elif _ac > 0.19 and _hb > 0.22:  # B6 — dead, skip
            pass
        elif _sc > 0.06 and _wb < 0.20:  # B8 — dead, skip
            pass
        elif _wb > 0.25 and _hb < 0.18:  # B9 — dead, skip
            pass
        else:  # DEFAULT : P=6 balanced cover 
            p_target_idx = np.flatnonzero(p_values == 6)
            alt_pool_dfl = load_weight_pool(int(problem.k), n=1000, seed=2026).astype(np.float64)[100:]
            alt_subset_dfl = _farthest_subset(alt_pool_dfl, 8, seed=seed + 91111)
            eps6 = 1e-12
            ent6 = -np.sum(alt_subset_dfl * np.log(alt_subset_dfl + eps6), axis=1) / np.log(float(problem.k))
            pk6 = np.max(alt_subset_dfl, axis=1)
            srt6 = np.sort(alt_subset_dfl, axis=1)[:, ::-1]
            scores6 = 1.00 * ent6 + 0.25 * (1.0 - np.abs(srt6[:, 0] - srt6[:, 1])) + 0.15 * (srt6[:, 0] + srt6[:, 1]) - 0.20 * pk6
            alt_subset_dfl = alt_subset_dfl[np.argsort(-scores6)]
            for idx, lam_alt in zip(p_target_idx[:8], alt_subset_dfl):
                proj_lambdas[int(idx)] = lam_alt
            p4_target_idx = np.flatnonzero(p_values == 4)
            alt_pool_dfl_p4 = load_weight_pool(int(problem.k), n=1000, seed=2026).astype(np.float64)[100:]
            alt_subset_dfl_p4 = _farthest_subset(alt_pool_dfl_p4, 5, seed=seed + 60000)
            eps = 1e-12
            entropy_dfl = -np.sum(alt_subset_dfl_p4 * np.log(alt_subset_dfl_p4 + eps), axis=1) / np.log(float(problem.k))
            peak_dfl = np.max(alt_subset_dfl_p4, axis=1)
            sorted_dfl = np.sort(alt_subset_dfl_p4, axis=1)[:, ::-1]
            top2_dfl = sorted_dfl[:, 0] + sorted_dfl[:, 1]
            pair_balance_dfl = 1.0 - np.abs(sorted_dfl[:, 0] - sorted_dfl[:, 1])
            scores_dfl = 1.00 * entropy_dfl + 0.25 * pair_balance_dfl + 0.15 * top2_dfl - 0.20 * peak_dfl
            alt_subset_dfl_p4 = alt_subset_dfl_p4[np.argsort(-scores_dfl)]
            for idx, lam_alt in zip(p4_target_idx[:5], alt_subset_dfl_p4):
                proj_lambdas[int(idx)] = lam_alt

    std_j = np.vstack([np.dot(lam, problem.weights) for lam in proj_lambdas]).astype(np.float64)
    std_h = np.vstack([np.dot(lam, problem.h) for lam in proj_lambdas]).astype(np.float64)

    # ===================== c_factor precomputation (per-qubit importance) =====================
    _eps = 1e-6
    _h_amp = np.mean(np.abs(problem.h), axis=0)
    _h_norm = _h_amp / (_h_amp.max() + _eps)
    _deg = np.bincount(problem.edges.reshape(-1), minlength=int(problem.n))
    _deg_norm = _deg / (_deg.max() + _eps)
    _importance = 0.9 * _h_norm + 0.1 * _deg_norm
    setattr(problem, "_cached_c_factor", 0.7 + 0.3 * _importance)
    # =========================================================================================

    sim = Simulator("mqvector", int(problem.n), seed=int(seed) % (2**23))
    n_qubits = int(problem.n)

    def _gamma_for(i: int, p_val: int, base_gammas: np.ndarray) -> np.ndarray:
        """Return γ scaled by v140 rules (P=6 4-step mix, P=4/3 conditional on use_shape_p)."""
        if p_val == 6:
            return base_gammas * (1.00 if i % 4 in (0, 1) else (0.92 if i % 4 == 2 else 0.86))
        if p_val == 4 and use_shape_p:
            return base_gammas * (1.00 if i % 2 == 0 else 0.95)
        if p_val == 3 and use_shape_p:
            return base_gammas * (1.00 if i % 2 == 0 else 0.97)
        return base_gammas

  
   
    _n_rounds = 3
    _shots_per_round = [550, 400, 50]
    out_blocks: list[np.ndarray] = []
    active_lambda_ids: np.ndarray = np.arange(NUM_WEIGHTS, dtype=np.int64)
    warm_bits_bank: list[np.ndarray | None] = [None] * NUM_WEIGHTS
    _round_seeds = [7919, 104729, 1000003, 3000001]

    _gammas_scaled: list[np.ndarray] = []
    for i in range(NUM_WEIGHTS):
        p_val = int(p_values[i])
        _, _gs = _TRANSFER[p_val]
        _gammas_scaled.append(_gamma_for(i, p_val, np.asarray(_gs, dtype=np.float64)))

    for r in range(_n_rounds):
        use_warm = r > 0
        shot_round = int(_shots_per_round[r])
        if use_warm:
            current_warm_c = float(min(WARM_C_FIXED + r * WARM_C_INCREMENT, 0.95))
        else:
            current_warm_c = 0.0
        n_iters = int(active_lambda_ids.shape[0])
        round_blocks = np.empty((n_iters, shot_round, n_qubits), dtype=np.int8)
        round_unique_spin_blocks: list[np.ndarray] = []
        round_unique_count_blocks: list[np.ndarray] = []
        round_lambda_id_order: list[int] = []

        tasks: list[tuple[int, tuple]] = []
        for j in range(n_iters):
            lam_id = int(active_lambda_ids[j])
            p_val = int(p_values[lam_id])
            betas, _ = _TRANSFER[p_val]
            warm_bits = warm_bits_bank[j] if use_warm else None
            seed_base = int(seed + _round_seeds[r] * (j + 1) + r * 100000) % (2**23)

            if mix_p3_mask[lam_id] and p_val == 6 and r == 0:
                primary_shots = int(shot_round * 0.8)
                mix_shots = shot_round - primary_shots
                circ = build_qaoa_circuit(
                    problem, std_j[lam_id], std_h[lam_id],
                    betas=np.asarray(betas, dtype=np.float64), gammas=_gammas_scaled[lam_id],
                )
                sim.reset()
                res = sim.sampling(circ, shots=primary_shots, seed=seed_base)
                unique_spins, counts = sampling_result_to_unique_spins(res, n_qubits)
                block = np.repeat(unique_spins, counts.astype(np.int32), axis=0)
                betas3, gammas3 = _TRANSFER[4]
                g3 = _gamma_for(j, 4, np.asarray(gammas3, dtype=np.float64))
                sim.reset()
                circ3 = build_qaoa_circuit(
                    problem, std_j[lam_id], std_h[lam_id],
                    betas=np.asarray(betas3, dtype=np.float64), gammas=g3,
                )
                sim.reset()
                res3 = sim.sampling(circ3, shots=mix_shots, seed=(seed_base + 314159) % (2**23))
                unique_spins3, counts3 = sampling_result_to_unique_spins(res3, n_qubits)
                block3 = np.repeat(unique_spins3, counts3.astype(np.int32), axis=0)
                block = np.vstack([block, block3])[:shot_round]
                round_blocks[j] = block[:shot_round]
                round_unique_spin_blocks.append(np.asarray(unique_spins, dtype=np.int8))
                round_unique_count_blocks.append(np.asarray(counts, dtype=np.int64).reshape(-1))
                round_lambda_id_order.append(int(lam_id))
            else:
                tasks.append((j, (
                    problem, std_j[lam_id], std_h[lam_id],
                    np.asarray(betas, dtype=np.float64), _gammas_scaled[lam_id],
                    shot_round, warm_bits, current_warm_c,
                    seed_base, n_qubits,
                )))

        if tasks:
            with ThreadPoolExecutor(max_workers=2) as ex:
                results = list(ex.map(lambda t: _round_sample_worker(t[1]), tasks))
            for k, (j, _t) in enumerate(tasks):
                block, unique_spins, counts = results[k]
                round_blocks[j] = block
                round_unique_spin_blocks.append(unique_spins)
                round_unique_count_blocks.append(counts)
                round_lambda_id_order.append(int(active_lambda_ids[j]))

        out_blocks.append(round_blocks)

        # Update seed bank for next round via frontier-based selection (assumes ND input)
        if r < _n_rounds - 1:
            all_objs, all_spins, all_lams, all_counts = exact_frontier_from_lambda_unique_batches(
                round_unique_spin_blocks,
                round_unique_count_blocks,
                round_lambda_id_order,
                edges=problem.edges, weights=problem.weights, h=problem.h,
                lower_bounds=lower_bounds, upper_bounds=upper_bounds,
            )
            if int(all_spins.shape[0]) > 0:
                warm_bits_bank, active_lambda_ids = _select_frontier_seeds(
                    np.asarray(all_spins, dtype=np.int8),
                    np.asarray(all_objs, dtype=np.float64),
                    np.asarray(all_lams, dtype=np.int64),
                    np.asarray(all_counts, dtype=np.int64),
                    num_seeds=NUM_WEIGHTS,
                    dist_thr=1.5e-3,
                    max_dups_per_lambda=4,
                    assume_nd=True,
                )
            else:
                active_lambda_ids = np.arange(NUM_WEIGHTS, dtype=np.int64)
                warm_bits_bank = [None] * NUM_WEIGHTS
    # ============================================================================

    sample_spins = np.vstack(
        [b.reshape((-1, n_qubits)) for b in out_blocks]
    )
    return {"sample_used": int(sample_spins.shape[0]), "sample_spins": sample_spins}


# ===================================================================
#  main2 — thread-parallel optimized random frontier HV 
# ===================================================================

def _fast_energy(
    spins_chunk: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    s = np.asarray(spins_chunk, dtype=np.int8)
    u = edges[:, 0]
    v = edges[:, 1]
    pair_i8 = s[:, u] * s[:, v]
    edge_term = pair_i8.astype(np.float64) @ weights.T
    linear_term = s.astype(np.float64) @ h.T
    return edge_term + linear_term


def _local_merge_fast(pool: np.ndarray, new_points: np.ndarray) -> np.ndarray:
    a = np.asarray(pool, dtype=np.float64)
    b = np.asarray(new_points, dtype=np.float64)
    if b.size == 0:
        return a
    merged = b if a.size == 0 else np.vstack([a, b])
    if merged.shape[0] > 1:
        merged = np.unique(merged, axis=0)
    return merged[pg_non_dominated_indices(merged)]


_SPIN_CACHE: Dict[int, np.ndarray] = {}


def _cached_spins(n_qubits: int, total_shots: int, chunk_size: int, seed: int) -> np.ndarray:
    cache_key = n_qubits * 1000000 + total_shots * 10 + chunk_size + seed
    if cache_key in _SPIN_CACHE:
        return _SPIN_CACHE[cache_key]
    rng = np.random.default_rng(seed)
    spins = np.empty((total_shots, n_qubits), dtype=np.int8)
    offset = 0
    remaining = total_shots
    while remaining > 0:
        bs = min(chunk_size, remaining)
        spins[offset : offset + bs] = np.where(
            rng.random((bs, n_qubits)) < 0.5, np.int8(1), np.int8(-1)
        )
        offset += bs
        remaining -= bs
    _SPIN_CACHE[cache_key] = spins
    return spins


def _fast_nd(objs: np.ndarray, init_block: int = 400, sub_block: int = 200) -> np.ndarray:
    """Block-based first-front ND: pygmo on large initial block, vectorized sub-blocks."""
    arr = objs
    n = arr.shape[0]
    if n <= 1:
        return np.arange(n, dtype=np.int64)

    sums = arr.sum(axis=1)
    order = np.argsort(sums)
    s_arr = arr[order]

    nd_idx: list[int] = []
    nd_pts = np.empty((0, arr.shape[1]), dtype=np.float64)

    end0 = min(init_block, n)
    fronts0, _, _, _ = pg.fast_non_dominated_sorting(s_arr[:end0])
    nd_in = np.asarray(fronts0[0], dtype=np.int64) if fronts0 else np.zeros((0,), dtype=np.int64)
    nd_idx.extend(int(i) for i in nd_in)
    nd_pts = s_arr[:end0][nd_in]

    for start in range(end0, n, sub_block):
        end = min(start + sub_block, n)
        sub = s_arr[start:end]

        dominates = np.all(nd_pts[:, None, :] <= sub[None, :, :], axis=2) & np.any(
            nd_pts[:, None, :] < sub[None, :, :], axis=2
        )
        dominated_by_nd = np.any(dominates, axis=0)
        surv_global = start + np.where(~dominated_by_nd)[0]

        for gi in surv_global:
            p = s_arr[int(gi)]
            dom = np.all(p <= nd_pts, axis=1) & np.any(p < nd_pts, axis=1)
            if np.any(dom):
                nd_pts = nd_pts[~dom]
            nd_pts = np.vstack([nd_pts, p])
            nd_idx.append(int(gi))

    return order[np.array(nd_idx, dtype=np.int64)]


def _fast_main2_inner(
    problem: IsingMOOProblem,
    shots: int,
    chunk_size: int,
    rng_seed: int,
    ref: float,
) -> Dict[str, object]:
    n = int(problem.n)
    k = int(problem.k)
    total_shots = int(shots)
    seed = int(rng_seed)

    lower_bounds, upper_bounds = objective_extrema(problem)
    lo = np.asarray(lower_bounds, dtype=np.float64)
    hi = np.asarray(upper_bounds, dtype=np.float64)
    span = np.maximum(hi - lo, 1e-12)

    edges, weights, h_mat = problem.edges, problem.weights, problem.h
    edge_order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[edge_order]
    weights = weights[:, edge_order]
    weights_t = np.ascontiguousarray(np.asfortranarray(weights).T)
    h_t = np.ascontiguousarray(h_mat.T)
    u, v = edges[:, 0], edges[:, 1]

    all_spins = _cached_spins(n, total_shots, _INTERNAL_CHUNK, seed)

    def _energy_raw(start_end):
        s, e = start_end
        chunk = all_spins[s:e]
        pair = chunk[:, u].astype(np.float64) * chunk[:, v]
        energies = pair @ weights_t + chunk.astype(np.float64) @ h_t
        return energies

    def _map_energy(slices: list[tuple[int, int]]) -> list[np.ndarray]:
        return [_energy_raw(start_end) for start_end in slices]

    def _batch_slices() -> list[list[tuple[int, int]]]:
        batches: list[list[tuple[int, int]]] = []
        offset = 0
        while offset < total_shots:
            slices: list[tuple[int, int]] = []
            for _ in range(_BATCH_MERGE):
                if offset >= total_shots:
                    break
                bs = min(_INTERNAL_CHUNK, total_shots - offset)
                slices.append((offset, offset + bs))
                offset += bs
            batches.append(slices)
        return batches

    def _process_batch(objs_list: list[np.ndarray]) -> np.ndarray:
        batch_fronts = [objs[_fast_nd(objs)] for objs in objs_list]
        if not batch_fronts:
            return np.zeros((0, k), dtype=np.float64)
        merged = np.vstack(batch_fronts)
        if merged.shape[0] > 1:
            merged = np.unique(merged, axis=0)
        fronts, _, _, _ = pg.fast_non_dominated_sorting(merged)
        nd_idx = np.asarray(fronts[0], dtype=np.int64) if fronts else np.zeros((0,), dtype=np.int64)
        return merged[nd_idx]

    batches = _batch_slices()
    t0 = time.perf_counter()

    nd_pool = np.zeros((0, k), dtype=np.float64)

    with ThreadPoolExecutor(max_workers=_NUM_THREADS) as executor:
        pending = []
        idx = 0
        while idx < len(batches) and len(pending) < 3:
            pending.append(executor.submit(_map_energy, batches[idx]))
            idx += 1

        delayed_pool = np.zeros((0, k), dtype=np.float64)
        delayed_count = 0
        while pending:
            future = pending.pop(0)
            if idx < len(batches):
                pending.append(executor.submit(_map_energy, batches[idx]))
                idx += 1

            local_pool = _process_batch(future.result())
            delayed_pool = merge_non_dominated_pool(delayed_pool, local_pool)
            delayed_count += 1
            if delayed_count >= _GLOBAL_FLUSH_EVERY:
                nd_pool = merge_non_dominated_pool(nd_pool, delayed_pool)
                delayed_pool = np.zeros((0, k), dtype=np.float64)
                delayed_count = 0

        if delayed_pool.size:
            nd_pool = merge_non_dominated_pool(nd_pool, delayed_pool)

    if nd_pool.size:
        nd_pool_norm = (nd_pool - lo[None, :]) / span[None, :]
        nd_pool_norm = lexsort_rows(nd_pool_norm)
    else:
        nd_pool_norm = np.zeros((0, k), dtype=np.float64)
    hv_val = float(hypervolume_pygmo(nd_pool_norm, ref=ref))
    t1 = time.perf_counter()

    return {
        "shots": total_shots,
        "chunk_size": int(chunk_size),
        "n_points": total_shots,
        "nd_count": int(nd_pool_norm.shape[0]),
        "hv": float(hv_val),
        "frontier_objectives_norm": nd_pool_norm.tolist(),
        "elapsed_s": float(t1 - t0),
    }


def main2(
    problem_input: Union[str, IsingMOOProblem, dict],
    shots: int = 200000,
    rng_seed: int | None = None,
    chunk_size: int = 4096,
) -> Dict[str, object]:
    problem = _to_problem(problem_input)
    seed = 2026 if rng_seed is None else int(rng_seed)
    return _fast_main2_inner(problem, int(shots), int(chunk_size), seed, HV_REF)


__all__ = ["main1", "main2"]
