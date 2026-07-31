from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pygmo as pg

from mindquantum.simulator import Simulator
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import H, RX, RY, RZ, Rzz

HV_REF = 1.01
DEFAULT_EXACT_EXTREMA_CACHE_PATH = Path(__file__).resolve().parent / "results" / "objective_extrema_cache.json"
_EXACT_EXTREMA_CACHE: Dict[str, Dict[str, Dict[str, object]]] = {}

@dataclass
class IsingMOOProblem:
    name: str
    a: int
    b: int
    k: int
    edges: np.ndarray  # [m, 2], int32
    weights: np.ndarray  # [k, m], float64
    h: np.ndarray  # [k, n], float64
    source_path: str | None = None

    @property
    def n(self) -> int:
        return self.a * self.b

    @property
    def m(self) -> int:
        return int(self.edges.shape[0])

def problem_from_npz(path: str) -> IsingMOOProblem:
    data = np.load(path)
    name = str(data["name"]) if "name" in data else path
    return IsingMOOProblem(
        name=name,
        a=int(data["a"]),
        b=int(data["b"]),
        k=int(data["k"]),
        edges=np.asarray(data["edges"], dtype=np.int32),
        weights=np.asarray(data["weights"], dtype=np.float64),
        h=np.asarray(data["h"], dtype=np.float64),
        source_path=str(Path(path).resolve()),
    )


# =========================
# Weight (lambda) pools
# =========================
def default_weight_pool_path(k: int, *, n: int = 1000, seed: int = 2026) -> Path:
    """Default JSON path for cached lambda pools.

    Naming convention: w_pool_k{k}_n{n}_seed{seed}.json stored next to this file.
    """
    here = Path(__file__).resolve().parent
    return here / f"data/w_pool_k{k}_n{int(n)}_seed{int(seed)}.json"

def load_weight_pool(
    k: int,
    *,
    n: int = 1000,
    seed: int = 2026,
    path: str | os.PathLike[str] | None = None,
) -> np.ndarray:
    """Load (or deterministically generate) a cached lambda pool.
    - First tries to read JSON from `path`.
    - If `path` is None, checks env `MOO_WEIGHT_POOL_PATH`.
    - Otherwise falls back to the default packaged pool path.
    """
    k = int(k)
    n = int(n)
    env_path = os.environ.get("MOO_WEIGHT_POOL_PATH", "").strip()
    if path is not None:
        p = Path(path)
    elif env_path:
        p = Path(env_path)
    else:
        p = default_weight_pool_path(k, n=n, seed=seed)

    data = json.loads(p.read_text(encoding="utf-8"))
    arr = np.asarray(data, dtype=np.float64)
    return arr

def spins_to_bitstrings(spins: np.ndarray) -> List[str]:
    arr = np.asarray(spins)
    return ["".join("0" if x > 0 else "1" for x in row) for row in arr]


def bitstring_to_spins(bitstring: str) -> np.ndarray:
    return np.fromiter((1 if c == "0" else -1 for c in bitstring.strip()), dtype=np.int8)


def sampling_result_to_unique_spins(res: object, n_qubits: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert MindQuantum sampling result to unique spins and their counts."""
    if isinstance(res, dict):
        counts = res
    else:
        counts = getattr(res, "data", None)
    if not isinstance(counts, dict):
        raise TypeError(f"Unsupported sampling result type: {type(res)}")

    rows: list[np.ndarray] = []
    reps: list[int] = []
    n_qubits = int(n_qubits)
    for bitstring, cnt in counts.items():
        rep = int(cnt)
        if rep <= 0:
            continue
        b = str(bitstring).strip()
        if b.startswith("0b"):
            b = b[2:]
        b = b.zfill(n_qubits)[-n_qubits:]
        row = np.fromiter((1 if c == "0" else -1 for c in reversed(b)), dtype=np.int8, count=n_qubits)
        rows.append(row)
        reps.append(rep)

    if not rows:
        raise ValueError("Sampling produced zero valid states.")

    unique = np.vstack(rows)
    return np.asarray(unique, dtype=np.int8), np.asarray(reps, dtype=np.int64)


def sampling_result_to_spins(res: object, n_qubits: int, shots: int) -> np.ndarray:
    """Convert MindQuantum sampling result to a dense spin matrix [shots, n]."""
    unique, reps = sampling_result_to_unique_spins(res, n_qubits=int(n_qubits))
    spins = np.repeat(unique, np.asarray(reps, dtype=np.int32), axis=0)
    if int(spins.shape[0]) != int(shots):
        raise ValueError(f"Sampling row count mismatch: got {spins.shape[0]}, expect {shots}")
    return spins


# =========================
# QAOA circuit helpers
# =========================
def ising_rms_scale(j: np.ndarray, h: np.ndarray, eps: float = 1e-12) -> float:
    """RMS scale of Ising coefficients, used for optional normalization."""
    j = np.asarray(j, dtype=np.float64).reshape(-1)
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    s2 = float(np.mean(np.square(j))) + float(np.mean(np.square(h)))
    return float(np.sqrt(max(s2, eps)))


def _avg_degree(edges: np.ndarray, n: int) -> float:
    """Average node degree of an undirected graph.

    For an undirected graph, avg_degree = |E|/n, equivalently mean(deg).
    """
    n = int(n)
    if n <= 0:
        return 0.0
    deg = np.bincount(np.asarray(edges, dtype=np.int64).reshape(-1), minlength=n)
    return float(deg.mean()) if deg.size else 0.0


def scale_gamma(
    gamma: float,
    *,
    edges: np.ndarray,
    n: int,
    J: np.ndarray,
    h: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Gamma scaling for weighted Ising (with both J and h).

    Keep the *degree normalization* term unchanged, and additionally apply the
    weight-magnitude factor for Ising instances with both J (edges) and h (fields):

        gamma_scaled = gamma * atan(1/sqrt(D-1)) * factor
        factor = 1/sqrt( mean(w_uv^2) + mean(h_i^2) )

    where D is the maximum degree, mean(w_uv^2) averages over edges, and mean(h_i^2)
    averages over nodes.

    Note: we internally normalize coefficients by an RMS factor for simulator stability.
    Therefore we also multiply by that RMS so the implemented evolution matches the
    intended raw Hamiltonian scale.
    """
    J = np.asarray(J, dtype=np.float64).reshape(-1)
    h = np.asarray(h, dtype=np.float64).reshape(-1)

    D = _avg_degree(np.asarray(edges, dtype=np.int32), int(n))
    if D <= 1:
        deg_term = 1.0
    else:
        deg_term = float(np.arctan(1.0 / np.sqrt(float(D - 1))))

    norm = ising_rms_scale(J, h, eps=eps)
    factor = float(1.0 / max(norm, eps))
    return float(gamma) * deg_term * factor 
 

def warm_theta_from_bits(bits01: np.ndarray, warm_c: float) -> np.ndarray:
    c = float(np.clip(warm_c, 0.0, 1.0))
    x = (1.0 - c) * 0.5 + c * np.asarray(bits01, dtype=np.float64)
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return 2.0 * np.arcsin(np.sqrt(x))

def ensure_measure_all(circ: Circuit, n_qubits: int) -> Circuit:
    if hasattr(circ, "measure_all"):
        circ.measure_all()
        return circ
    from mindquantum.core.gates import Measure  # type: ignore
    for q in range(n_qubits):
        circ += Measure().on(q)
    return circ

def build_qaoa_circuit(
    problem: IsingMOOProblem,
    lam: np.ndarray,
    *,
    betas: np.ndarray,
    gammas: np.ndarray,
    warm_bits01: np.ndarray | None = None,
    warm_c: float = 0.5,
) -> Circuit:
    """Unified QAOA builder.

    - If warm_bits01 is None: standard QAOA (|+> init + RX mixer).
    - Else: warm-start QAOA (RY init + Egger mixer using the warm bitstring).
    """
    n = int(problem.n)
    m = int(problem.m)
    p = int(len(betas))
    if len(gammas) != p:
        raise ValueError("betas/gammas length mismatch")

    # Build the weighted-sum Ising coefficients for this scalarized objective.
    j_raw = np.asarray(lam @ problem.weights, dtype=np.float64).reshape(m)
    h_raw = np.asarray(lam @ problem.h, dtype=np.float64).reshape(n)
    return build_qaoa_circuit_from_projected_ising(
        problem,
        j_raw,
        h_raw,
        betas=betas,
        gammas=gammas,
        warm_bits01=warm_bits01,
        warm_c=warm_c,
    )


def build_qaoa_circuit_from_projected_ising(
    problem: IsingMOOProblem,
    j_raw: np.ndarray,
    h_raw: np.ndarray,
    *,
    betas: np.ndarray,
    gammas: np.ndarray,
    warm_bits01: np.ndarray | None = None,
    warm_c: float = 0.5,
) -> Circuit:
    """Build QAOA circuit from already projected scalarized Ising coefficients."""
    n = int(problem.n)
    m = int(problem.m)
    p = int(len(betas))
    if len(gammas) != p:
        raise ValueError("betas/gammas length mismatch")

    # Normalize with a shared scale so J/h relative magnitudes are preserved.
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
        for q in range(n):
            circ += H.on(q)
    else:
        bits01 = np.asarray(warm_bits01, dtype=np.int8).reshape(n)
        thetas = warm_theta_from_bits(bits01, warm_c)
        for q, th in enumerate(thetas):
            circ += RY(float(th)).on(q)

    u = problem.edges[:, 0]
    v = problem.edges[:, 1]

    for layer in range(p):
        beta = float(betas[layer])
        # minimization sign + transfer scaling for weights
        gamma_eff = -scale_gamma(float(gammas[layer]), edges=problem.edges, n=n, J=j, h=h)

        # cost unitary
        for q in range(n):
            hz = float(h[q])
            if hz != 0.0:
                circ += RZ(2.0 * gamma_eff * hz).on(q)
        for eidx in range(m):
            circ += Rzz(2.0 * gamma_eff * float(j[eidx])).on([int(u[eidx]), int(v[eidx])])

        # mixer
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

    return ensure_measure_all(circ, n)

def init_simulator(n_qubits: int, seed: int) -> Simulator:
    return Simulator("mqvector", int(n_qubits), seed=int(seed) % (2**23))

def energy_batch_fast(
    spins: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    """Vectorized energy for multiple samples and objectives.

    spins: [s, n], values in {-1, +1}
    edges: [m, 2]
    weights: [k, m]
    h: [k, n]
    returns: [s, k]
    """
    s = np.asarray(spins, dtype=np.float64)
    u = edges[:, 0]
    v = edges[:, 1]
    pair = s[:, u] * s[:, v]  # [s, m]
    edge_term = pair @ weights.T  # [s, k]
    linear_term = s @ h.T  # [s, k]
    return edge_term + linear_term

def objective_bounds(weights: np.ndarray, h: np.ndarray) -> np.ndarray:
    # Conservative symmetric bound where |s_i|=1.
    return np.maximum(np.sum(np.abs(weights), axis=1) + np.sum(np.abs(h), axis=1), 1e-8)


def _problem_signature(problem: IsingMOOProblem) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(problem.edges).view(np.uint8))
    h.update(np.ascontiguousarray(problem.weights).view(np.uint8))
    h.update(np.ascontiguousarray(problem.h).view(np.uint8))
    return h.hexdigest()


def _exact_extrema_cache_path(problem: IsingMOOProblem) -> Path:
    src = getattr(problem, "source_path", None)
    if src:
        path = Path(src)
        parent = path.parent.name
        if parent in {"public", "_hidden"}:
            return path.parent / "objective_extrema_cache.json"
    return DEFAULT_EXACT_EXTREMA_CACHE_PATH


def _load_exact_extrema_cache(cache_path: Path) -> Dict[str, Dict[str, object]]:
    key = str(cache_path.resolve())
    if key in _EXACT_EXTREMA_CACHE:
        return _EXACT_EXTREMA_CACHE[key]
    if not cache_path.exists():
        _EXACT_EXTREMA_CACHE[key] = {}
        return _EXACT_EXTREMA_CACHE[key]
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        _EXACT_EXTREMA_CACHE[key] = raw if isinstance(raw, dict) else {}
    except Exception:
        _EXACT_EXTREMA_CACHE[key] = {}
    return _EXACT_EXTREMA_CACHE[key]


def _save_exact_extrema_cache(cache_path: Path, cache: Dict[str, Dict[str, object]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _state_index_block_to_spins(start: int, count: int, n: int) -> np.ndarray:
    idx = np.arange(int(start), int(start) + int(count), dtype=np.uint32)[:, None]
    bits = (idx >> np.arange(int(n), dtype=np.uint32)[None, :]) & 1
    return np.where(bits == 0, 1, -1).astype(np.int8, copy=False)


def exact_objective_extrema(
    problem: IsingMOOProblem,
    *,
    chunk_size: int = 16384,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact per-objective minima/maxima for small Ising problems via full enumeration."""
    cache_path = _exact_extrema_cache_path(problem)
    cache = _load_exact_extrema_cache(cache_path)
    key = _problem_signature(problem)
    if key in cache:
        entry = cache[key]
        return (
            np.asarray(entry["lower"], dtype=np.float64),
            np.asarray(entry["upper"], dtype=np.float64),
        )

    n = int(problem.n)
    if n > 20:
        raise ValueError(f"Exact objective extrema only supported for n<=20, got {n}")

    total = 1 << n
    k = int(problem.k)
    lower = np.full((k,), np.inf, dtype=np.float64)
    upper = np.full((k,), -np.inf, dtype=np.float64)
    step = max(1, int(chunk_size))
    for st in range(0, total, step):
        bs = min(step, total - st)
        spins = _state_index_block_to_spins(st, bs, n)
        energies = np.asarray(
            energy_batch_fast(spins, problem.edges, problem.weights, problem.h),
            dtype=np.float64,
        )
        lower = np.minimum(lower, energies.min(axis=0))
        upper = np.maximum(upper, energies.max(axis=0))

    cache[key] = {
        "name": str(problem.name),
        "n": n,
        "k": int(problem.k),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
    }
    _save_exact_extrema_cache(cache_path, cache)
    return lower, upper


def objective_extrema(problem: IsingMOOProblem) -> Tuple[np.ndarray, np.ndarray]:
    """Return normalization extrema. Small cases use exact cached extrema."""
    if int(problem.n) <= 20:
        return exact_objective_extrema(problem)
    bounds = objective_bounds(problem.weights, problem.h)
    return -bounds, bounds


def normalize_energies(
    energies: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    # Map exact/conservative objective range [lo, hi] to [0, 1], minimization objective.
    lo = np.asarray(lower_bounds, dtype=np.float64)
    hi = np.asarray(upper_bounds, dtype=np.float64)
    span = np.maximum(hi - lo, 1e-12)
    return (np.asarray(energies, dtype=np.float64) - lo[None, :]) / span[None, :]


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def non_dominated_indices(objs: np.ndarray) -> np.ndarray:
    n = len(objs)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if dominates(objs[j], objs[i]):
                keep[i] = False
                break
    return np.flatnonzero(keep)


def non_dominated_sort(objs: np.ndarray) -> List[List[int]]:
    n = len(objs)
    dom_count = np.zeros(n, dtype=np.int32)
    dom_to: List[List[int]] = [[] for _ in range(n)]
    fronts: List[List[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(objs[i], objs[j]):
                dom_to[i].append(j)
            elif dominates(objs[j], objs[i]):
                dom_count[i] += 1
        if dom_count[i] == 0:
            fronts[0].append(i)

    f = 0
    while f < len(fronts) and fronts[f]:
        nxt: List[int] = []
        for i in fronts[f]:
            for j in dom_to[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        if nxt:
            fronts.append(nxt)
        f += 1
    return fronts


def crowding_distance(objs: np.ndarray, front: List[int]) -> np.ndarray:
    d = np.zeros(len(front), dtype=np.float64)
    if len(front) <= 2:
        d[:] = np.inf
        return d

    vals = objs[np.asarray(front, dtype=np.int32)]
    n_obj = vals.shape[1]
    for c in range(n_obj):
        order = np.argsort(vals[:, c])
        d[order[0]] = np.inf
        d[order[-1]] = np.inf
        lo = vals[order[0], c]
        hi = vals[order[-1], c]
        span = hi - lo
        if span <= 1e-12:
            continue
        for i in range(1, len(front) - 1):
            left = vals[order[i - 1], c]
            right = vals[order[i + 1], c]
            d[order[i]] += (right - left) / span
    return d


def hypervolume_pygmo(objs: np.ndarray, ref: float | np.ndarray = HV_REF) -> float:
    """Exact hypervolume by pygmo for minimization objectives."""
    arr = np.asarray(objs, dtype=np.float64)
    if arr.size == 0:
        return 0.0

    if np.isscalar(ref):
        ref_vec = np.full(arr.shape[1], float(ref), dtype=np.float64)
    else:
        ref_vec = np.asarray(ref, dtype=np.float64).reshape(-1)
        if ref_vec.shape[0] != arr.shape[1]:
            raise ValueError(f"ref dimension mismatch: got {ref_vec.shape[0]}, expect {arr.shape[1]}")

    # For minimization HV with a finite reference point, points outside ref do not contribute.
    mask = np.all(arr <= ref_vec[None, :], axis=1)
    arr = arr[mask]
    if arr.size == 0:
        return 0.0

    # Exact HV is unchanged by duplicate objective vectors.
    arr = np.unique(arr, axis=0)
    nd = arr[pg_non_dominated_indices(arr)]
    if nd.size == 0:
        return 0.0
    return float(pg.hypervolume(nd).compute(ref_vec))


def pg_non_dominated_indices(objs: np.ndarray) -> np.ndarray:
    """Fast first-front indices via pygmo (much faster than O(n^2) check)."""
    arr = np.asarray(objs, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    fronts, _, _, _ = pg.fast_non_dominated_sorting(arr)
    if not fronts:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(fronts[0], dtype=np.int64)


def merge_non_dominated_pool(pool: np.ndarray, new_points: np.ndarray) -> np.ndarray:
    """Merge-and-filter a normalized objective pool, keeping only the ND set."""
    a = np.asarray(pool, dtype=np.float64)
    b = np.asarray(new_points, dtype=np.float64)
    if b.size == 0:
        return a
    merged = b if a.size == 0 else np.vstack([a, b])
    if int(merged.shape[0]) > 1:
        merged = np.unique(merged, axis=0)
    return merged[pg_non_dominated_indices(merged)]


def lexsort_rows(arr: np.ndarray) -> np.ndarray:
    """Return rows sorted lexicographically for stable serialization."""
    mat = np.asarray(arr)
    if mat.ndim != 2 or int(mat.shape[0]) <= 1:
        return mat
    order = np.lexsort(mat[:, ::-1].T)
    return mat[order]


def exact_frontier_from_lambda_unique_batches(
    unique_spin_blocks: Sequence[np.ndarray],
    unique_count_blocks: Sequence[np.ndarray],
    lambda_id_order: Sequence[int],
    *,
    edges: np.ndarray,
    weights: np.ndarray,
    h: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    chunk_size: int = 8192,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact ND frontier from per-lambda unique samples.

    This keeps all information needed for exact seed selection:
    - one row per unique spin within each lambda
    - original per-spin counts preserved
    - exact objective values computed once per globally unique spin
    """
    if not (len(unique_spin_blocks) == len(unique_count_blocks) == len(lambda_id_order)):
        raise ValueError("unique spin/count/lambda batches length mismatch")
    if len(unique_spin_blocks) == 0:
        k = int(np.asarray(lower_bounds).shape[0])
        n = int(np.asarray(h).shape[1])
        return (
            np.zeros((0, k), dtype=np.float64),
            np.zeros((0, n), dtype=np.int8),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    spin_blocks = [np.asarray(x, dtype=np.int8) for x in unique_spin_blocks]
    count_blocks = [np.asarray(x, dtype=np.int64).reshape(-1) for x in unique_count_blocks]
    if any(int(spins.shape[0]) != int(counts.shape[0]) for spins, counts in zip(spin_blocks, count_blocks)):
        raise ValueError("unique spin/count size mismatch within a lambda block")

    all_unique_spins = np.vstack(spin_blocks)
    global_unique_spins, inverse = np.unique(all_unique_spins, axis=0, return_inverse=True)

    k = int(np.asarray(lower_bounds).shape[0])
    global_objs = np.empty((int(global_unique_spins.shape[0]), k), dtype=np.float64)
    step = max(1, int(chunk_size))
    for st in range(0, int(global_unique_spins.shape[0]), step):
        ed = min(st + step, int(global_unique_spins.shape[0]))
        energies = np.asarray(
            energy_batch_fast(global_unique_spins[st:ed], edges, weights, h),
            dtype=np.float64,
        )
        global_objs[st:ed] = normalize_energies(energies, lower_bounds, upper_bounds)

    block_objs_parts: List[np.ndarray] = []
    block_spins_parts: List[np.ndarray] = []
    block_lam_parts: List[np.ndarray] = []
    block_count_parts: List[np.ndarray] = []
    offset = 0
    for spins, counts, lam_id in zip(spin_blocks, count_blocks, lambda_id_order):
        size = int(spins.shape[0])
        obj_idx = inverse[offset : offset + size]
        offset += size
        objs = np.asarray(global_objs[obj_idx], dtype=np.float64)
        local_nd = pg_non_dominated_indices(objs)
        if int(local_nd.size) == 0:
            continue
        block_objs_parts.append(np.asarray(objs[local_nd], dtype=np.float64))
        block_spins_parts.append(np.asarray(spins[local_nd], dtype=np.int8))
        block_lam_parts.append(np.full((int(local_nd.size),), int(lam_id), dtype=np.int64))
        block_count_parts.append(np.asarray(counts[local_nd], dtype=np.int64))

    if not block_objs_parts:
        n = int(np.asarray(h).shape[1])
        return (
            np.zeros((0, k), dtype=np.float64),
            np.zeros((0, n), dtype=np.int8),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    all_objs = np.vstack(block_objs_parts)
    all_spins = np.vstack(block_spins_parts)
    all_lams = np.concatenate(block_lam_parts)
    all_counts = np.concatenate(block_count_parts)
    keep = pg_non_dominated_indices(all_objs)
    return (
        np.asarray(all_objs[keep], dtype=np.float64),
        np.asarray(all_spins[keep], dtype=np.int8),
        np.asarray(all_lams[keep], dtype=np.int64),
        np.asarray(all_counts[keep], dtype=np.int64),
    )


def large_random_frontier_hv(
    problem: IsingMOOProblem,
    *,
    shots: int = 100000,
    chunk_size: int = 512,
    rng_seed: int = 2026,
    ref: float = HV_REF,
) -> Dict[str, object]:
    """Baseline/answer main2 helper: random sampling + fast chunked ND merge + exact HV.

    Returns a dict compatible with README's main2 requirements.
    """
    rng = np.random.default_rng(int(rng_seed))
    lower_bounds, upper_bounds = objective_extrema(problem)
    k = int(problem.k)

    remaining = int(shots)
    nd_pool = np.zeros((0, k), dtype=np.float64)
    n_points = 0

    t0 = time.perf_counter()
    while remaining > 0:
        bs = min(int(chunk_size), remaining)
        spins = np.where(rng.random((bs, int(problem.n))) < 0.5, 1, -1).astype(np.int8)
        energies = np.asarray(energy_batch_fast(spins, problem.edges, problem.weights, problem.h), dtype=np.float64)
        objs = normalize_energies(energies, lower_bounds, upper_bounds)
        nd_pool = merge_non_dominated_pool(nd_pool, objs[pg_non_dominated_indices(objs)])
        n_points += bs
        remaining -= bs
    t1 = time.perf_counter()

    nd_pool = np.asarray(lexsort_rows(nd_pool), dtype=np.float64)
    hv = float(hypervolume_pygmo(nd_pool, ref=ref))
    return {
        "shots": int(shots),
        "chunk_size": int(chunk_size),
        "n_points": int(n_points),
        "nd_count": int(nd_pool.shape[0]),
        "hv": float(hv),
        "frontier_objectives_norm": nd_pool.tolist(),
        "elapsed_s": float(t1 - t0),
    }

def problem_feature(problem: IsingMOOProblem) -> np.ndarray:
    abs_w = np.abs(problem.weights)
    return np.array(
        [
            float(problem.k),
            float(problem.n),
            float(problem.m),
            float(abs_w.mean()),
            float(abs_w.std()),
            float(np.abs(problem.h).mean()),
        ],
        dtype=np.float64,
    )


def objective_fields(spins: np.ndarray, edges: np.ndarray, weights: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Compute field[k, n] at given spin config for each objective."""
    z = np.asarray(spins, dtype=np.float64)
    k, n = h.shape
    out = h.copy()
    u = edges[:, 0]
    v = edges[:, 1]
    for t in range(k):
        w = weights[t]
        np.add.at(out[t], u, w * z[v])
        np.add.at(out[t], v, w * z[u])
    return out


def load_transfer_params_csv(csv_path: str, q_target: int, p_list: Sequence[int]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Parse transfer_data.csv into round -> (betas, gammas).

    CSV rows follow:
      q,p,nu,gamma_1..gamma_p,beta_1..beta_p
    Returns rounds in integer p keys.
    """
    p_set = {int(p) for p in p_list}
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        lines = f.read().splitlines()
    if len(lines) < 3:
        return out

    reader = csv.reader(lines[1:])  # skip first description line
    rows = list(reader)
    if len(rows) < 2:
        return out

    for r in rows[1:]:  # skip header line
        if not r or len(r) < 4:
            continue
        try:
            q = int(float(r[0]))
            p = int(float(r[1]))
        except Exception:
            continue
        if q != int(q_target) or p not in p_set:
            continue

        try:
            gammas = np.asarray([float(x) for x in r[3 : 3 + p]], dtype=np.float64)
            betas = np.asarray([float(x) for x in r[3 + p : 3 + (2 * p)]], dtype=np.float64)
        except Exception:
            continue

        if gammas.size != p or betas.size != p:
            continue
        out[int(p)] = (betas, gammas)

    return out
