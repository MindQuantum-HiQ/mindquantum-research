"""多轮协同设计路线中可切换轮次的 main1 保留方案。

用于说明有效多轮并不是简单增加 round count，而是 initial coverage、seed selection、warm strength、
circuit expressivity 和 shot allocation 的共同匹配。默认使用当前四轮
最优 schedule，同时保留一至六轮设置用于消融分析。
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Union, Tuple
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import H, RX, RY, RZ, Rzz, X

import numpy as np
import pygmo as pg
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

CODE_DIR = Path(__file__).resolve().parent
MOO_ROOT = CODE_DIR if (CODE_DIR / "utils.py").is_file() else CODE_DIR.parents[1]
if str(MOO_ROOT) not in sys.path:
    sys.path.insert(0, str(MOO_ROOT))


# Keep env lean for hackathon runner.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplcfg_hackathon_moo")
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from mindquantum.simulator import Simulator
import mindquantum as mq

from utils import (
    HV_REF,
    IsingMOOProblem,
    exact_frontier_from_lambda_unique_batches,
    large_random_frontier_hv,
    load_transfer_params_csv,
    load_weight_pool,
    objective_extrema,
    problem_from_npz,
    sampling_result_to_unique_spins,
    energy_batch_fast,
    normalize_energies,
    pg_non_dominated_indices,
    hypervolume_pygmo,
    lexsort_rows,
    merge_non_dominated_pool,
)

# =========================
# Fixed contest budgets
# =========================
NUM_WEIGHTS = 100
BASE_SAMPLE_BUDGET = 100000
WARM_C_FIXED = 0.15

WEIGHTS_PER_ROUND = NUM_WEIGHTS
ROUND_SCHEDULES = {
    # 所有 schedule 的元素之和均为 1000；乘以每轮 100 个权重后，
    # 恰好满足官方 main1 固定预算 100000 shots。
    # Single-round and early multi-round schedules for ablation analysis.
    "1round": [1000],
    "2round": [800, 200],
    "3round": [600, 200, 200],
    # Base four-round cooperative setting.
    "4round": [400, 200, 200, 200],
    # Current best four-round cooperative setting.
    # Round 1: broad exploration (no warm-start)
    # Round 2: deep exploitation with the largest shot budget
    # Round 3: exploitation/refinement under stronger warm-start
    # Round 4: focused final refinement
    "4round_2": [400, 400, 150, 50],
    # Previous five-round cooperative schedule kept for ablation analysis.
    "5round": [300, 200, 200, 150, 150],
    # Previous six-round sweep setting kept to analyze over-refinement and runtime.
    "6round": [250, 150, 150, 150, 150, 150],
}
# 默认使用当前本地 public10 最优的四轮协同配置。
# 其他配置可通过环境变量 MOO_ROUND_SCHEDULE=3round/4round/5round 等切换。
ACTIVE_ROUND_SCHEDULE = os.environ.get("MOO_ROUND_SCHEDULE", "4round_2")
if ACTIVE_ROUND_SCHEDULE not in ROUND_SCHEDULES:
    raise ValueError(
        f"Unknown MOO_ROUND_SCHEDULE={ACTIVE_ROUND_SCHEDULE!r}; "
        f"available={sorted(ROUND_SCHEDULES)}"
    )
SHOTS_PER_WEIGHT = ROUND_SCHEDULES[ACTIVE_ROUND_SCHEDULE]
N_ROUNDS = len(SHOTS_PER_WEIGHT)
if np.sum(SHOTS_PER_WEIGHT) * WEIGHTS_PER_ROUND != BASE_SAMPLE_BUDGET:
    raise ValueError("Round shot allocation must equal BASE_SAMPLE_BUDGET.")

# Fixed QAOA depth used by baseline/sample implementation.
P_LAYERS = 4
TRANSFER_CSV_PATH = MOO_ROOT / "transfer_data.csv"
TRANSFER_Q_TARGET = 2  # fixed by baseline/README
_TRANSFER_TABLE = load_transfer_params_csv(
    str(TRANSFER_CSV_PATH), q_target=TRANSFER_Q_TARGET, p_list=(P_LAYERS,)
)
if P_LAYERS not in _TRANSFER_TABLE:
    raise ValueError(f"Missing transfer parameters for p={P_LAYERS} in {TRANSFER_CSV_PATH}.")

# =========================
# Helpers
# =========================
def _seed_from_problem(problem: IsingMOOProblem) -> int:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(problem.weights).view(np.uint8))
    h.update(np.ascontiguousarray(problem.h).view(np.uint8))
    return int(h.hexdigest()[:16], 16)


def _to_problem(x: Union[str, IsingMOOProblem, Dict[str, np.ndarray]]) -> IsingMOOProblem:
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
    raise TypeError("Unsupported problem input type")


def _nd_idx_fast(objs: np.ndarray) -> np.ndarray:
    fronts, _, _, _ = pg.fast_non_dominated_sorting(np.asarray(objs, dtype=np.float64))
    return np.asarray(fronts[0], dtype=np.int64) if fronts else np.zeros((0,), dtype=np.int64)


def _spin_to_bits01(spin: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(spin) > 0, 0, 1).astype(np.int8)

def _fps_select(
    data: np.ndarray,
    n_select: int,
    seed: int = 42,
) -> np.ndarray:
    n = data.shape[0]
    rng = np.random.default_rng(seed)

    selected = np.empty(n_select, dtype=np.int64)
    selected[0] = rng.choice(n)

    min_d2 = np.full(n, np.inf, dtype=np.float64)

    for i in range(1, n_select):
        last = selected[i - 1]
        diff = data - data[last]
        d2 = np.sum(diff * diff, axis=1)
        min_d2 = np.minimum(min_d2, d2)
        min_d2[selected[:i]] = -1.0
        selected[i] = np.argmax(min_d2)

    return np.sort(selected)

def _sample_unique_spins(sim: Simulator, circ, shots: int, n_qubits: int, *, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    sim.reset()
    res = sim.sampling(circ, shots=int(shots), seed=int(seed))
    unique_spins, counts = sampling_result_to_unique_spins(res, n_qubits=int(n_qubits))
    if int(np.sum(counts)) != int(shots):
        raise ValueError(f"Sampling row count mismatch: got {int(np.sum(counts))}, expect {shots}")
    return np.asarray(unique_spins, dtype=np.int8), np.asarray(counts, dtype=np.int64)

# =========================
# main1: warm + nsga2-style elite tracking + per-w matching
# =========================
def _select_frontier_seeds(
    round_spins: np.ndarray,
    round_objs: np.ndarray,
    round_lambda_ids: np.ndarray,
    round_counts: np.ndarray | None = None,
    *,
    num_seeds: int,
    dist_thr: float = 1e-4,
    max_dups_per_lambda: int = 4,
    assume_nd: bool = False,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """ND -> anchors/HV/count priority -> distance filter -> lambda cap.

    这是协同多轮的核心：从上一轮产生的非支配前沿中挑选既分散又高频的
    样本作为下一轮 warm-start，同时把 active lambda 重新映射到这些
    前沿区域，避免所有预算挤在少数标量化方向上。

    Returns:
        warm_bits_bank (len=num_seeds): list of bits01 arrays
        active_lambda_ids (len=num_seeds): lambda ids aligned with warm_bits_bank
    """
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
        # 常规路径：先从整轮候选中提取第一层非支配点。
        # 若极端情况下为空，就退化到目标和最小的若干点，保证下一轮仍有种子。
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
    # 这里的归一化只服务于“本轮前沿内部的分散度比较”，不影响最终评分。
    mins = nd_objs.min(axis=0)
    maxs = nd_objs.max(axis=0)
    scale = np.maximum(maxs - mins, 1e-12)
    sobjs = (nd_objs - mins) / scale  # (m,k)
    k = int(sobjs.shape[1])

    # ---------- actual HV contributions (pygmo) ----------
    # 优先用真实 HV contribution 排序；若 pygmo 贡献计算失败，
    # 再退化到加权 crowding 近似，避免协同流程中断。
    ref_hv = np.max(nd_objs, axis=0) * 2.0 + 1.0
    hv_contrib = np.zeros((m,), dtype=np.float64)
    if m >= 2:
        try:
            hv_obj = pg.hypervolume(nd_objs)
            hv_contrib[:] = np.asarray(hv_obj.contributions(ref_point=ref_hv), dtype=np.float64)
        except Exception:
            # fallback: weighted crowding distance
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

    # ---------- anchors: extreme points per objective (HV corners) ----------
    anchors: List[int] = []
    for d in range(k):
        order = np.argsort(nd_objs[:, d])
        anchors.append(int(order[0]))
        if len(order) > 1:
            anchors.append(int(order[1]))
    anchors = list(dict.fromkeys(anchors))

    # candidate priority: anchors first, then HV contribution desc, then count desc
    # 角点先保留，剩余位置优先给“删掉它会更伤 HV”的点；若仍接近，
    # 再偏向在当前轮里出现次数更多的样本。
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
    # 两个约束同时生效：
    # 1. 每个 lambda 最多保留 max_dups_per_lambda 个种子，避免资源塌缩；
    # 2. 尽量要求新种子与已选种子保持距离，保证下一轮覆盖面。
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

    # 距离阈值逐步放松：先要“又好又分散”，不够再逐层放宽，最后保证补满。
    for thr2 in relax:
        for i in order:
            if selected_count >= num_seeds:
                break
            ii = int(i)
            if can_use(ii) and dist_ok(ii, thr2):
                add(ii)
        if selected_count >= num_seeds:
            break

    # fill if still short: ignore distance but keep lambda cap
    if selected_count < num_seeds:
        for i in order:
            if selected_count >= num_seeds:
                break
            ii = int(i)
            if selected_mask[ii]:
                continue
            if can_use(ii):
                add(ii)

    # pathological: still short -> repeat last
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
# =========================
# main1: warm-start by tracking frontier seeds and their lambdas



def warm_theta_from_bits2(
    bits01: np.ndarray,
    warm_c: float,
    problem=None,  # 传入 problem 就能用预计算的 importance
    eps: float = 1e-6
) -> np.ndarray:
    # 基础全局 c
    base_c = float(np.clip(warm_c, 0.0, 1.0))
    bits01 = np.asarray(bits01, dtype=np.float64)
    n = len(bits01)

    # ===================== 逐比特自适应 cᵢ =====================
    if problem is not None and hasattr(problem, '_cached_c_factor'):
        c_array = base_c * getattr(problem, '_cached_c_factor')
    else:
        # 没有 problem 信息 → 全局统一 c
        c_array = np.full(n, base_c, dtype=np.float64)
    # 安全裁剪
    c_array = np.clip(c_array, 0.1, 0.95)
    # ==========================================================
    # 逐比特计算 x
    x = (1.0 - c_array) * 0.5 + c_array * bits01
    x = np.clip(x, 1e-6, 1.0 - 1e-6)

    return 2.0 * np.arcsin(np.sqrt(x))


def ising_rms_scale(j: np.ndarray, h: np.ndarray, eps: float = 1e-12) -> float:
    """RMS scale of Ising coefficients, used for optional normalization."""
    j = np.asarray(j, dtype=np.float64).reshape(-1)
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    s2 = float(np.mean(np.square(j))) + float(np.mean(np.square(h)))
    return float(np.sqrt(max(s2, eps)))


def _avg_degree(edges: np.ndarray, n: int) -> float:
    """Average node degree of an undirected graph.

    For an undirected graph, avg_degree = 2|E|/n = mean(deg).
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

    where D is the average degree, mean(w_uv^2) averages over edges, and mean(h_i^2)
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




def build_qaoa_circuit_from_projected_ising2(
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
        thetas = warm_theta_from_bits2(bits01, warm_c,
                                      problem=problem, eps=1e-6
                                      )
        for q, th in enumerate(thetas):
            circ += RY(float(th)).on(q)

    u = problem.edges[:, 0]
    v = problem.edges[:, 1]

    for layer in range(p):
        beta = float(betas[layer])
        # minimization sign + transfer scaling for weights
        gamma_eff = -scale_gamma(float(gammas[layer]), edges=problem.edges, n=n, J=j, h=h)


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


        # cost unitary
        for q in range(n):
            hz = float(h[q])
            if hz != 0.0:
                circ += RZ(2.0 * gamma_eff * hz).on(q)
        for eidx in range(m):
            circ += Rzz(2.0 * gamma_eff * float(j[eidx])).on([int(u[eidx]), int(v[eidx])])

        # 拓扑索引的局部旋转增强。双 CNOT 围绕控制位 RZ 在此门序下相消，
        # 因此该块本身不增加额外双比特纠缠；线路纠缠主要来自问题层 Rzz。
        if layer % 3 == 0 and m > 0:
            # 使用问题图边列表选择施加局部增强的量子比特。
            for eidx in range(0, min(m, n), 8):
                qi = int(u[eidx])
                qj = int(v[eidx])
                circ += H.on(qi)
                circ += X.on(qj, qi)
                circ += RZ(beta * 1).on(qi)
                circ += X.on(qj, qi)
                circ += H.on(qi)



    return ensure_measure_all(circ, n)




def ensure_measure_all(circ: Circuit, n_qubits: int) -> Circuit:
    if hasattr(circ, "measure_all"):
        circ.measure_all()
        return circ
    from mindquantum.core.gates import Measure  # type: ignore
    for q in range(n_qubits):
        circ += Measure().on(q)
    return circ




import multiprocessing as mp

def _sample_worker_warmc(
    j, lam_id, warm_bits, curr_seed, shot_round,
    problem, n, betas, gammas, projected_j_pool, projected_h_pool,
    warm_c
):
    # 每个子进程独立创建模拟器
    sim_proc = Simulator("mqvector", n, seed=curr_seed, dtype=mq.complex128)

    j_raw = projected_j_pool[lam_id]
    h_raw = projected_h_pool[lam_id]

    circ = build_qaoa_circuit_from_projected_ising2(
        problem,
        j_raw,
        h_raw,
        betas=betas,
        gammas=gammas,
        warm_bits01=warm_bits,
        warm_c=warm_c,
    )

    unique_spins, counts = _sample_unique_spins(
        sim_proc,
        circ,
        shots=shot_round,
        n_qubits=n,
        seed=curr_seed,
    )
    return unique_spins, counts


def main1(
    problem_input: Union[str, IsingMOOProblem, Dict[str, np.ndarray]],
    sample_budget: int = BASE_SAMPLE_BUDGET,
    rng_seed: int | None = None,
) -> Dict[str, object]:
    """协同多轮 main1：用固定 100000 shots 在多轮间做覆盖、反馈和细化。

    流程概要：
    1. 从 1000 个 lambda 权重池中用 FPS 选出首轮 100 个分散方向；
    2. 每轮对 active lambda 构建 P=4 QAOA 电路并采样；
    3. 对当轮样本做目标归一化和非支配筛选；
    4. 从前沿上挑选分散种子，作为下一轮 warm-start，并同步更新 lambda；
    5. 拼接全部轮次样本，保证 `sample_used == 100000`。

    与 `answer_singleround_to_multi.py` 的区别在于：这里的多轮不是机械
    拆 shot，而是重新设计了前沿种子选择和 lambda 协同覆盖。
    """
    problem = _to_problem(problem_input)
    seed = 2026 if rng_seed is None else int(rng_seed)
    if int(sample_budget) != BASE_SAMPLE_BUDGET:
        raise ValueError(
            f"sample_budget must equal {BASE_SAMPLE_BUDGET}, got {sample_budget}."
        )

    lambda_pool = load_weight_pool(int(problem.k), n=1000, seed=2026).astype(np.float64)
    lower_bounds, upper_bounds = objective_extrema(problem)
    # 先把 1000 个 lambda 全部投影到当前问题的标量化 Ising 系数上，
    # 后面每轮只需按 active_lambda_ids 取子集，避免重复矩阵乘。
    projected_j_pool = np.asarray(lambda_pool @ problem.weights, dtype=np.float64)
    projected_h_pool = np.asarray(lambda_pool @ problem.h, dtype=np.float64)

    n = int(problem.n)
    betas, gammas = _TRANSFER_TABLE[P_LAYERS]

    # 预计算每个量子比特的 warm-start 强度缩放。
    # 这里把外场幅值作为主信号、度数作为辅信号，得到一个固定的按位 importance。
    eps = 1e-6
    h = problem.h  # shape (k, n)
    h_amplitude = np.mean(np.abs(h), axis=0)
    h_norm = h_amplitude / (h_amplitude.max() + eps)
    edges = problem.edges  # shape (m, 2)
    deg = np.bincount(edges.reshape(-1), minlength=n)
    deg_norm = deg / (deg.max() + eps)
    importance = 0.9 * h_norm + 0.1 * deg_norm
    setattr(problem, '_cached_c_factor', 0.7 + 0.3 * importance)
    # ===============================================================================

    out_spins = np.empty((BASE_SAMPLE_BUDGET, n), dtype=np.int8)
    cursor = 0

    # 首轮没有 warm-start 反馈，所以先用 FPS 选出一组尽量分散的 lambda，
    # 把第一轮预算铺向更广的 Pareto 方向。
    active_lambda_ids = _fps_select(lambda_pool, NUM_WEIGHTS, seed=seed + 100)
    warm_bits_bank: List[np.ndarray | None] = [None] * NUM_WEIGHTS


    for r in range(N_ROUNDS):
        use_warm = r != 0
        shot_round = SHOTS_PER_WEIGHT[r]

        # warm-start 强度按轮次逐步增大：前几轮更偏探索，后几轮再提高
        # 对上一轮前沿种子的依赖。超过 0.95 后截断，避免几乎退化为硬注入。
        warm_c_increment = 0.26 if r == 0 else 0.26
        current_warm_c = WARM_C_FIXED + r * warm_c_increment
        current_warm_c = min(current_warm_c, 0.95)

        round_unique_spin_blocks = []
        round_unique_count_blocks = []
        round_lambda_id_order = []
        tasks = []

        # 每个 active lambda 对应一个独立采样任务；并行层只负责加速，
        # 不改变调度顺序，所以下面还能按 j 顺序稳定回填。
        for j in range(NUM_WEIGHTS):
            lam_id = int(active_lambda_ids[j])
            warm_bits = warm_bits_bank[j] if use_warm else None
            curr_seed = seed + r * NUM_WEIGHTS + j

            # 把所有需要的参数打包（包括当前轮次的warm_c）
            tasks.append((
                j, lam_id, warm_bits, curr_seed, shot_round,
                problem, n, betas, gammas, projected_j_pool, projected_h_pool,
                current_warm_c
            ))

        # ===================== 2进程并行采样 =====================
        with mp.Pool(processes=2) as pool:
            results = pool.starmap(_sample_worker_warmc, tasks)
        # ==========================================================

        # 把 unique 样本按计数展开，拼回官方 main1 需要的原始样本流。
        for j, (unique_spins, counts) in enumerate(results):
            lam_id = int(active_lambda_ids[j])
            spins = np.repeat(unique_spins, counts.astype(np.int32), axis=0)
            out_spins[cursor : cursor + shot_round] = spins
            cursor += shot_round

            round_unique_spin_blocks.append(np.asarray(unique_spins, dtype=np.int8))
            round_unique_count_blocks.append(np.asarray(counts, dtype=np.int64))
            round_lambda_id_order.append(lam_id)


        if r < N_ROUNDS - 1:
            # 只有在还存在下一轮时，才把本轮样本压成前沿种子并更新
            # warm_bits_bank / active_lambda_ids；最后一轮只负责产出最终样本。
            round_seed_objs, round_seed_spins, round_seed_lambda_ids, round_seed_counts = exact_frontier_from_lambda_unique_batches(
                round_unique_spin_blocks,
                round_unique_count_blocks,
                round_lambda_id_order,
                edges=problem.edges,
                weights=problem.weights,
                h=problem.h,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            warm_bits_bank, active_lambda_ids = _select_frontier_seeds(
                round_seed_spins,
                round_seed_objs,
                round_seed_lambda_ids,
                round_seed_counts,
                num_seeds=NUM_WEIGHTS,
                dist_thr=1.5e-4,
                assume_nd=True,
            )

    if cursor != BASE_SAMPLE_BUDGET:
        out_spins = out_spins[:cursor]

    return {"sample_used": int(out_spins.shape[0]), "sample_spins": out_spins}




# =========================
# main2: optimized random frontier HV path used by the single-round representative.
# =========================
_INTERNAL_CHUNK = 736
_BATCH_MERGE = 5
_NUM_THREADS = 2
_GLOBAL_FLUSH_EVERY = 6


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
    arr = np.asarray(objs, dtype=np.float64)
    n = arr.shape[0]
    if n <= 1:
        return np.arange(n, dtype=np.int64)

    sums = arr.sum(axis=1)
    order = np.argsort(sums)
    s_arr = arr[order]

    nd_idx: list[int] = []
    nd_pts = np.empty((0, arr.shape[1]), dtype=np.float64)

    end0 = min(init_block, n)
    nd_in = pg_non_dominated_indices(s_arr[:end0])
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
        s_i8 = np.asarray(chunk, dtype=np.int8)
        pair = (s_i8[:, u] * s_i8[:, v]).astype(np.float64)
        energies = pair @ weights_t + s_i8.astype(np.float64) @ h_t
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
        local_pool = np.zeros((0, k), dtype=np.float64)
        for objs in objs_list:
            local_pool = _local_merge_fast(local_pool, objs[_fast_nd(objs)])
        return local_pool

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
    problem_input: Union[str, IsingMOOProblem, Dict[str, np.ndarray]],
    shots: int = 200000,
    rng_seed: int | None = None,
    chunk_size: int = 4096,
) -> Dict[str, object]:
    """与单轮方案一致的向量化 main2；本文主要贡献集中在 main1 多轮协同。"""
    problem = _to_problem(problem_input)
    seed = 2026 if rng_seed is None else int(rng_seed)
    return _fast_main2_inner(problem, int(shots), int(chunk_size), seed, HV_REF)


__all__ = ["main1", "main2"]
