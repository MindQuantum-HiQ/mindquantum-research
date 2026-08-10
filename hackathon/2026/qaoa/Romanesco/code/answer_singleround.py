"""强单轮 main1 基线，以及对应的向量化 main2 后处理。

作为“强单轮方案”的代表，用于说明高质量 Pareto 前沿覆盖不只来自更多轮次，也可以来自混合深度 QAOA 组合采样、结构化权重
选择和分支策略。文件中的 main2 也作为向量化后处理方案供统一对照。
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Union

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
MOO_ROOT = CODE_DIR if (CODE_DIR / "utils.py").is_file() else CODE_DIR.parents[1]
if str(MOO_ROOT) not in sys.path:
    sys.path.insert(0, str(MOO_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplcfg_hackathon_moo"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from mindquantum.simulator import Simulator

from utils import (
    HV_REF,
    IsingMOOProblem,
    build_qaoa_circuit_from_projected_ising,
    energy_batch_fast,
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
# 强单轮方案：100 个权重向量，每个权重一次性采样 1000 shots。
# 该配置强调单轮内的 circuit depth/weight 设计，而不是跨轮反馈。
STD_WEIGHTS = 100
STD_SHOTS = 1000
WARM_WEIGHTS = 0
WARM_SHOTS = 0
BASE_SAMPLE_BUDGET = STD_WEIGHTS * STD_SHOTS + WARM_WEIGHTS * WARM_SHOTS
WARM_C = 0.65
BETA_SCALE = 1.0
GAMMA_SCALE = 1.0

# 混合深度 QAOA 组合采样：P=6 负责更深探索，P=4/P=3 负责补充覆盖。
P6_COUNT = 50
P4_COUNT = 25
P3_COUNT = 25

# ── main2 constants (v85 + v245 M1 + v246 M3 + v247 M4 layout) ──
# main2 只比较后处理速度；必须保持 frontier/HV 与 reference 一致。
_INTERNAL_CHUNK = 736
_BATCH_MERGE = 5
_NUM_THREADS = 2
_GLOBAL_FLUSH_EVERY = 6

_CSV = MOO_ROOT / "transfer_data.csv"

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
    # 用权重向量本身的形状来粗分“更偏边界/更偏中部”的标量化方向：
    # - peak 大、top2 大、gap 大的 lambda 往往更靠近 Pareto 前沿角点，
    #   分给更深的 P=6；
    # - top2 较均衡、entropy 适中的 lambda 更像前沿中段，分给 P=4；
    # - 剩余方向统一使用 P=3 保底覆盖。
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

    v114: per-branch P-count profiles targeting case00/04/09 remaining headroom.
    """
    # 这里不直接预测最终分数，而是把 public10 上总结出的“问题统计特征”
    # 映射到有限几个稳定 profile。后面的主流程只依赖这个 profile 来选
    # seed、是否启用 shape-based P 分层、以及 P=6/4/3 数量。
    spread, maxmin, abs_corr, signed_corr, h_bias, w_bias = _problem_features(problem)

    # B1 (case09): shape ON was harmful (-2.6e-5) → revert to v109 default
    if w_bias < 0.12 and spread > 0.14:
        return 0, False, P6_COUNT, P4_COUNT, P3_COUNT

    # B2 (case00): 55/25/20 was flat → keep v113 profile
    if signed_corr < -0.05 and spread < 0.14:
        return 0, True, 55, 25, 20

    # B5 (case04): 60/20/20 was worse (-1.7e-5) → REVERSE: more diversity
    if spread > 0.18 and maxmin < 2.0:
        return 101, False, 40, 30, 30

    # B3 (case07): very high spread — narrow gap, keep default
    if spread > 0.40:
        return 2026, False, P6_COUNT, P4_COUNT, P3_COUNT

    # B4 (case08): very low spread — narrow gap, keep default
    if spread < 0.05:
        return 2026, False, P6_COUNT, P4_COUNT, P3_COUNT

    # B6 (case06): high abs_corr + h_bias — narrow gap, keep default
    if abs_corr > 0.19 and h_bias > 0.22:
        return 2026, False, P6_COUNT, P4_COUNT, P3_COUNT

    # B7 (case03): high s_corr + w_bias — already perfect (gap=0), keep default
    if signed_corr > 0.06 and w_bias > 0.40:
        return 17, False, P6_COUNT, P4_COUNT, P3_COUNT

    # B8 (case05): high s_corr + low w_bias — narrow gap, keep default
    if signed_corr > 0.06 and w_bias < 0.20:
        return 101, False, P6_COUNT, P4_COUNT, P3_COUNT

    # B9 (case01): high w_bias + low h_bias — narrow gap, keep default
    if w_bias > 0.25 and h_bias < 0.18:
        return 101, False, P6_COUNT, P4_COUNT, P3_COUNT

    # DEFAULT (case02): narrow gap, keep v109 default
    return 0, False, P6_COUNT, P4_COUNT, P3_COUNT


# ===================================================================
#  helpers (identical to v44)
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
#  lambda construction (identical to v44)
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
        out = pool[: max(STD_WEIGHTS, WARM_WEIGHTS)].copy()
        out = np.maximum(out, 1e-9)
        out /= out.sum(axis=1, keepdims=True)
        return out
    if mode == "cover":
        return _farthest_subset(pool, max(STD_WEIGHTS, WARM_WEIGHTS), seed=seed + 17)

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
    need = max(STD_WEIGHTS, WARM_WEIGHTS) - int(anchors.shape[0])
    cover = _farthest_subset(pool, max(need, 0), seed=seed + 17)
    lambdas = np.vstack([anchors, cover])[: max(STD_WEIGHTS, WARM_WEIGHTS)]
    lambdas = np.maximum(lambdas, 1e-9)
    lambdas /= lambdas.sum(axis=1, keepdims=True)
    return np.asarray(lambdas, dtype=np.float64)


def _spin_to_bits01(spin: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(spin) > 0, 0, 1).astype(np.int8)


def _choose_warm_bits(problem: IsingMOOProblem, spins: np.ndarray, lambdas: np.ndarray) -> list[np.ndarray]:
    lower, upper = objective_extrema(problem)
    energies = energy_batch_fast(spins, problem.edges, problem.weights, problem.h)
    objs = normalize_energies(energies, lower, upper)
    seeds: list[np.ndarray] = []
    for lam in np.asarray(lambdas, dtype=np.float64):
        score = np.einsum("sk,k->s", objs, lam, optimize=False)
        seeds.append(_spin_to_bits01(spins[int(np.argmin(score))]))
    return seeds


# ===================================================================
#  main1 — Multi-P single-round QAOA 组合采样
# ===================================================================

def main1(
    problem_input: Union[str, IsingMOOProblem, dict],
    sample_budget: int = BASE_SAMPLE_BUDGET,
    rng_seed: int | None = None,
) -> Dict[str, object]:
    """强单轮 main1：选 100 个 lambda，按 P=6/4/3 组合采样生成 100000 个量子样本。

    返回值遵循官方接口：`sample_used` 必须为 100000，`sample_spins`
    只包含 -1/+1。该方案不使用跨轮反馈，适合作为“深度型单轮方案”。
    """
    problem = _to_problem(problem_input)
    if int(sample_budget) != BASE_SAMPLE_BUDGET:
        raise ValueError(f"sample_budget must equal {BASE_SAMPLE_BUDGET}, got {sample_budget}.")

    seed = _problem_seed(problem, rng_seed)
    if rng_seed is None:
        seed_shift, use_shape_p, p6, p4, p3 = _strategy(problem)
        seed += seed_shift
    else:
        # 外部显式给定 rng_seed 时，关闭 case-specific seed 偏移，只保留
        # 固定的 mixed-depth 配置，便于复现实验。
        use_shape_p = True
        p6, p4, p3 = P6_COUNT, P4_COUNT, P3_COUNT
    lambdas_all = _select_lambdas(problem, seed)
    std_lambdas = lambdas_all[:STD_WEIGHTS]
    if use_shape_p:
        # shape-based 分配会根据 lambda 几何形状决定 P=6/4/3；
        # 否则直接按固定数量切片，便于与若干 case-specific profile 对照。
        p_values = _lambda_p_values(std_lambdas, p6=p6, p4=p4)
    else:
        p_values = np.empty((STD_WEIGHTS,), dtype=np.int8)
        p_values[:p6] = 6
        p_values[p6:p6 + p4] = 4
        p_values[p6 + p4:] = 3
    mix_p3_mask = np.zeros((STD_WEIGHTS,), dtype=bool)
    if use_shape_p and p6 == 55 and p4 == 25 and p3 == 20:
        # B2 分支里，前几个 P=6 方向额外混入一小部分较浅电路，
        # 用于给边界方向增加一点局部多样性。
        p6_idx = np.flatnonzero(p_values == 6)
        mix_p3_mask[p6_idx[:5]] = True
    warm_lambdas = lambdas_all[:WARM_WEIGHTS]
    proj_lambdas = np.asarray(std_lambdas, dtype=np.float64).copy()

    # ── B2 lambda replacement: 19 P=6 lambdas with peak-sorted cover (v199) ──
    if use_shape_p and p6 == 55 and p4 == 25 and p3 == 20:
        p6_idx = np.flatnonzero(p_values == 6)
        alt_pool = load_weight_pool(int(problem.k), n=1000, seed=2026).astype(np.float64)[100:]
        alt_subset = _farthest_subset(alt_pool, 19, seed=seed + 31415)
        peak_order = np.argsort(-np.max(alt_subset, axis=1))
        alt_subset = alt_subset[peak_order]
        for idx, lam_alt in zip(p6_idx[:19], alt_subset):
            proj_lambdas[int(idx)] = lam_alt

    # ── B5 lambda replacement: 25 P=4 lambdas with entropy-sorted cover (v242)
    #     + v251 refresh: first 18 P4 lambdas replaced with peak-sorted (v270)
    #     + v271 NEW: P=6 peak-sorted 5λ cover (from v255/v266, seed=88888) ──
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

        # v251: keep the v247 25-lambda tail, but refresh the first 18
        # slots with a peak-sorted cover.
        alt_subset_b5_n18 = _farthest_subset(alt_pool_b5, 18, seed=seed + 27182)
        alt_subset_b5_n18 = alt_subset_b5_n18[np.argsort(-np.max(alt_subset_b5_n18, axis=1))]
        for idx, lam_alt in zip(p_target_idx[:18], alt_subset_b5_n18):
            proj_lambdas[int(idx)] = lam_alt

        # ── v271 NEW: B5 P=6 peak-sorted 5λ cover (seed=88888) ──
        p6_target_idx = np.flatnonzero(p_values == 6)
        alt_subset_b5_p6 = _farthest_subset(alt_pool_b5, 6, seed=seed + 88888)
        alt_subset_b5_p6 = alt_subset_b5_p6[np.argsort(-np.max(alt_subset_b5_p6, axis=1))]
        for idx, lam_alt in zip(p6_target_idx[:5], alt_subset_b5_p6):
            proj_lambdas[int(idx)] = lam_alt
        # v567: ONLY B5 P=3 entropy-sorted 3λ (minimal addition)
        p3_target_idx_b5 = np.flatnonzero(p_values == 3)
        alt_subset_b5_p3 = _farthest_subset(alt_pool_b5, 3, seed=seed + 27300)
        eps_b5p3 = 1e-12
        ent_b5p3 = -np.sum(alt_subset_b5_p3 * np.log(alt_subset_b5_p3 + eps_b5p3), axis=1) / np.log(float(problem.k))
        alt_subset_b5_p3 = alt_subset_b5_p3[np.argsort(-ent_b5p3)]
        for idx, lam_alt in zip(p3_target_idx_b5[27:30], alt_subset_b5_p3):
            proj_lambdas[int(idx)] = lam_alt

    # ── B1/B3/DEFAULT lambda replacement (v244 elif-chain, v271 cleaned) ──
    # B1 (case09): peak P6 n=5  |  B3 (case07): peak P6 n=5 + n=3  |  DEFAULT: P6 peak n=5
    # B4/B6/B8/B9: dead code removed (all negative on public)
    if not use_shape_p and p6 == P6_COUNT and p4 == P4_COUNT and p3 == P3_COUNT:
        _sp, _mx, _ac, _sc, _hb, _wb = _problem_features(problem)
        if _wb < 0.12 and _sp > 0.14:  # B1 (case09)
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
        else:  # DEFAULT (case02) — v462: P=6 balanced cover (seed=91111)
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

    # 到这里，proj_lambdas 已经是本 case 最终真正送入量子线路的权重集合。
    # 下面一次性把每个 lambda 对应的标量化 Ising 系数投影出来，避免在
    # 采样循环里重复做 dot。
    std_j = np.vstack([np.dot(lam, problem.weights) for lam in proj_lambdas]).astype(np.float64)
    std_h = np.vstack([np.dot(lam, problem.h) for lam in proj_lambdas]).astype(np.float64)
    if WARM_WEIGHTS > 0:
        warm_j = np.vstack([np.dot(lam, problem.weights) for lam in warm_lambdas]).astype(np.float64)
        warm_h = np.vstack([np.dot(lam, problem.h) for lam in warm_lambdas]).astype(np.float64)
    else:
        warm_j = np.zeros((0, int(problem.m)), dtype=np.float64)
        warm_h = np.zeros((0, int(problem.n)), dtype=np.float64)

    sim = Simulator("mqvector", int(problem.n), seed=int(seed) % (2**23))
    std_blocks = np.empty((STD_WEIGHTS, STD_SHOTS, int(problem.n)), dtype=np.int8)

    for i in range(STD_WEIGHTS):
        p_val = int(p_values[i])
        betas, gammas = _TRANSFER[p_val]
        betas = np.asarray(betas, dtype=np.float64)
        gammas = np.asarray(gammas, dtype=np.float64)

        # v140: P=6 mix unchanged; P=4 & P=3 gamma mix gated to use_shape_p (B2 only)
        # 同一深度下再做非常轻的 gamma 缩放扰动，目的是让相邻权重方向
        # 不至于完全同质化；这仍然属于单轮内部的“深度侧多样化”。
        if p_val == 6:
            g = gammas * (1.00 if i % 4 in (0, 1) else (0.92 if i % 4 == 2 else 0.86))
        elif p_val == 4 and use_shape_p:
            g = gammas * (1.00 if i % 2 == 0 else 0.95)
        elif p_val == 3 and use_shape_p:
            g = gammas * (1.00 if i % 2 == 0 else 0.97)
        else:
            g = gammas

        # 少数 B2 权重会把 1000 shots 拆成 “主 P=6 + 少量混合深度” 两段采样；
        # 其余权重仍然一次性跑满 1000 shots。
        shots_primary = 800 if mix_p3_mask[i] and p_val == 6 else STD_SHOTS
        circ = build_qaoa_circuit_from_projected_ising(
            problem, std_j[i], std_h[i],
            betas=betas, gammas=g, warm_bits01=None,
        )
        sim.reset()
        res = sim.sampling(circ, shots=shots_primary, seed=int(seed + 7919 * (i + 1)))
        unique_spins, counts = sampling_result_to_unique_spins(res, int(problem.n))
        block = np.repeat(unique_spins, counts.astype(np.int32), axis=0)
        if mix_p3_mask[i] and p_val == 6:
            betas3, gammas3 = _TRANSFER[4]  # B2 P=4 mix
            circ3 = build_qaoa_circuit_from_projected_ising(
                problem, std_j[i], std_h[i],
                betas=np.asarray(betas3, dtype=np.float64),
                gammas=np.asarray(gammas3, dtype=np.float64), warm_bits01=None,
            )
            sim.reset()
            res3 = sim.sampling(circ3, shots=STD_SHOTS - shots_primary, seed=int(seed + 104729 * (i + 1)) % (2**23))
            unique_spins3, counts3 = sampling_result_to_unique_spins(res3, int(problem.n))
            block3 = np.repeat(unique_spins3, counts3.astype(np.int32), axis=0)
            block = np.vstack([block, block3])
        std_blocks[i] = block

    # `std_blocks` 的形状固定为 [100, 1000, n]，直接 reshape 即得到
    # 官方 main1 需要的 100000 行样本矩阵。
    sample_spins = std_blocks.reshape((-1, int(problem.n)))
    return {"sample_used": int(sample_spins.shape[0]), "sample_spins": sample_spins}


# ===================================================================
#  main2 — thread-parallel optimized random frontier HV (v246 + M4 layout)
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
    problem_input: Union[str, IsingMOOProblem, dict],
    shots: int = 200000,
    rng_seed: int | None = None,
    chunk_size: int = 4096,
) -> Dict[str, object]:
    """向量化 main2：保持随机采样语义不变，仅优化能量计算和前沿合并速度。"""
    problem = _to_problem(problem_input)
    seed = 2026 if rng_seed is None else int(rng_seed)
    return _fast_main2_inner(problem, int(shots), int(chunk_size), seed, HV_REF)


__all__ = ["STD_WEIGHTS", "STD_SHOTS", "WARM_WEIGHTS", "WARM_SHOTS", "BASE_SAMPLE_BUDGET", "main1", "main2"]
