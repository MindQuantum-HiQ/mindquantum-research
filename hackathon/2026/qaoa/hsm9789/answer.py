from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Union
import time

import numpy as np
from mindquantum.simulator import Simulator

from utils import (
    HV_REF,
    IsingMOOProblem,
    build_qaoa_circuit_from_projected_ising,
    load_transfer_params_csv,
    objective_extrema,
    problem_from_npz,
    sampling_result_to_unique_spins,
    pg_non_dominated_indices,
    lexsort_rows,
    hypervolume_pygmo,
    energy_batch_fast,
    normalize_energies,
)

# 限制底层数值库线程数，避免评测环境中多线程竞争导致运行不稳定。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplcfg_hackathon_moo")
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# main1 固定采样方案：第一轮 210 个 Simplex Lattice 权重，每个 300 shots；
# 第二轮根据第一轮 ND 指标筛选 74 个权重，并用指数衰减分配剩余 shots。
BASE_SAMPLE_BUDGET = 100000
SIMPLEX_H = 6
FIRST_ROUND_WEIGHTS = 210
SECOND_ROUND_WEIGHTS = 74
FIRST_ROUND_SHOTS_PER_WEIGHT = 300
P_LAYERS = [5, 4]
TRANSFER_Q_TARGET = 2
TRANSFER_CSV_PATH = Path(__file__).resolve().parent / "transfer_data.csv"
_TRANSFER_TABLE = load_transfer_params_csv(
    str(TRANSFER_CSV_PATH), q_target=TRANSFER_Q_TARGET, p_list=P_LAYERS
)


def _seed_from_problem(problem: IsingMOOProblem) -> int:
    """根据实例系数生成稳定随机种子，使 main2 在同一实例上可复现。"""
    digest = hashlib.sha1()
    digest.update(np.ascontiguousarray(problem.weights).view(np.uint8))
    digest.update(np.ascontiguousarray(problem.h).view(np.uint8))
    return int(digest.hexdigest()[:16], 16)


def _to_problem(x: Union[str, IsingMOOProblem, Dict[str, np.ndarray]]) -> IsingMOOProblem:
    """统一入口格式，兼容路径、IsingMOOProblem 对象和字典输入。"""
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


def _sample_unique_spins(
    sim: Simulator,
    circ,
    shots: int,
    n_qubits: int,
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """执行一次 QAOA 采样，并返回去重后的自旋样本及其出现次数。"""
    sim.reset()
    result = sim.sampling(circ, shots=int(shots), seed=int(seed))
    unique_spins, counts = sampling_result_to_unique_spins(result, n_qubits=int(n_qubits))
    if int(np.sum(counts)) != int(shots):
        raise ValueError(
            f"Sampling row count mismatch: got {int(np.sum(counts))}, expect {shots}"
        )
    return np.asarray(unique_spins, dtype=np.int8), np.asarray(counts, dtype=np.int64)


def make_simplex_lattice_bank(
    k: int = 5,
    H: int = SIMPLEX_H,
    total: int = FIRST_ROUND_WEIGHTS,
    seed: int = 226,
    eps: float = 1e-12,
) -> np.ndarray:
    """生成 Simplex Lattice 权重集合。

    对 k 目标问题，枚举所有非负整数向量 a，使得 sum(a)=H，
    并令 lambda=a/H。若 total 小于完整 lattice 大小，则按支撑集
    大小分层抽样；在本方案中 H=6、k=5、total=210，正好使用完整集合。
    """
    if H <= 0:
        raise ValueError("H must be positive")
    if total <= 0:
        raise ValueError("total must be positive")

    rng = np.random.default_rng(seed)
    compositions: List[List[int]] = []

    def rec(remaining: int, dim: int, prefix: List[int]) -> None:
        if dim == k - 1:
            compositions.append(prefix + [remaining])
            return
        for value in range(remaining + 1):
            rec(remaining - value, dim + 1, prefix + [value])

    rec(H, 0, [])
    lattice = np.asarray(compositions, dtype=np.float64) / float(H)
    lattice_size = int(lattice.shape[0])
    if total > lattice_size:
        raise ValueError(f"total={total} exceeds lattice size={lattice_size}")

    support_size = np.count_nonzero(lattice > 0.0, axis=1)
    groups: Dict[int, np.ndarray] = {
        s: np.where(support_size == s)[0] for s in range(1, k + 1)
    }

    raw_counts = {s: len(idx) for s, idx in groups.items() if len(idx) > 0}
    ideal = {s: total * raw_counts[s] / lattice_size for s in raw_counts}
    take = {s: int(np.floor(ideal[s])) for s in raw_counts}

    remaining = total - sum(take.values())
    frac_order = sorted(raw_counts, key=lambda s: ideal[s] - take[s], reverse=True)
    for s in frac_order:
        if remaining <= 0:
            break
        if take[s] < raw_counts[s]:
            take[s] += 1
            remaining -= 1

    while remaining > 0:
        candidates = [s for s in raw_counts if take[s] < raw_counts[s]]
        if not candidates:
            raise RuntimeError("Failed to allocate lattice subset")
        s = max(candidates, key=lambda x: raw_counts[x] - take[x])
        take[s] += 1
        remaining -= 1

    chosen_indices: List[int] = []
    for s in range(1, k + 1):
        n_take = take.get(s, 0)
        if n_take > 0:
            chosen = rng.choice(groups[s], size=n_take, replace=False)
            chosen_indices.extend(chosen.tolist())

    rng.shuffle(chosen_indices)
    lambdas = lattice[chosen_indices]

    # 极小 eps 用于避免个别后续实现对严格零权重处理不稳。
    lambdas = np.maximum(lambdas, eps)
    lambdas = lambdas / lambdas.sum(axis=1, keepdims=True)
    if lambdas.shape != (total, k):
        raise AssertionError(f"lambdas shape={lambdas.shape}, expected={(total, k)}")
    return np.asarray(lambdas, dtype=np.float64)


def count_direction_nd(
    unique_spin_blocks: List[np.ndarray],
    edges: np.ndarray,
    weights: np.ndarray,
    h: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """计算每个权重方向在第一轮采样中产生的方向内 ND 解数量。"""
    nd_counts: List[int] = []
    for unique_spins in unique_spin_blocks:
        energies = energy_batch_fast(unique_spins, edges, weights, h)
        objectives = normalize_energies(energies, lower, upper)
        nd_idx = pg_non_dominated_indices(objectives)
        nd_counts.append(int(nd_idx.size))
    return np.asarray(nd_counts, dtype=np.int64)


def allocate_shots_exp(
    num_weights: int,
    total_shots: int,
    min_shots: int,
    max_shots: int,
    decay: float = 4.0,
) -> List[int]:
    """按排名指数衰减分配第二轮 shots。

    排名越靠前的权重获得越多 shots；所有权重至少获得 min_shots，
    至多获得 max_shots。函数会修正取整误差，确保总 shots 精确等于 total_shots。
    """
    n = int(num_weights)
    total = int(total_shots)
    smin = int(min_shots)
    smax = int(max_shots)

    if n <= 0:
        return []
    if total < n * smin:
        raise ValueError(f"total_shots={total} < n*min_shots={n * smin}")
    if total > n * smax:
        raise ValueError(f"total_shots={total} > n*max_shots={n * smax}")

    shots = np.full(n, smin, dtype=np.int64)
    remaining = total - n * smin
    if remaining == 0:
        return shots.tolist()

    rank = np.arange(n, dtype=np.float64)
    weights = np.exp(-float(decay) * rank / max(n - 1, 1))
    weights /= weights.sum()
    shots += np.floor(remaining * weights).astype(np.int64)
    shots = np.minimum(shots, smax)

    deficit = total - int(shots.sum())
    while deficit > 0:
        updated = False
        for i in range(n):
            if shots[i] < smax:
                shots[i] += 1
                deficit -= 1
                updated = True
                if deficit == 0:
                    break
        if not updated:
            break

    return np.sort(shots)[::-1].tolist()


def main1(
    problem_input: Union[str, IsingMOOProblem, Dict[str, np.ndarray]],
    sample_budget: int = BASE_SAMPLE_BUDGET,
    rng_seed: int | None = None,
) -> Dict[str, object]:
    """小规模多目标 Ising 问题的两阶段 QAOA 采样入口。"""
    problem = _to_problem(problem_input)
    seed = 2026 if rng_seed is None else int(rng_seed)
    n_qubits = int(problem.n)

    lower_bounds, upper_bounds = objective_extrema(problem)
    objective_weights = problem.weights
    objective_fields = problem.h

    lambda_pool = make_simplex_lattice_bank(
        k=int(problem.k), H=SIMPLEX_H, total=FIRST_ROUND_WEIGHTS, seed=226
    )
    projected_j_pool = np.asarray(lambda_pool @ objective_weights, dtype=np.float64)
    projected_h_pool = np.asarray(lambda_pool @ objective_fields, dtype=np.float64)

    sim = Simulator("mqvector", n_qubits, seed=int(seed))
    out_spins = np.empty((int(sample_budget), n_qubits), dtype=np.int8)
    cursor = 0

    first_round_shots = [FIRST_ROUND_SHOTS_PER_WEIGHT] * FIRST_ROUND_WEIGHTS
    second_round_budget = int(sample_budget) - FIRST_ROUND_WEIGHTS * FIRST_ROUND_SHOTS_PER_WEIGHT
    if second_round_budget <= 0:
        raise ValueError("sample_budget is too small for the fixed first-round plan")

    shots_by_round: List[List[int]] = [first_round_shots]
    weights_by_round = [FIRST_ROUND_WEIGHTS, SECOND_ROUND_WEIGHTS]

    for round_id, num_weights in enumerate(weights_by_round):
        betas, gammas = _TRANSFER_TABLE[P_LAYERS[round_id]]
        shot_plan = shots_by_round[round_id]

        round_unique_spin_blocks: List[np.ndarray] = []
        round_lambda_ids: List[int] = []

        for j in range(num_weights):
            j_raw = projected_j_pool[j]
            h_raw = projected_h_pool[j]
            circ = build_qaoa_circuit_from_projected_ising(
                problem,
                j_raw,
                h_raw,
                betas=betas,
                gammas=gammas,
                warm_bits01=None,
                warm_c=0,
            )
            shots = int(shot_plan[j])
            unique_spins, counts = _sample_unique_spins(
                sim,
                circ,
                shots=shots,
                n_qubits=n_qubits,
                seed=seed + round_id * num_weights + j,
            )
            expanded_spins = np.repeat(unique_spins, counts.astype(np.int32), axis=0)
            out_spins[cursor : cursor + shots] = expanded_spins
            cursor += shots

            round_unique_spin_blocks.append(unique_spins)
            round_lambda_ids.append(j)

        # 第一轮结束后，根据方向内 ND 数量筛选第二轮权重，并生成第二轮 shots 计划。
        if round_id == 0:
            nd_counts = count_direction_nd(
                round_unique_spin_blocks,
                problem.edges,
                problem.weights,
                problem.h,
                lower_bounds,
                upper_bounds,
            )
            sorted_pairs = sorted(
                zip(round_lambda_ids, nd_counts), key=lambda item: item[1], reverse=True
            )
            selected_indices = [idx for idx, _ in sorted_pairs[:SECOND_ROUND_WEIGHTS]]
            lambda_pool = np.asarray(lambda_pool[selected_indices], dtype=np.float64)
            projected_j_pool = np.asarray(lambda_pool @ objective_weights, dtype=np.float64)
            projected_h_pool = np.asarray(lambda_pool @ objective_fields, dtype=np.float64)
            shots_by_round.append(
                allocate_shots_exp(
                    SECOND_ROUND_WEIGHTS,
                    second_round_budget,
                    min_shots=200,
                    max_shots=800,
                    decay=4.0,
                )
            )

    if cursor != int(sample_budget):
        raise RuntimeError(f"sample count mismatch: got {cursor}, expect {sample_budget}")
    return {"sample_used": int(cursor), "sample_spins": out_spins}


def _filter_not_dominated_by_pool(objs: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """删除已被当前全局前沿支配的候选目标向量。"""
    if pool is None or pool.size == 0 or objs.size == 0:
        return objs
    keep = np.ones((objs.shape[0],), dtype=bool)
    for point in pool:
        idx = np.nonzero(keep)[0]
        if idx.size == 0:
            break
        dominated = np.all(objs[idx] >= point, axis=1)
        if np.any(dominated):
            keep[idx[dominated]] = False
    return objs[keep]


def _merge_two_nd_pools_cross(pool: np.ndarray, local: np.ndarray) -> np.ndarray:
    """合并两个各自已经非支配的前沿集合，只检查集合之间的交叉支配关系。"""
    if pool is None or pool.size == 0:
        return np.asarray(local, dtype=np.float64)
    if local is None or local.size == 0:
        return np.asarray(pool, dtype=np.float64)

    pool = np.ascontiguousarray(pool, dtype=np.float64)
    local = np.ascontiguousarray(local, dtype=np.float64)
    n_pool = int(pool.shape[0])
    n_local = int(local.shape[0])

    if n_pool * n_local <= 8192:
        keep_local = np.ones((n_local,), dtype=bool)
        for point in pool:
            idx = np.nonzero(keep_local)[0]
            if idx.size == 0:
                break
            dominated = np.all(local[idx] >= point, axis=1)
            if np.any(dominated):
                keep_local[idx[dominated]] = False
        local = local[keep_local]
        if local.size == 0:
            return pool

        keep_pool = np.ones((n_pool,), dtype=bool)
        for point in local:
            idx = np.nonzero(keep_pool)[0]
            if idx.size == 0:
                break
            dominated = np.all(pool[idx] >= point, axis=1)
            if np.any(dominated):
                keep_pool[idx[dominated]] = False
        return np.vstack([pool[keep_pool], local])

    block = 384
    keep_local = np.ones((n_local,), dtype=bool)
    for start in range(0, n_local, block):
        end = min(start + block, n_local)
        blk = local[start:end]
        dominated = np.any(np.all(blk[:, None, :] >= pool[None, :, :], axis=2), axis=1)
        keep_local[start:end] = ~dominated
    local = local[keep_local]
    if local.size == 0:
        return pool

    n_pool = int(pool.shape[0])
    keep_pool = np.ones((n_pool,), dtype=bool)
    for start in range(0, n_pool, block):
        end = min(start + block, n_pool)
        blk = pool[start:end]
        dominated = np.any(np.all(blk[:, None, :] >= local[None, :, :], axis=2), axis=1)
        keep_pool[start:end] = ~dominated

    return np.vstack([pool[keep_pool], local])


def _large_random_frontier_hv_bool_opt(
    problem: IsingMOOProblem,
    *,
    shots: int = 200000,
    chunk_size: int = 4096,
    rng_seed: int = 2026,
    ref: float = HV_REF,
) -> Dict[str, object]:
    """main2 的随机样本前沿计算与 HV 评估。

    该实现首先尝试识别规则二维格点结构，并在满足条件时使用切片方式
    批量计算水平边和垂直边贡献；随后对候选目标向量进行分块预筛选、
    局部 ND 筛选和全局前沿交叉合并。
    """

    rng = np.random.default_rng(int(rng_seed))
    lower_bounds, upper_bounds = objective_extrema(problem)
    lo = np.asarray(lower_bounds, dtype=np.float64)
    hi = np.asarray(upper_bounds, dtype=np.float64)
    span = np.maximum(hi - lo, 1e-12)

    weights_scaled = np.ascontiguousarray(problem.weights / span[:, None], dtype=np.float64)
    h_scaled = np.ascontiguousarray(problem.h / span[:, None], dtype=np.float64)
    const = (-(np.sum(problem.weights, axis=1) + np.sum(problem.h, axis=1)) - lo) / span
    field_matrix = np.ascontiguousarray(h_scaled.T, dtype=np.float64)

    edges = np.asarray(problem.edges, dtype=np.int64)
    u = np.ascontiguousarray(edges[:, 0])
    v = np.ascontiguousarray(edges[:, 1])
    k = int(problem.k)
    n = int(problem.n)
    a = int(problem.a)
    bdim = int(problem.b)

    delta = v - u
    horizontal_mask = delta == 1
    vertical_mask = delta == bdim
    use_grid_fast = (
        a * bdim == n
        and int(np.count_nonzero(horizontal_mask)) == a * (bdim - 1)
        and int(np.count_nonzero(vertical_mask)) == (a - 1) * bdim
        and int(np.count_nonzero(horizontal_mask) + np.count_nonzero(vertical_mask))
        == int(edges.shape[0])
    )
    # 显式初始化分支专用矩阵，避免静态检查器报告“赋值前引用”。
    weight_horizontal: np.ndarray | None = None
    weight_vertical: np.ndarray | None = None
    edge_weight_matrix: np.ndarray | None = None

    if use_grid_fast:
        weight_horizontal = np.ascontiguousarray(weights_scaled[:, horizontal_mask].T)
        weight_vertical = np.ascontiguousarray(weights_scaled[:, vertical_mask].T)
    else:
        edge_weight_matrix = np.ascontiguousarray(weights_scaled.T)

    remaining = int(shots)
    nd_pool = np.zeros((0, k), dtype=np.float64)
    n_points = 0
    t0 = time.perf_counter()

    while remaining > 0:
        batch_size = min(int(chunk_size), remaining)
        bits = rng.random((batch_size, n)) < 0.5

        objs = bits @ field_matrix
        if use_grid_fast:
            assert weight_horizontal is not None and weight_vertical is not None
            grid = bits.reshape(batch_size, a, bdim)
            eq_h = (grid[:, :, :-1] == grid[:, :, 1:]).reshape(batch_size, -1)
            eq_v = (grid[:, :-1, :] == grid[:, 1:, :]).reshape(batch_size, -1)
            objs += eq_h @ weight_horizontal
            objs += eq_v @ weight_vertical
        else:
            assert edge_weight_matrix is not None
            edge_equal = bits[:, u] == bits[:, v]
            objs += edge_equal @ edge_weight_matrix

        objs *= 2.0
        objs += const[None, :]
        objs = np.asarray(objs, dtype=np.float64)

        if nd_pool.size:
            prefilter_pool = nd_pool
            if prefilter_pool.shape[0] > 128:
                idx = np.argpartition(np.sum(prefilter_pool, axis=1), 128)[:128]
                prefilter_pool = prefilter_pool[idx]
            objs = _filter_not_dominated_by_pool(objs, prefilter_pool)

        if objs.size:
            local_front = objs[pg_non_dominated_indices(objs)]
            nd_pool = _merge_two_nd_pools_cross(nd_pool, local_front)

        n_points += batch_size
        remaining -= batch_size

    elapsed = time.perf_counter() - t0
    nd_pool = np.asarray(lexsort_rows(nd_pool), dtype=np.float64)
    hv = float(hypervolume_pygmo(nd_pool, ref=ref))
    return {
        "shots": int(shots),
        "chunk_size": int(chunk_size),
        "n_points": int(n_points),
        "nd_count": int(nd_pool.shape[0]),
        "hv": hv,
        "frontier_objectives_norm": nd_pool.tolist(),
        "elapsed_s": float(elapsed),
    }


def main2(
    problem_input: Union[str, IsingMOOProblem, Dict[str, np.ndarray]],
    shots: int = 200000,
    rng_seed: int | None = None,
    chunk_size: int = 4096,
) -> Dict[str, object]:
    """大规模样本经典后处理入口。"""
    # 2304 是实验中较稳定的分块大小，保留参数入口以兼容评测函数签名。
    del chunk_size
    problem = _to_problem(problem_input)
    seed = (_seed_from_problem(problem) + 701) if rng_seed is None else int(rng_seed)
    return _large_random_frontier_hv_bool_opt(
        problem,
        shots=int(shots),
        chunk_size=2304,
        rng_seed=seed,
        ref=HV_REF,
    )


__all__ = ["main1", "main2"]
