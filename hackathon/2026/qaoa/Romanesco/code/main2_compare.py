"""
后处理三方案统一对照入口

官方参考实现、向量化后处理方案、内联 Numba 热路径 refinement。
该脚本用于论文中的 main2 加速对照：先运行官方 baseline 得到一致性
基准，再验证候选方案是否保持 HV/frontier 完全一致，只有一致时才计入
速度提升。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numba import njit

CODE_DIR = Path(__file__).resolve().parent
PAPER_DIR = CODE_DIR.parent
ROOT = PAPER_DIR.parent
RESULTS_DIR = PAPER_DIR / "results"
DEFAULT_LARGE_DIR = ROOT / "data" / "large"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import baseline as baseline_module
from run import HV_REF, _hv_from_nd_objs, _large_eval_case_main2, _nd_idx_fast
from utils import (
    IsingMOOProblem,
    hypervolume_pygmo,
    lexsort_rows,
    merge_non_dominated_pool,
    objective_extrema,
    problem_from_npz,
)


def _load_module(name: str, path: Path):
    """按文件路径加载方案模块，避免受当前工作目录和同名模块影响。"""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_OFFICIAL = _load_module("moo_official_reference_main2", ROOT / "answer.py")
_VECTORIZED = _load_module("moo_vectorized_main2", CODE_DIR / "answer_singleround.py")


def main2(
    problem_input: str | IsingMOOProblem | dict,
    shots: int = 200000,
    rng_seed: int | None = None,
    chunk_size: int = 4096,
):
    """官方 answer.py 的 main2 包装，用作 reference 方案。"""
    return _OFFICIAL.main2(
        problem_input=problem_input,
        shots=int(shots),
        rng_seed=rng_seed,
        chunk_size=int(chunk_size),
    )


def main2_vectorized(
    problem_input: str | IsingMOOProblem | dict,
    shots: int = 200000,
    rng_seed: int | None = None,
    chunk_size: int = 4096,
):
    """强单轮文件中的向量化 main2 包装，用作已有优化方案。"""
    return _VECTORIZED.main2(
        problem_input=problem_input,
        shots=int(shots),
        rng_seed=rng_seed,
        chunk_size=int(chunk_size),
    )


# =========================
# main2_numba：内联的 Numba 热路径方案
# =========================
# 这个区块来自原独立 Numba refinement 的 main2 部分。保留在三方案对照
# 文件中，是为了让 main2 的三个方案在一个文件里完成统一展示、调用和验证。
_INTERNAL_CHUNK = 736
_BATCH_MERGE = 5
_NUM_THREADS = 2
_GLOBAL_FLUSH_EVERY = 6
_SPIN_CACHE: dict[int, np.ndarray] = {}


def _to_problem(problem_input: str | IsingMOOProblem | dict) -> IsingMOOProblem:
    """将路径、问题对象或字典统一转换为赛题问题对象。"""
    if isinstance(problem_input, IsingMOOProblem):
        return problem_input
    if isinstance(problem_input, str):
        return problem_from_npz(problem_input)
    if isinstance(problem_input, dict):
        return IsingMOOProblem(**problem_input)
    raise TypeError(f"unsupported problem input: {type(problem_input)!r}")


def _local_merge_fast(pool: np.ndarray, new_points: np.ndarray) -> np.ndarray:
    """合并局部候选点，并立刻用 Numba 非支配筛选压缩前沿规模。"""
    a = np.asarray(pool, dtype=np.float64)
    b = np.asarray(new_points, dtype=np.float64)
    if b.size == 0:
        return a
    merged = b if a.size == 0 else np.vstack([a, b])
    if merged.shape[0] > 1:
        merged = np.unique(merged, axis=0)
    return merged[_fast_nd_numba(merged)]


@njit(cache=True)
def _fast_nd_sorted_indices_numba(sorted_arr: np.ndarray) -> np.ndarray:
    """在已排序数组上计算第一层非支配点索引。"""
    n, k = sorted_arr.shape
    survivors = np.empty(n, dtype=np.int64)
    survivor_count = 0
    for i in range(n):
        dominated = False
        for pos in range(survivor_count):
            j = survivors[pos]
            better_or_equal = True
            strictly_better = False
            for d in range(k):
                lhs = sorted_arr[j, d]
                rhs = sorted_arr[i, d]
                if lhs > rhs:
                    better_or_equal = False
                    break
                if lhs < rhs:
                    strictly_better = True
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            survivors[survivor_count] = i
            survivor_count += 1
    return survivors[:survivor_count]


def _fast_nd_numba(objs: np.ndarray) -> np.ndarray:
    """Numba 版本的非支配筛选入口，按目标和预排序以减少比较次数。"""
    arr = np.asarray(objs, dtype=np.float64)
    n = arr.shape[0]
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    order = np.argsort(np.sum(arr, axis=1), kind="mergesort")
    sorted_arr = arr[order]
    keep_sorted = _fast_nd_sorted_indices_numba(sorted_arr)
    return order[np.asarray(keep_sorted, dtype=np.int64)]


def _cached_spins(n_qubits: int, total_shots: int, chunk_size: int, seed: int) -> np.ndarray:
    """缓存同一随机种子下的 spin 样本，避免三方案对照时重复生成。"""
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
            rng.random((bs, n_qubits)) < 0.5,
            np.int8(1),
            np.int8(-1),
        )
        offset += bs
        remaining -= bs
    _SPIN_CACHE[cache_key] = spins
    return spins


def _fast_main2_inner(
    problem: IsingMOOProblem,
    shots: int,
    chunk_size: int,
    rng_seed: int,
    ref: float,
) -> dict[str, object]:
    """执行 Numba refinement 方案：随机采样、能量计算、非支配筛选和 HV 计算。"""
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

    def _energy_raw(start_end: tuple[int, int]) -> np.ndarray:
        start, end = start_end
        chunk = all_spins[start:end]
        s_i8 = np.asarray(chunk, dtype=np.int8)
        pair = (s_i8[:, u] * s_i8[:, v]).astype(np.float64)
        return pair @ weights_t + s_i8.astype(np.float64) @ h_t

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
            local_pool = _local_merge_fast(local_pool, objs[_fast_nd_numba(objs)])
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


def main2_numba(
    problem_input: str | IsingMOOProblem | dict,
    shots: int = 200000,
    rng_seed: int | None = None,
    chunk_size: int = 4096,
):
    """Numba 热路径 main2 包装，用作本文最快后处理候选方案。"""
    problem = _to_problem(problem_input)
    seed = 2026 if rng_seed is None else int(rng_seed)
    return _fast_main2_inner(problem, int(shots), int(chunk_size), seed, HV_REF)


# Backward-compatible alias for older local notebooks/scripts.
main2_my = main2_vectorized


def _evaluate_baseline(files: list[Path], shots: int, chunk_size: int, seed: int):
    """逐 case 运行官方 baseline，并保留前沿供候选方案一致性比较。"""
    rows = []
    by_case = {}
    for index, path in enumerate(files, start=1):
        problem = problem_from_npz(str(path))
        t0 = time.time()
        base = _large_eval_case_main2(
            baseline_module.main2,
            problem=problem,
            shots=shots,
            chunk_size=chunk_size,
            rng_seed=seed,
        )
        row = {
            "case": path.name,
            "elapsed_s": float(base["elapsed_s"]),
            "hv": float(base["hv"]),
            "nd_count": int(base["nd_count"]),
            "wall_s": float(time.time() - t0),
            "shots": int(shots),
        }
        rows.append(row)
        by_case[path.name] = base
        print(
            f"[baseline] {index:02d}/{len(files)} {path.name} "
            f"hv={row['hv']:.12f} nd={row['nd_count']} elapsed={row['elapsed_s']:.4f}s",
            flush=True,
        )
    return {
        "score_large_bonus_raw": 0.0,
        "score_large_bonus": 0.0,
        "avg_elapsed_s": float(sum(r["elapsed_s"] for r in rows) / len(rows)) if rows else 0.0,
        "total_elapsed_s": float(sum(r["elapsed_s"] for r in rows)),
        "rows": rows,
        "_by_case": by_case,
    }


def _evaluate_variant(
    label: str,
    fn,
    files: list[Path],
    baseline_by_case: dict[str, dict[str, object]],
    shots: int,
    chunk_size: int,
    seed: int,
):
    """评估一个候选 main2：先验前沿一致，再计算有效 speedup。"""
    rows = []
    speedups = []
    for index, path in enumerate(files, start=1):
        problem = problem_from_npz(str(path))
        base = baseline_by_case[path.name]
        t0 = time.time()
        try:
            cand = _large_eval_case_main2(
                fn,
                problem=problem,
                shots=shots,
                chunk_size=chunk_size,
                rng_seed=seed,
            )
        except Exception as exc:  # keep one failed scheme from stopping the whole comparison
            wall_s = float(time.time() - t0)
            speedups.append(0.0)
            rows.append(
                {
                    "case": path.name,
                    "baseline_s": float(base["elapsed_s"]),
                    "candidate_s": None,
                    "candidate_wall_s": float(wall_s),
                    "baseline_hv": float(base["hv"]),
                    "candidate_hv": None,
                    "baseline_nd_count": int(base["nd_count"]),
                    "candidate_nd_count": None,
                    "hv_abs_diff": None,
                    "frontier_hv_abs_diff": None,
                    "frontier_match": False,
                    "frontier_nd_ok": False,
                    "nd_count_match": False,
                    "valid": False,
                    "speedup_ratio": 0.0,
                    "error": repr(exc),
                }
            )
            print(
                f"[{label}] {index:02d}/{len(files)} {path.name} "
                f"valid=False speedup=0.000000 error={type(exc).__name__}",
                flush=True,
            )
            continue
        wall_s = float(time.time() - t0)
        base_frontier = base["frontier_objectives_norm"]
        frontier = cand["frontier_objectives_norm"]
        nd_idx = _nd_idx_fast(frontier)
        frontier_nd_ok = int(len(nd_idx)) == int(frontier.shape[0])
        frontier_hv = _hv_from_nd_objs(frontier[nd_idx], ref=HV_REF)
        frontier_hv_diff = abs(float(cand["hv"]) - float(frontier_hv))
        hv_diff = abs(float(cand["hv"]) - float(base["hv"]))
        frontier_match = frontier.shape == base_frontier.shape and np.allclose(frontier, base_frontier, atol=1e-8, rtol=0.0)
        nd_count_match = int(cand["nd_count"]) == int(frontier.shape[0]) == int(base["nd_count"])
        valid = bool(
            hv_diff <= 1e-8
            and frontier_hv_diff <= 1e-8
            and frontier_nd_ok
            and frontier_match
            and nd_count_match
        )
        speedup_ratio = 0.0
        if valid:
            raw_speedup = (float(base["elapsed_s"]) - float(cand["elapsed_s"])) / max(float(base["elapsed_s"]), 1e-12)
            speedup_ratio = max(min(raw_speedup, 1.0), 0.0)
        speedups.append(speedup_ratio)
        rows.append(
            {
                "case": path.name,
                "baseline_s": float(base["elapsed_s"]),
                "candidate_s": float(cand["elapsed_s"]),
                "candidate_wall_s": float(wall_s),
                "baseline_hv": float(base["hv"]),
                "candidate_hv": float(cand["hv"]),
                "baseline_nd_count": int(base["nd_count"]),
                "candidate_nd_count": int(cand["nd_count"]),
                "hv_abs_diff": float(hv_diff),
                "frontier_hv_abs_diff": float(frontier_hv_diff),
                "frontier_match": bool(frontier_match),
                "frontier_nd_ok": bool(frontier_nd_ok),
                "nd_count_match": bool(nd_count_match),
                "valid": bool(valid),
                "speedup_ratio": float(speedup_ratio),
            }
        )
        print(
            f"[{label}] {index:02d}/{len(files)} {path.name} "
            f"valid={valid} speedup={speedup_ratio:.6f} "
            f"base={float(base['elapsed_s']):.4f}s cand={float(cand['elapsed_s']):.4f}s",
            flush=True,
        )
    return {
        "score_large_bonus_raw": float(sum(speedups) / len(speedups)) if speedups else 0.0,
        "score_large_bonus": float(10.0 * sum(speedups) / len(speedups)) if speedups else 0.0,
        "avg_elapsed_s": float(
            sum(float(r["candidate_s"]) for r in rows if r["candidate_s"] is not None)
            / max(sum(1 for r in rows if r["candidate_s"] is not None), 1)
        )
        if rows
        else 0.0,
        "total_elapsed_s": float(sum(float(r["candidate_s"]) for r in rows if r["candidate_s"] is not None)),
        "avg_baseline_s": float(sum(r["baseline_s"] for r in rows) / len(rows)) if rows else 0.0,
        "n_valid": int(sum(1 for r in rows if r["valid"])),
        "n_cases": int(len(rows)),
        "rows": rows,
    }


def benchmark_all_main2(
    shots: int = 200000,
    chunk_size: int = 4096,
    seed: int = 101,
    max_cases: int = 0,
    data_dir: Path | str = DEFAULT_LARGE_DIR,
):
    """统一运行三种 main2 方案，并生成论文中使用的对照 JSON。"""
    data_dir = Path(data_dir).expanduser().resolve()
    files = sorted(data_dir.glob("large_k5_grid40x50_*.npz"))
    if max_cases > 0:
        files = files[:max_cases]
    if not files:
        raise FileNotFoundError(f"No large cases found in {data_dir}")
    t0 = time.time()
    baseline = _evaluate_baseline(files, int(shots), int(chunk_size), int(seed))
    baseline_by_case = baseline.pop("_by_case")
    variants = {
        "official_reference": main2,
        "vectorized_pipeline": main2_vectorized,
        "numba_refinement": main2_numba,
    }
    results = {
        "baseline_main2": baseline,
    }
    for label, fn in variants.items():
        results[label] = _evaluate_variant(
            label,
            fn,
            files,
            baseline_by_case,
            int(shots),
            int(chunk_size),
            int(seed),
        )

    ranking = sorted(
        (
            {
                "scheme": label,
                "score_large_bonus": float(result["score_large_bonus"]),
                "score_large_bonus_raw": float(result["score_large_bonus_raw"]),
                "avg_elapsed_s": float(result["avg_elapsed_s"]),
                "n_valid": int(result.get("n_valid", len(result["rows"]))),
                "n_cases": int(result.get("n_cases", len(result["rows"]))),
            }
            for label, result in results.items()
        ),
        key=lambda item: (item["score_large_bonus"], -item["avg_elapsed_s"]),
        reverse=True,
    )
    payload = {
        "split": "large",
        "shots": int(shots),
        "chunk_size": int(chunk_size),
        "seed": int(seed),
        "data_dir": str(data_dir),
        "n_cases": len(files),
        "elapsed": float(time.time() - t0),
        "score_scale": 10.0,
        "ranking": ranking,
        "results": results,
    }
    return payload


def _prompt_data_dir(default_dir: Path) -> Path:
    """交互式确认 large 数据目录，便于从论文目录或项目根目录运行。"""
    try:
        raw = input(f"Large 数据文件夹路径 [默认: {default_dir}]: ").strip()
    except EOFError:
        raw = ""
    return default_dir if raw == "" else Path(raw).expanduser()


def main() -> int:
    """命令行入口：执行 main2 benchmark 并写出汇总结果。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=200000)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Large case directory. If omitted, the script asks interactively and Enter uses MOO/data/large.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Use default data directory without interactive input when --data-dir is omitted.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "main2_compare_latest_score.json",
    )
    args = parser.parse_args()
    if args.data_dir is None:
        data_dir = DEFAULT_LARGE_DIR if bool(args.no_prompt) else _prompt_data_dir(DEFAULT_LARGE_DIR)
    else:
        data_dir = args.data_dir

    payload = benchmark_all_main2(
        shots=int(args.shots),
        chunk_size=int(args.chunk_size),
        seed=int(args.seed),
        max_cases=int(args.max_cases),
        data_dir=data_dir,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n========== MAIN2 TRIPLET SUMMARY ==========")
    for item in payload["ranking"]:
        print(
            f"{item['scheme']:>20s} "
            f"bonus={item['score_large_bonus']:.6f} "
            f"raw={item['score_large_bonus_raw']:.6f} "
            f"avg_s={item['avg_elapsed_s']:.6f} "
            f"valid={item['n_valid']}/{item['n_cases']}"
        )
    print(f"elapsed(s): {payload['elapsed']:.2f}")
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
