"""public10 经典全枚举上界计算脚本。

该脚本只用于论文分析，不属于可提交求解器：对 20-qubit public cases
枚举全部 2^20 个 spin 构型，得到 main1 的理论 HV 上限，并可选计算
baseline 差值分数。结果用于说明本地量子采样方案距离 public10 上界的
差距，不能在正式 main1 中使用。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pygmo as pg


THIS_FILE = Path(__file__).resolve()
MOO_ROOT = THIS_FILE.parents[2]
if str(MOO_ROOT) not in sys.path:
    sys.path.insert(0, str(MOO_ROOT))

import baseline as baseline_module
from utils import (
    HV_REF,
    _state_index_block_to_spins,
    energy_batch_fast,
    hypervolume_pygmo,
    merge_non_dominated_pool,
    normalize_energies,
    objective_extrema,
    problem_from_npz,
)


PUBLIC_DIR = MOO_ROOT / "data" / "public"
DEFAULT_OUT = THIS_FILE.parent.parent / "results" / "classical_compute_results2.json"


def _first_front(objs: np.ndarray) -> np.ndarray:
    """返回一批目标点中的第一非支配层。"""
    if objs.size == 0:
        return np.zeros((0, objs.shape[1] if objs.ndim == 2 else 0), dtype=np.float64)
    fronts, _, _, _ = pg.fast_non_dominated_sorting(np.asarray(objs, dtype=np.float64))
    if not fronts:
        return np.zeros((0, objs.shape[1]), dtype=np.float64)
    return np.asarray(objs, dtype=np.float64)[np.asarray(fronts[0], dtype=np.int64)]


def _hv_from_spins(problem, spins: np.ndarray, chunk_size: int) -> float:
    """把 spin 样本转成归一化目标值，并计算其非支配前沿 HV。"""
    lower, upper = objective_extrema(problem)
    nd_pool = np.zeros((0, int(problem.k)), dtype=np.float64)
    for start in range(0, int(spins.shape[0]), int(chunk_size)):
        blk = np.asarray(spins[start : start + int(chunk_size)], dtype=np.int8)
        energies = energy_batch_fast(blk, problem.edges, problem.weights, problem.h)
        objs = normalize_energies(energies, lower, upper)
        nd_pool = merge_non_dominated_pool(nd_pool, _first_front(objs))
    if nd_pool.size == 0:
        return 0.0
    return float(hypervolume_pygmo(nd_pool, ref=HV_REF))


def exact_case_hv(case_path: Path, chunk_size: int) -> Dict[str, object]:
    """对单个 public case 枚举全部构型，返回 exact HV 和非支配点数量。"""
    problem = problem_from_npz(str(case_path))
    if int(problem.n) > 20:
        raise ValueError(f"Only n<=20 is supported for exact enumeration, got n={problem.n}")

    lower, upper = objective_extrema(problem)
    total = 1 << int(problem.n)
    nd_pool = np.zeros((0, int(problem.k)), dtype=np.float64)

    t0 = time.time()
    for start in range(0, total, int(chunk_size)):
        count = min(int(chunk_size), total - start)
        spins = _state_index_block_to_spins(start, count, int(problem.n))
        energies = energy_batch_fast(spins, problem.edges, problem.weights, problem.h)
        objs = normalize_energies(energies, lower, upper)
        nd_pool = merge_non_dominated_pool(nd_pool, _first_front(objs))

    exact_hv = float(hypervolume_pygmo(nd_pool, ref=HV_REF))
    return {
        "case": case_path.name,
        "exact_hv": exact_hv,
        "nd_count": int(nd_pool.shape[0]),
        "elapsed_s": float(time.time() - t0),
    }


def baseline_case_hv(case_path: Path, chunk_size: int) -> float:
    """运行官方 baseline main1，得到与 exact 上界比较所需的 baseline HV。"""
    problem = problem_from_npz(str(case_path))
    result = baseline_module.main1(
        problem_input=problem,
        sample_budget=int(baseline_module.BASE_SAMPLE_BUDGET),
        rng_seed=2026,
    )
    spins = np.asarray(result["sample_spins"], dtype=np.int8)
    return _hv_from_spins(problem, spins, chunk_size=chunk_size)


def main() -> None:
    """命令行入口：批量计算 public cases 的经典上界并写出 JSON。"""
    parser = argparse.ArgumentParser(
        description="Exact public10 HV oracle for the 20-qubit MOO main1 cases."
    )
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all public cases.")
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--with-baseline", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    case_paths = sorted(PUBLIC_DIR.glob("k5_grid4x5_*.npz"))
    if int(args.max_cases) > 0:
        case_paths = case_paths[: int(args.max_cases)]
    if not case_paths:
        raise FileNotFoundError(f"No public cases found in {PUBLIC_DIR}")

    rows: List[Dict[str, object]] = []
    t0 = time.time()
    for index, case_path in enumerate(case_paths, start=1):
        row = exact_case_hv(case_path, chunk_size=int(args.chunk_size))
        if bool(args.with_baseline):
            base_hv = baseline_case_hv(case_path, chunk_size=int(args.chunk_size))
            row["baseline_hv"] = float(base_hv)
            row["exact_score"] = float(100000.0 * max(float(row["exact_hv"]) - base_hv, 0.0))
        rows.append(row)
        msg = (
            f"[{index:02d}/{len(case_paths)}] {row['case']} "
            f"exact_hv={float(row['exact_hv']):.12f} nd={int(row['nd_count'])} "
            f"elapsed={float(row['elapsed_s']):.2f}s"
        )
        if "exact_score" in row:
            msg += f" exact_score={float(row['exact_score']):.4f}"
        print(msg, flush=True)

    mean_exact = float(np.mean([float(row["exact_hv"]) for row in rows])) if rows else 0.0
    payload: Dict[str, object] = {
        "dataset": "public10",
        "method": "full enumeration of all 2^20 spin configurations per case",
        "hv_ref": float(HV_REF),
        "chunk_size": int(args.chunk_size),
        "elapsed_s": float(time.time() - t0),
        "mean_exact_hv": mean_exact,
        "rows": rows,
    }
    if bool(args.with_baseline):
        scores = [float(row["exact_score"]) for row in rows if "exact_score" in row]
        payload["mean_exact_score"] = float(np.mean(scores)) if scores else 0.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"mean_exact_hv={mean_exact:.12f}")
    if "mean_exact_score" in payload:
        print(f"mean_exact_score={float(payload['mean_exact_score']):.4f}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
