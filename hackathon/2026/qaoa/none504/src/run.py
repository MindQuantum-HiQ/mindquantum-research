from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pygmo as pg

import answer
import baseline as baseline_module
from utils import (
    HV_REF,
    default_weight_pool_path,
    energy_batch_fast,
    hypervolume_pygmo,
    merge_non_dominated_pool,
    normalize_energies,
    objective_extrema,
    problem_from_npz,
)

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
PUBLIC_DIR = DATA_ROOT / "public"
HIDDEN_DIR = DATA_ROOT / "_hidden"
LARGE_DIR = DATA_ROOT / "large"
TIME_LIMIT_SECONDS = 60 * 60
DEFAULT_REPORT_PATH = ROOT / "results" / "latest_score.json"
BASELINE_CACHE_PATH = ROOT / "results" / "baseline_cache.json"
SMALL_SCORE_SCALE = 100000.0
LARGE_SCORE_SCALE = 10.0
SMALL_OBJECTIVE_K = 5

BASELINE_SAMPLE_BUDGET = int(baseline_module.BASE_SAMPLE_BUDGET)
LARGE_SHOTS_DEFAULT = 200000

solver_main1 = answer.main1
solver_main2 = answer.main2

# Persisted caches
_BASE_HV_CACHE: Dict[str, float] = {}
_BASE_LARGE_CACHE: Dict[str, Dict[str, object]] = {}

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")


def _load_baseline_cache() -> None:
    if not BASELINE_CACHE_PATH.exists():
        return
    try:
        raw = json.loads(BASELINE_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return

    small = raw.get("small_hv", {})
    if isinstance(small, dict):
        for k, v in small.items():
            try:
                _BASE_HV_CACHE[str(k)] = float(v)
            except Exception:
                continue

    large = raw.get("large_baseline", {})
    if isinstance(large, dict):
        for k, v in large.items():
            if not isinstance(v, dict):
                continue
            try:
                frontier_raw = v.get("frontier_objectives_norm")
                frontier = None
                if frontier_raw is not None:
                    frontier_arr = np.asarray(frontier_raw, dtype=np.float64)
                    if frontier_arr.ndim != 2:
                        raise ValueError("frontier_objectives_norm cache must be a 2D array.")
                    frontier = frontier_arr.tolist()
                _BASE_LARGE_CACHE[str(k)] = {
                    "hv": float(v["hv"]),
                    "elapsed_s": float(v["elapsed_s"]),
                    "nd_count": int(v["nd_count"]),
                    "frontier_objectives_norm": frontier,
                }
            except Exception:
                continue


def _save_baseline_cache() -> None:
    BASELINE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"small_hv": _BASE_HV_CACHE, "large_baseline": _BASE_LARGE_CACHE}
    BASELINE_CACHE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def _time_exceeded(t0: float) -> bool:
    return (time.time() - t0) > TIME_LIMIT_SECONDS



def list_small_cases(split_dir: Path, k: int, max_cases: int = 0) -> List[Path]:
    files = sorted(split_dir.glob(f"k{k}_grid4x5_*.npz"))
    if max_cases > 0:
        files = files[:max_cases]
    return files


def list_large_cases(max_cases: int = 0) -> List[Path]:
    files = sorted(LARGE_DIR.glob(f"large_k{SMALL_OBJECTIVE_K}_grid40x50_*.npz"))
    if max_cases > 0:
        files = files[:max_cases]
    return files


def _require_sample_spins(problem, result: Dict[str, object]) -> np.ndarray:
    if not isinstance(result, dict):
        raise TypeError("Solver main1() must return a dict.")
    if "sample_spins" not in result:
        raise KeyError("Solver main1() result must contain 'sample_spins'.")

    spins = np.asarray(result["sample_spins"], dtype=np.int8)
    if spins.ndim != 2 or spins.shape[1] != int(problem.n):
        raise ValueError(
            f"sample_spins must have shape [n_samples, {int(problem.n)}], got {spins.shape}."
        )
    if spins.size > 0 and not np.all(np.isin(np.unique(spins), np.array([-1, 1], dtype=np.int8))):
        raise ValueError("sample_spins must contain only -1/+1 spins.")
    return spins


def _nd_idx_fast(objs: np.ndarray) -> np.ndarray:
    if objs.size == 0:
        return np.zeros((0,), dtype=np.int64)
    arr = np.asarray(objs, dtype=np.float64)
    fronts, _, _, _ = pg.fast_non_dominated_sorting(arr)
    if len(fronts) == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(fronts[0], dtype=np.int64)


def _hv_from_nd_objs(nd_objs: np.ndarray, ref: float = HV_REF) -> float:
    if nd_objs.size == 0:
        return 0.0
    return float(hypervolume_pygmo(np.asarray(nd_objs, dtype=np.float64), ref=ref))


def _hv_from_spins(problem, spins: np.ndarray) -> float:
    """Exact HV evaluation for main1 samples."""
    if spins.size == 0:
        return 0.0
    spins = np.asarray(spins, dtype=np.int8)
    lower_bounds, upper_bounds = objective_extrema(problem)
    chunk = 4096
    nd_pool = np.zeros((0, int(problem.k)), dtype=np.float64)

    n = spins.shape[0]
    for s in range(0, n, chunk):
        blk = spins[s : s + chunk]
        energies = energy_batch_fast(blk, problem.edges, problem.weights, problem.h)
        objs = normalize_energies(energies, lower_bounds, upper_bounds)
        nd_pool = merge_non_dominated_pool(nd_pool, objs[_nd_idx_fast(objs)])
    return _hv_from_nd_objs(nd_pool, ref=HV_REF)


def _small_weight_pool_signature() -> str:
    pool_path = default_weight_pool_path(SMALL_OBJECTIVE_K, n=1000, seed=2026)
    try:
        raw = pool_path.read_bytes()
    except Exception:
        return "pool=missing"
    return f"pool={hashlib.sha1(raw).hexdigest()[:12]}"


def _small_baseline_cache_key(case_path: Path) -> str:
    return (
        f"small::{case_path.parent.name}/{case_path.name}"
        f"::budget={BASELINE_SAMPLE_BUDGET}"
        f"::ref={HV_REF:.6f}"
        f"::norm=exact_extrema_v1"
        f"::{_small_weight_pool_signature()}"
    )


def baseline_hv(case_path: Path, problem) -> float:
    """Compute & cache baseline HV for small cases."""
    key = _small_baseline_cache_key(case_path)
    if key in _BASE_HV_CACHE:
        return _BASE_HV_CACHE[key]

    base_res = baseline_module.main1(
        problem_input=problem,
        sample_budget=BASELINE_SAMPLE_BUDGET,
        rng_seed=2026,
    )
    spins = _require_sample_spins(problem, base_res)
    hv = _hv_from_spins(problem, spins)
    _BASE_HV_CACHE[key] = float(hv)
    _save_baseline_cache()
    return float(hv)

def _call_solver_main1(problem):
    return solver_main1(problem_input=problem, sample_budget=int(BASELINE_SAMPLE_BUDGET))


def score_small_objective_set(
    split_dir: Path,
    k: int,
    max_cases: int,
    t0: float,
) -> Tuple[float, List[Dict[str, object]], bool]:
    rows: List[Dict[str, object]] = []
    files = list_small_cases(split_dir, k, max_cases=max_cases)
    if not files:
        raise FileNotFoundError(f"No k={k} cases found in {split_dir}")

    timeout = False
    for i, path in enumerate(files, start=1):
        if _time_exceeded(t0):
            timeout = True
            break

        problem = problem_from_npz(str(path))
        c0 = time.time()
        result = _call_solver_main1(problem)
        solve_t = time.time() - c0
        if not isinstance(result, dict):
            result = {}

        spins = _require_sample_spins(problem, result)
        hv_solver = _hv_from_spins(problem, spins)
        hv_base = baseline_hv(path, problem)

        sample_count = int(spins.shape[0])
        sample_used = int(result.get("sample_used", sample_count))
        budget_ok = sample_used == BASELINE_SAMPLE_BUDGET and sample_count == BASELINE_SAMPLE_BUDGET
        hv_gain = float(max(hv_solver - hv_base, 0.0)) if budget_ok else 0.0
        score_case = float(SMALL_SCORE_SCALE * hv_gain)

        rows.append(
            {
                "case": path.name,
                "k": k,
                "solve_time": float(solve_t),
                "sample_used": int(sample_used),
                "sample_rows": int(sample_count),
                "budget_ok": bool(budget_ok),
                "hv_base": float(hv_base),
                "hv_solver": float(hv_solver),
                "hv_gain": float(hv_gain),
                "score_case": float(score_case),
            }
        )

        print(
            f"[k={k}] {i:02d}/{len(files)} {path.name} "
            f"hv={hv_solver:.5f} base={hv_base:.5f} gain={hv_gain:.5f} "
            f"shots={sample_count} t={solve_t:.2f}s"
        )

        if _time_exceeded(t0):
            timeout = True
            break

    if timeout:
        return 0.0, rows, True

    score = float(np.mean([r["score_case"] for r in rows]))
    return score, rows, False


# ---------------- Large (main2) with baseline caching, but ALWAYS call candidate main2 ----------------

def _large_baseline_cache_key(case_path: Path, shots: int, chunk_size: int, rng_seed: int) -> str:
    return (
        f"large::{case_path.name}::shots={int(shots)}::chunk={int(chunk_size)}"
        f"::seed={int(rng_seed)}::ref={HV_REF:.6f}::norm=exact_extrema_v1"
        f"::judge=wallclock_frontier_v2"
    )


def _large_eval_case_main2(fn, problem, shots: int, chunk_size: int, rng_seed: int) -> Dict[str, object]:
    t0 = time.perf_counter()
    out = fn(
        problem_input=problem,
        shots=int(shots),
        chunk_size=int(chunk_size),
        rng_seed=int(rng_seed),
    )
    elapsed = float(time.perf_counter() - t0)
    if not isinstance(out, dict):
        raise TypeError("Solver main2() must return a dict.")

    hv = float(out["hv"])
    if "nd_count" not in out:
        raise KeyError("Solver main2() result must contain 'nd_count'.")
    ndc = int(out["nd_count"])
    reported_elapsed = out.get("elapsed_s")
    if reported_elapsed is not None:
        reported_elapsed = float(reported_elapsed)

    if "frontier_objectives_norm" not in out:
        raise KeyError("Solver main2() result must contain 'frontier_objectives_norm'.")
    frontier_raw = out["frontier_objectives_norm"]
    if frontier_raw is None:
        raise ValueError("frontier_objectives_norm must not be None.")
    frontier = np.asarray(frontier_raw, dtype=np.float64)
    if frontier.ndim == 1 and frontier.size == 0:
        frontier = np.zeros((0, int(problem.k)), dtype=np.float64)
    if frontier.ndim != 2 or frontier.shape[1] != int(problem.k):
        raise ValueError(
            "frontier_objectives_norm must have shape "
            f"[n_points, {int(problem.k)}], got {frontier.shape}."
        )
    if int(frontier.shape[0]) > 1:
        order = np.lexsort(frontier[:, ::-1].T)
        frontier = frontier[order]
    return {
        "hv": hv,
        "elapsed_s": elapsed,
        "elapsed_s_reported": reported_elapsed,
        "nd_count": ndc,
        "frontier_objectives_norm": frontier,
    }


def _baseline_large_eval(case_path: Path, problem, shots: int, chunk_size: int, rng_seed: int) -> Dict[str, object]:
    key = _large_baseline_cache_key(case_path, shots, chunk_size, rng_seed)
    if key in _BASE_LARGE_CACHE:
        return dict(_BASE_LARGE_CACHE[key])

    out = _large_eval_case_main2(
        baseline_module.main2,
        problem=problem,
        shots=shots,
        chunk_size=chunk_size,
        rng_seed=rng_seed,
    )
    core = {
        "hv": float(out["hv"]),
        "elapsed_s": float(out["elapsed_s"]),
        "nd_count": int(out["nd_count"]),
        "frontier_objectives_norm": np.asarray(out["frontier_objectives_norm"], dtype=np.float64).tolist(),
    }
    _BASE_LARGE_CACHE[key] = dict(core)
    _save_baseline_cache()
    return core

def _call_solver_main2(problem, shots: int, rng_seed: int, chunk_size: int) -> Dict[str, object]:
    return _large_eval_case_main2(
        solver_main2,
        problem=problem,
        shots=shots,
        chunk_size=chunk_size,
        rng_seed=rng_seed,
    )


def bonus_large_set(max_cases: int, shots: int, t0: float) -> Dict[str, object]:
    files = list_large_cases(max_cases=max_cases)
    if not files:
        return {"timeout": False, "score_bonus_raw": 0.0, "rows": [], "note": "no large cases"}

    rows = []
    bonus_items = []
    for i, path in enumerate(files, start=1):
        if _time_exceeded(t0):
            return {"timeout": True, "score_bonus_raw": 0.0, "rows": rows}

        problem = problem_from_npz(str(path))
        seed = 101
        chunk_size = 4096

        # baseline: cached after first run
        base = _baseline_large_eval(path, problem, shots, chunk_size, seed)
        base_frontier = np.asarray(base["frontier_objectives_norm"], dtype=np.float64)

        # candidate: ALWAYS call answer.main2 (required)
        cand_error = None
        try:
            cand_res = _call_solver_main2(problem, shots, seed, chunk_size)
        except Exception as exc:
            cand_res = None
            cand_error = f"{type(exc).__name__}: {exc}"

        cand_elapsed = np.nan
        cand_elapsed_reported = None
        cand_hv = np.nan
        hv_diff = np.nan
        frontier_hv_diff = np.nan
        frontier_nd_ok = False
        frontier_match = False
        nd_count_match = False
        valid = False
        speedup_ratio = 0.0

        if cand_res is not None:
            cand_elapsed = float(cand_res["elapsed_s"])
            cand_elapsed_reported = cand_res.get("elapsed_s_reported")
            cand_hv = float(cand_res["hv"])
            frontier = np.asarray(cand_res["frontier_objectives_norm"], dtype=np.float64)

            nd_idx = _nd_idx_fast(frontier)
            frontier_nd_ok = int(len(nd_idx)) == int(frontier.shape[0])
            frontier_hv = _hv_from_nd_objs(frontier[nd_idx], ref=HV_REF)
            frontier_hv_diff = abs(cand_hv - frontier_hv)

            hv_diff = abs(cand_hv - float(base["hv"]))
            frontier_match = (
                frontier.shape == base_frontier.shape
                and np.allclose(frontier, base_frontier, atol=1e-8, rtol=0.0)
            )
            nd_count_match = int(cand_res["nd_count"]) == int(frontier.shape[0]) == int(base["nd_count"])
            valid = bool(
                hv_diff <= 1e-8
                and frontier_hv_diff <= 1e-8
                and frontier_nd_ok
                and frontier_match
                and nd_count_match
            )
            if valid:
                raw_speedup = (float(base["elapsed_s"]) - cand_elapsed) / max(float(base["elapsed_s"]), 1e-12)
                speedup_ratio = float(np.clip(raw_speedup, 0.0, 1.0))

        bonus_items.append(speedup_ratio)
        row = {
            "case": path.name,
            "shots": int(shots),
            "baseline_hv": float(base["hv"]),
            "candidate_hv": float(cand_hv) if np.isfinite(cand_hv) else None,
            "hv_abs_diff": float(hv_diff) if np.isfinite(hv_diff) else None,
            "frontier_hv_abs_diff": float(frontier_hv_diff) if np.isfinite(frontier_hv_diff) else None,
            "frontier_nd_ok": bool(frontier_nd_ok),
            "frontier_match_baseline": bool(frontier_match),
            "nd_count_match": bool(nd_count_match),
            "baseline_s": float(base["elapsed_s"]),
            "candidate_s": float(cand_elapsed) if np.isfinite(cand_elapsed) else None,
            "candidate_s_reported": float(cand_elapsed_reported) if cand_elapsed_reported is not None else None,
            "speedup_ratio": float(speedup_ratio),
            "valid": bool(valid),
            "error": cand_error,
        }
        rows.append(row)

        cand_text = f"{float(cand_elapsed):.2f}s" if np.isfinite(cand_elapsed) else "ERR"
        hv_text = f"{float(hv_diff):.2e}" if np.isfinite(hv_diff) else "ERR"
        print(
            f"[large] {i:02d}/{len(files)} {path.name} "
            f"base={float(base['elapsed_s']):.2f}s cand={cand_text} "
            f"speedup={speedup_ratio:.4f} hv_diff={hv_text} "
            f"frontier_match={frontier_match} valid={valid}"
        )
        if cand_error:
            print(f"          error={cand_error}")

        if _time_exceeded(t0):
            return {"timeout": True, "score_bonus_raw": 0.0, "rows": rows}

    score_bonus_raw = float(np.mean(bonus_items)) if bonus_items else 0.0
    return {
        "timeout": False,
        "score_bonus_raw": score_bonus_raw,
        "rows": rows,
        "shots": int(shots),
        "n_cases": len(files),
    }


def _split_dir(split: str) -> Path:
    if split == "public":
        return PUBLIC_DIR
    if split == "hidden":
        return HIDDEN_DIR
    raise ValueError(f"Unsupported split for small-set evaluation: {split}")


def _score_from_rows(rows: List[Dict[str, object]]) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(r["score_case"]) for r in rows]))


def evaluate_split(split: str, max_cases: int, large_shots: int) -> Dict[str, object]:
    if split not in ("public", "hidden"):
        raise ValueError("evaluate_split only supports 'public' or 'hidden'.")

    t0 = time.time()
    split_dir = _split_dir(split)
    score_k5, rows5, timeout5 = score_small_objective_set(split_dir, SMALL_OBJECTIVE_K, max_cases, t0)
    if timeout5:
        return {
            "split": split,
            "timeout": True,
            "elapsed": float(time.time() - t0),
            "score": 0.0,
            "score_k5_raw": 0.0,
            "score_k5": 0.0,
            "score_large_bonus": 0.0,
            "score_large_bonus_raw": 0.0,
            "k5_rows": rows5,
            "large_rows": [],
        }

    if _time_exceeded(t0):
        return {
            "split": split,
            "timeout": True,
            "elapsed": float(time.time() - t0),
            "score": 0.0,
            "score_k5_raw": 0.0,
            "score_k5": 0.0,
            "score_large_bonus": 0.0,
            "score_large_bonus_raw": 0.0,
            "k5_rows": rows5,
            "large_rows": [],
        }

    large = bonus_large_set(max_cases=max_cases, shots=large_shots, t0=t0)
    if bool(large.get("timeout", False)):
        return {
            "split": split,
            "timeout": True,
            "elapsed": float(time.time() - t0),
            "score": 0.0,
            "score_k5_raw": 0.0,
            "score_k5": 0.0,
            "score_large_bonus": 0.0,
            "score_large_bonus_raw": 0.0,
            "k5_rows": rows5,
            "large_rows": list(large.get("rows", [])),
        }

    score_k5_raw = float(np.mean([float(r["hv_gain"]) for r in rows5])) if rows5 else 0.0
    score_large_raw = float(large["score_bonus_raw"])
    score_k5 = float(SMALL_SCORE_SCALE * score_k5_raw)
    score_large = float(LARGE_SCORE_SCALE * score_large_raw)
    total = float(score_k5 + score_large)

    return {
        "split": split,
        "timeout": False,
        "elapsed": float(time.time() - t0),
        "score": total,
        "score_k5_raw": float(score_k5_raw),
        "score_k5": float(score_k5),
        "score_large_bonus": float(score_large),
        "score_large_bonus_raw": float(score_large_raw),
        "k5_rows": rows5,
        "large_rows": large["rows"],
        "large_meta": {
            "shots": int(large_shots),
            "n_cases": int(large.get("n_cases", 0)),
            "score_weight": float(LARGE_SCORE_SCALE),
        },
    }


def evaluate_all(max_cases: int, large_shots: int) -> Dict[str, object]:
    t0 = time.time()

    score_pub, rows_pub, timeout_pub = score_small_objective_set(
        PUBLIC_DIR, SMALL_OBJECTIVE_K, max_cases, t0
    )
    if timeout_pub:
        return {
            "split": "all",
            "timeout": True,
            "elapsed": float(time.time() - t0),
            "score": 0.0,
            "score_k5_raw": 0.0,
            "score_k5": 0.0,
            "score_k5_public": 0.0,
            "score_k5_hidden": 0.0,
            "score_large_bonus": 0.0,
            "score_large_bonus_raw": 0.0,
            "k5_public_rows": rows_pub,
            "k5_hidden_rows": [],
            "k5_rows": rows_pub,
            "large_rows": [],
        }

    score_hid, rows_hid, timeout_hid = score_small_objective_set(
        HIDDEN_DIR, SMALL_OBJECTIVE_K, max_cases, t0
    )
    if timeout_hid:
        rows_all = rows_pub + rows_hid
        return {
            "split": "all",
            "timeout": True,
            "elapsed": float(time.time() - t0),
            "score": 0.0,
            "score_k5_raw": 0.0,
            "score_k5": 0.0,
            "score_k5_public": float(score_pub),
            "score_k5_hidden": 0.0,
            "score_large_bonus": 0.0,
            "score_large_bonus_raw": 0.0,
            "k5_public_rows": rows_pub,
            "k5_hidden_rows": rows_hid,
            "k5_rows": rows_all,
            "large_rows": [],
        }

    rows_all = rows_pub + rows_hid
    score_k5_all = _score_from_rows(rows_all)

    if _time_exceeded(t0):
        return {
            "split": "all",
            "timeout": True,
            "elapsed": float(time.time() - t0),
            "score": 0.0,
            "score_k5_raw": 0.0,
            "score_k5": 0.0,
            "score_k5_public": float(score_pub),
            "score_k5_hidden": float(score_hid),
            "score_large_bonus": 0.0,
            "score_large_bonus_raw": 0.0,
            "k5_public_rows": rows_pub,
            "k5_hidden_rows": rows_hid,
            "k5_rows": rows_all,
            "large_rows": [],
        }

    # Evaluate large set exactly once for --split all.
    large = bonus_large_set(max_cases=max_cases, shots=large_shots, t0=t0)
    if bool(large.get("timeout", False)):
        return {
            "split": "all",
            "timeout": True,
            "elapsed": float(time.time() - t0),
            "score": 0.0,
            "score_k5_raw": 0.0,
            "score_k5": 0.0,
            "score_k5_public": float(score_pub),
            "score_k5_hidden": float(score_hid),
            "score_large_bonus": 0.0,
            "score_large_bonus_raw": 0.0,
            "k5_public_rows": rows_pub,
            "k5_hidden_rows": rows_hid,
            "k5_rows": rows_all,
            "large_rows": list(large.get("rows", [])),
        }

    score_k5_raw = float(np.mean([float(r["hv_gain"]) for r in rows_all])) if rows_all else 0.0
    score_large_raw = float(large["score_bonus_raw"])
    score_large = float(LARGE_SCORE_SCALE * score_large_raw)
    total = float(score_k5_all + score_large)

    return {
        "split": "all",
        "timeout": False,
        "elapsed": float(time.time() - t0),
        "score": total,
        "score_k5_raw": float(score_k5_raw),
        "score_k5": float(score_k5_all),
        "score_k5_public": float(score_pub),
        "score_k5_hidden": float(score_hid),
        "score_large_bonus": float(score_large),
        "score_large_bonus_raw": float(score_large_raw),
        "k5_public_rows": rows_pub,
        "k5_hidden_rows": rows_hid,
        "k5_rows": rows_all,
        "large_rows": large["rows"],
        "large_meta": {
            "shots": int(large_shots),
            "n_cases": int(large.get("n_cases", 0)),
            "score_weight": float(LARGE_SCORE_SCALE),
        },
    }


def main() -> Dict[str, object]:
    parser = argparse.ArgumentParser(description="Hackathon-MOO local judge")
    parser.add_argument("--split", choices=["public", "hidden", "all"], default="all")
    parser.add_argument("--sample-budget", type=int, default=BASELINE_SAMPLE_BUDGET)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--large-shots", type=int, default=LARGE_SHOTS_DEFAULT)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    if int(args.sample_budget) != BASELINE_SAMPLE_BUDGET:
        print(
            f"[judge] ignore --sample-budget={args.sample_budget}; "
            f"fixed to baseline budget {BASELINE_SAMPLE_BUDGET} for fairness."
        )

    if args.split == "all":
        final = evaluate_all(args.max_cases, args.large_shots)
    else:
        final = evaluate_split(args.split, args.max_cases, args.large_shots)

    print("\n========== SUMMARY ==========")
    if final["split"] == "all":
        print(f"score             : {final['score']:.6f}")
        print(f"score_k5_raw      : {final['score_k5_raw']:.6f}")
        print(f"score_k5_all      : {final['score_k5']:.6f}")
        print(f"score_k5_public   : {final['score_k5_public']:.6f}")
        print(f"score_k5_hidden   : {final['score_k5_hidden']:.6f}")
        print(f"score_large_bonus_raw : {final['score_large_bonus_raw']:.6f}")
        print(f"score_large_bonus     : {final['score_large_bonus']:.6f} ")
        print(f"elapsed(s)        : {final['elapsed']:.2f}")
        print(f"timeout           : {final['timeout']}")
        print(f"final_score : {final['score']:.6f}")
    else:
        print(f"split             : {final['split']}")
        print(f"score             : {final['score']:.6f}")
        print(f"score_k5_raw      : {final['score_k5_raw']:.6f}")
        print(f"score_k5          : {final['score_k5']:.6f}")
        print(f"score_large_bonus_raw : {final['score_large_bonus_raw']:.6f}")
        print(f"score_large_bonus     : {final['score_large_bonus']:.6f} ")
        print(f"elapsed(s)        : {final['elapsed']:.2f}")
        print(f"timeout           : {final['timeout']}")

    DEFAULT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT_PATH.write_text(
        json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"saved latest report to: {DEFAULT_REPORT_PATH}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"saved report to: {out_path}")

    return final


_load_baseline_cache()

if __name__ == "__main__":
    main()
