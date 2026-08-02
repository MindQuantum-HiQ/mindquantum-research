"""官方原始 `MOO/answer.py` 的轮次数消融脚本。

该文件只在运行时通过 monkey patch 修改原始 answer 模块中的
`N_ROUNDS` 和 `SHOTS_PER_WEIGHT`，用于记录 1 至 6 轮的原始方案分数。
脚本本身不会改写官方 `MOO/answer.py` 文件。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

MOO_ROOT = Path(__file__).resolve().parents[2]
if str(MOO_ROOT) not in sys.path:
    sys.path.insert(0, str(MOO_ROOT))

import answer as original_answer
import run as judge


SCHEDULES = {
    # 每轮有 100 个权重，所以各列表元素之和必须为 1000，
    # 从而满足 100 * 1000 = 100000 的官方 main1 sample budget。
    1: [1000],
    2: [800, 200],
    3: [600, 200, 200],
    4: [400, 200, 200, 200],
    5: [300, 200, 200, 150, 150],
    6: [250, 150, 150, 150, 150, 150],
}


def configure_rounds(rounds: int) -> list[int]:
    """把指定轮次数的 shot 表注入原始 answer，并校验总预算。"""
    shots = list(SCHEDULES[rounds])
    if len(shots) != rounds:
        raise ValueError(f"round count mismatch: rounds={rounds}, shots={shots}")
    if int(np.sum(shots)) * int(original_answer.NUM_WEIGHTS) != int(
        original_answer.BASE_SAMPLE_BUDGET
    ):
        raise ValueError(f"shot budget mismatch: rounds={rounds}, shots={shots}")
    original_answer.N_ROUNDS = rounds
    original_answer.SHOTS_PER_WEIGHT = shots
    judge.solver_main1 = original_answer.main1
    judge.solver_main2 = original_answer.main2
    return shots


def main() -> dict[str, object]:
    """命令行入口：运行某一个原始 answer 轮次数并保存评分 JSON。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, choices=sorted(SCHEDULES), required=True)
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--large-shots", type=int, default=200000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    shots = configure_rounds(args.rounds)
    print("========== EXPERIMENT CONFIG ==========")
    print("solver             : original MOO/answer.py")
    print(f"rounds             : {args.rounds}")
    print(f"shots_per_weight   : {shots}")
    print(f"weights_per_round  : {original_answer.NUM_WEIGHTS}")
    print(f"sample_budget      : {original_answer.BASE_SAMPLE_BUDGET}")
    print(f"warm_c             : {original_answer.WARM_C_FIXED}")
    print(f"qaoa_depth         : {original_answer.P_LAYERS}")

    started = time.time()
    final = judge.evaluate_split("public", args.max_cases, args.large_shots)
    final["experiment"] = {
        "solver": "original MOO/answer.py",
        "rounds": int(args.rounds),
        "shots_per_weight": shots,
        "weights_per_round": int(original_answer.NUM_WEIGHTS),
        "sample_budget": int(original_answer.BASE_SAMPLE_BUDGET),
        "warm_c": float(original_answer.WARM_C_FIXED),
        "qaoa_depth": int(original_answer.P_LAYERS),
    }
    final["experiment_wall_time_s"] = float(time.time() - started)

    print("\n========== SUMMARY ==========")
    print(f"split             : {final['split']}")
    print(f"rounds            : {args.rounds}")
    print(f"shots_per_weight  : {shots}")
    print(f"score             : {final['score']:.6f}")
    print(f"score_k5_raw      : {final['score_k5_raw']:.6f}")
    print(f"score_k5          : {final['score_k5']:.6f}")
    print(f"score_large_bonus_raw : {final['score_large_bonus_raw']:.6f}")
    print(f"score_large_bonus     : {final['score_large_bonus']:.6f}")
    print(f"elapsed(s)        : {final['elapsed']:.2f}")
    print(f"timeout           : {final['timeout']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"saved report to: {args.out}")
    return final


if __name__ == "__main__":
    main()
