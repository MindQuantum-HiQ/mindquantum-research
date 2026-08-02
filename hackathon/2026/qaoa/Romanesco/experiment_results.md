# MOO 实验结果速查

本文档只保留总分、耗时、shots 分配和结果文件映射。逐 case HV、large 明细和日志不在此展开，均以对应 JSON/log 为准。云端分数以提交记录为准；本地分数基于 public10，不代表 hidden/all。

## 最高结论

| 结论 | 数值/说明 |
| --- | --- |
| 当前本地 public10 最高总分 | `coord_4round_2`，`238.014593` |
| 当前本地 public10 最高多轮协调设计 | `coord_4round_2`，`238.014593` |
| 协调设计中最优 shots 分配 | `[400, 400, 150, 50]` |
| 原始 `answer.py` round sweep 最优 | 5 轮 `[300, 200, 200, 150, 150]`，`score_k5=150.653053`，总分 `152.061707` |
| main1 经典全枚举 public10 理论上限 | `score_k5 = 233.0624`；本地 main1 最好为 `answer_singleround.py` 的 `230.016421`，协调设计最好为 `coord_4round_2` 的 `229.920256` |
| 1 小时边界 | 两组 round sweep 均通过；协调设计最慢为 `coord_6round` 的 `1439.08s`，原始扫描最慢为 6 轮的 `1914.36s` |

## 当前方案总分

| 方案 | 定位 | 结果文件 | score_k5 | main2 bonus | 本地总分 | 完整耗时 | 1h 边界 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `answer_singleround.py` | 强单轮 mixed-depth ensemble | `results/answer_singleround_score.json` | 230.016421 | 7.472922 | 237.489343 | 639.40s | 通过 |
| `answer_singleround_to_multi.py` | 强单轮朴素拆三轮 warm-start | `results/answer_singleround_to_multi_score.json` | 223.877869 | 7.836676 | 231.714545 | 1449.80s | 通过 |
| `answer_multiround.py` | 多轮协调设计代表 | `results/answer_multiround_score/answer_multiround_score_4round_2.json` | 229.920256 | 8.094337 | 238.014593 | 987.41s | 通过 |

说明：当前多轮实验统一以 `results/answer_multiround_score/` 子目录为准；根目录旧 `answer_multiround_score.json` 不再作为论文主表引用。

## 协调设计轮次消融

这些结果运行在协同设计版 `answer_multiround*.py` 上，不等同于原始 `MOO/answer.py`。所有配置均使用 `NUM_WEIGHTS=100`，满足 `100 * sum(SHOTS_PER_WEIGHT) = 100000`。

| 实验名 | `SHOTS_PER_WEIGHT` | 每轮总 shots | 结果文件 | score_k5 | main2 bonus | 本地总分 | 完整耗时 | 状态 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `coord_1round` | `[1000]` | `100000` | `results/answer_multiround_score/answer_multiround_score_1round.json` | 195.563747 | 8.048260 | 203.612007 | 227.71s | 已测 |
| `coord_2round` | `[800, 200]` | `80000 + 20000` | `results/answer_multiround_score/answer_multiround_score_2round.json` | 219.058470 | 8.028937 | 227.087407 | 593.05s | 已测 |
| `coord_3round` | `[600, 200, 200]` | `60000 + 20000 + 20000` | `results/answer_multiround_score/answer_multiround_score_3round.json` | 225.759503 | 8.041676 | 233.801179 | 787.07s | 已测 |
| `coord_4round` | `[400, 200, 200, 200]` | `40000 + 20000 + 20000 + 20000` | `results/answer_multiround_score/answer_multiround_score_4round.json` | 227.981622 | 8.029948 | 236.011569 | 998.10s | 已测 |
| `coord_4round_2` | `[400, 400, 150, 50]` | `40000 + 40000 + 15000 + 5000` | `results/answer_multiround_score/answer_multiround_score_4round_2.json` | 229.920256 | 8.094337 | 238.014593 | 987.41s | 已测，当前最高 |
| `coord_5round` | `[300, 200, 200, 150, 150]` | `30000 + 20000 + 20000 + 15000 + 15000` | `results/answer_multiround_score/answer_multiround_score_5round.json` | 225.500097 | 8.074428 | 233.574525 | 1188.79s | 已测 |
| `coord_6round` | `[250, 150, 150, 150, 150, 150]` | `25000 + 5*15000` | `results/answer_multiround_score/answer_multiround_score_6round.json` | 224.855101 | 7.865146 | 232.720247 | 1439.08s | 已测 |

### 协调设计结论

1. `coord_4round_2` 目前最好，总分 `238.014593`，超过强单轮 `237.489343`。
2. 从 `coord_1round` 到 `coord_4round`，score_k5 由 `195.563747` 提升到 `227.981622`，说明协同反馈确实改善 Pareto 覆盖。
3. `coord_4round_2` 相比 `coord_4round`，score_k5 提升 `1.938634`，总分提升 `2.003024`。
4. `coord_5round` 和 `coord_6round` 没有继续提升，其中 `coord_6round` 本次重跑 `score_k5=224.855101`、总分 `232.720247`，说明更多轮次仍不是单调改进条件。
5. 后续论文最佳四轮例子使用 `coord_4round_2=[400,400,150,50]`，对应 `answer_multiround_score_4round_2.json`。

## 原始 Answer Round Sweep

这部分是未修改官方 `MOO/answer.py` 的 round sweep 结果，只记录总分和耗时。逐 case 原始数据见 `results/original_answer_round_sweep/round_*.json` 和同名 `.log`。

| round | `SHOTS_PER_WEIGHT` | 结果文件 | score_k5 | main2 bonus | 本地总分 | 完整耗时 | timeout |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `[1000]` | `results/original_answer_round_sweep/round_1.json` | 0.000000 | 0.772233 | 0.772233 | 391.69s | false |
| 2 | `[800, 200]` | `results/original_answer_round_sweep/round_2.json` | 120.015698 | 1.267013 | 121.282711 | 718.24s | false |
| 3 | `[600, 200, 200]` | `results/original_answer_round_sweep/round_3.json` | 129.949585 | 1.337968 | 131.287553 | 986.96s | false |
| 4 | `[400, 200, 200, 200]` | `results/original_answer_round_sweep/round_4.json` | 131.031024 | 1.340578 | 132.371602 | 1289.36s | false |
| 5 | `[300, 200, 200, 150, 150]` | `results/original_answer_round_sweep/round_5.json` | 150.653053 | 1.408654 | 152.061707 | 1558.28s | false |
| 6 | `[250, 150, 150, 150, 150, 150]` | `results/original_answer_round_sweep/round_6.json` | 144.379745 | 1.419895 | 145.799640 | 1914.36s | false |

### 原始 Answer 结论

1. 原始 `answer.py` 默认是三轮 `[600, 200, 200]`，不是四轮。
2. 原始 round sweep 中 5 轮最高，`score_k5=150.653053`、总分 `152.061707`，但仍远低于当前强单轮和协调设计方案。
3. 6 轮的 `main2 bonus` 略增至 `1.419895`，但 `score_k5` 回落到 `144.379745`、总分回落到 `145.799640`，说明更多轮次不一定改善量子采样质量。
4. 最新重跑完整耗时从 1 轮的 `391.69s` 增至 6 轮的 `1914.36s`，所有配置均未触发 timeout，且处于 1 小时限制内。

## 云端提交记录

| 方案 | 文件/组合 | 云端总分 | 备注 |
| --- | --- | ---: | --- |
| v590 hybrid | `vs205.13.py` main1 + v568 main2 | 211.16355 | 当前记录最高云端提交 |
| v568 | v568 main1 + v568 main2 | 210.65164 | 前一个高分方案 |

## Main2 Large 速度

结果文件：`results/main2_compare_latest_score.json`。

| 方案 | 定位 | score_large_bonus | raw speedup | avg elapsed(s) | valid cases |
| --- | --- | ---: | ---: | ---: | ---: |
| `numba_refinement` | 优化方案 | 7.603685 | 0.760369 | 4.259303 | 10/10 |
| `vectorized_pipeline` | 优化方案 | 7.141069 | 0.714107 | 5.092364 | 10/10 |
| `baseline_main2` | 评分基准 | 0.000000 | 0.000000 | 17.804750 | 10/10 |
| `official_reference` | 官方参考实现 | 0.000000 | 0.000000 | 20.587866 | 10/10 |

## main1 经典理论上限

结果文件：`results/classical_compute_results.json`。该结果来自 public10 的 `2^20` 全枚举，仅作为 main1 分析上界，不是提交方案。旁边附当前本地 main1 最好结果，便于判断与理论上限的差距。

| 指标 | 数值 |
| --- | ---: |
| mean exact HV | 0.598481003864 |
| exact score_k5 | 233.0624 |
| 本地 main1 最好 score_k5 | 230.016421（`answer_singleround.py`） |
| 本地多轮协调设计最好 score_k5 | 229.920256（`coord_4round_2`） |
| elapsed | 3356.88s |

## 明细索引

| 明细类型 | 查找位置 |
| --- | --- |
| 协调设计逐 case 分数 | `results/answer_multiround_score/*.json` |
| 原始 answer 逐 round / 逐 case 分数 | `results/original_answer_round_sweep/round_*.json` |
| 原始 answer 日志 | `results/original_answer_round_sweep/round_*.log` |
| 强单轮逐 case 分数 | `results/answer_singleround_score.json` |
| 朴素三轮逐 case 分数 | `results/answer_singleround_to_multi_score.json` |
| main2 large 明细 | `results/main2_compare_latest_score.json` |
| 经典全枚举逐 case 上界 | `results/classical_compute_results.json` |
