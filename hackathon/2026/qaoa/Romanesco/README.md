# 罗马花椰菜团队 MOO 决赛材料说明

本目录是罗马花椰菜团队在量子多目标优化（MOO）赛题中的决赛论文 PDF、实验代码和复现实验记录集合。目录内代码以论文复现和方案对照为目标，不覆盖组委会提供的 `MOO/answer.py`、`MOO/baseline.py`、`MOO/run.py`、`MOO/run_origin.py`、`MOO/utils.py` 等官方文件。

代码入口均通过 `Path(__file__).resolve()` 从自身位置向上定位官方 `MOO/` 根目录，因此推荐将本目录作为官方 `MOO/` 的直接子目录整体打包、移动和运行。

## 1. 放置方式

推荐保持如下结构：

```text
MOO/
├── answer.py                         # 官方提供，保持不变
├── baseline.py                       # 官方提供，保持不变
├── run.py                            # 官方评测入口，保持不变
├── run_origin.py                     # 官方原始评测流程，round sweep 使用
├── utils.py                          # 官方工具函数，保持不变
├── transfer_data.csv                 # 官方 QAOA transfer angles
├── README.md                         # 官方赛题说明
├── data/
│   ├── w_pool_k5_n1000_seed2026.json
│   ├── public/                       # public10 小规模 4x5, k=5 数据
│   └── large/                        # large 40x50, k=5 数据
├── results/                          # 官方评测缓存/输出，可由 run.py 生成
└── Romanesco/                        # 本决赛材料目录
    ├── README.md                     # 本说明文件
    ├── requirements.txt              # conda huawei_2026 的 pip freeze 快照
    ├── experiment_results.md         # 实验分数、耗时和结果文件索引
    ├── code/                         # 实验代码
    │   ├── answer_singleround.py
    │   ├── answer_singleround_to_multi.py
    │   ├── answer_multiround.py
    │   ├── run_answer.py
    │   ├── run_original_answer_rounds.py
    │   ├── main2_compare.py
    │   └── classical_compute.py
    ├── picture/                      # 论文图表和绘图脚本
    │   ├── generate_figures.py
    │   ├── answer_round_sweep.png
    │   ├── coord_round_sweep.png
    │   └── main2_runtime_comparison.png
    ├── results/                      # 已完成实验 JSON 和汇总表
    │   ├── answer_singleround_score.json
    │   ├── answer_singleround_to_multi_score.json
    │   ├── answer_multiround_score/
    │   ├── original_answer_round_sweep/
    │   ├── main2_compare_latest_score.json
    │   └── classical_compute_results.json
    └── 决赛论文-moo-罗马花椰菜.pdf         # 当前导出的 PDF 版本
```

若移动本目录，请保持它仍为官方 `MOO/` 根目录的直接子目录。否则 `code/` 内脚本无法可靠找到官方 `utils.py`、`baseline.py`、`transfer_data.csv` 和 `data/`。

## 2. 内容概览

| 文件或目录 | 作用 |
| --- | --- |
| `README.md` | 本说明文件，记录目录结构、环境、代码对应关系和复现命令。 |
| `requirements.txt` | `conda run -n huawei_2026 python -m pip list --format=freeze` 快照，Python 版本 `3.9.7`。 |
| `experiment_results.md` | 实验结果速查：当前最高 public10 分数、轮数消融、main2 速度对照、经典上界和结果文件索引。 |
| `code/` | 方案代码、复现实验脚本和分析脚本。 |
| `picture/` | 论文图表 PNG 和图表生成脚本。 |
| `results/` | 本地 public10/large 评测结果、round sweep JSON 汇总和论文表格依据。 |
| `决赛论文-moo-罗马花椰菜.pdf` | 当前导出的决赛论文 PDF 版本。 |

## 3. 代码文件对应关系

`code/` 中的 `answer_*.py` 是候选方案或对照方案，不会自动替换官方 `MOO/answer.py`。如需正式提交，应由人工选择目标版本并复制/改名为提交用 `answer.py`。

| 文件 | 定位 | 主要用途 |
| --- | --- | --- |
| `code/answer_singleround.py` | 强单轮 mixed-depth QAOA ensemble | 提供 `main1` 和向量化 `main2`；作为强单轮主对照。public10 本地结果见 `results/answer_singleround_score.json`。 |
| `code/answer_singleround_to_multi.py` | 强单轮朴素改三轮 warm-start | 用于证明“直接增加轮次”不一定有效。结果见 `results/answer_singleround_to_multi_score.json`。 |
| `code/answer_multiround.py` | 多轮协同设计代表方案 | 默认 `MOO_ROUND_SCHEDULE=4round_2`，shots 为 `[400, 400, 150, 50]`；可用环境变量切换 `1round` 至 `6round` 消融。主结果见 `results/answer_multiround_score/answer_multiround_score_4round_2.json`。 |
| `code/run_answer.py` | 包内本地评分入口 | 复刻官方评分关键检查，当前 import switchboard 默认加载 `answer_multiround.py`，输出到包内 `results/`。 |
| `code/run_original_answer_rounds.py` | 官方原始 `MOO/answer.py` 轮数消融 | 运行时 monkey patch 原始模块的 `N_ROUNDS` 和 `SHOTS_PER_WEIGHT`，不改写官方文件。 |
| `code/main2_compare.py` | `main2` 后处理三方案速度对照 | 比较官方参考、向量化 pipeline、Numba refinement，并验证 HV/frontier 一致性。 |
| `code/classical_compute.py` | public10 经典全枚举上界 | 枚举 20 比特 public cases 的全部 `2^20` 自旋构型，仅用于论文分析，不是提交算法。 |

## 4. 结果文件对应关系

| 文件或目录 | 内容 |
| --- | --- |
| `results/answer_singleround_score.json` | 强单轮方案 public10 + large 本地评分结果。 |
| `results/answer_singleround_to_multi_score.json` | 强单轮朴素三轮 warm-start 对照结果。 |
| `results/answer_multiround_score/answer_multiround_score_1round.json` | 协调设计 1 轮 `[1000]` 消融结果。 |
| `results/answer_multiround_score/answer_multiround_score_2round.json` | 协调设计 2 轮 `[800, 200]` 消融结果。 |
| `results/answer_multiround_score/answer_multiround_score_3round.json` | 协调设计 3 轮 `[600, 200, 200]` 消融结果。 |
| `results/answer_multiround_score/answer_multiround_score_4round.json` | 协调设计 4 轮 `[400, 200, 200, 200]` 消融结果。 |
| `results/answer_multiround_score/answer_multiround_score_4round_2.json` | 当前本地最高协调设计 4 轮 `[400, 400, 150, 50]` 结果。 |
| `results/answer_multiround_score/answer_multiround_score_5round.json` | 协调设计 5 轮 `[300, 200, 200, 150, 150]` 消融结果。 |
| `results/answer_multiround_score/answer_multiround_score_6round.json` | 协调设计 6 轮 `[250, 150, 150, 150, 150, 150]` 消融结果；重跑后仍低于 4/5 轮质量。 |
| `results/original_answer_round_sweep/round_*.json` | 未修改官方 `MOO/answer.py` 的 1 至 6 轮消融 JSON。 |
| `results/original_answer_round_sweep/summary.tsv` | 原始 `answer.py` 1 至 6 轮消融汇总表。 |
| `results/main2_compare_latest_score.json` | 官方参考、向量化、Numba refinement 的 `main2` 速度和一致性对照。 |
| `results/classical_compute_results.json` | public10 全枚举 HV 理论上界。 |

实验分数和结论以 `experiment_results.md` 为入口。该文件记录当前本地 public10 最高总分为 `coord_4round_2` 的 `238.014593`，强单轮方案为 `237.489343`。云端分数以实际提交记录为准，本地 public10 结果不代表 hidden/all。

## 5. 论文与图表文件

| 文件或目录 | 作用 |
| --- | --- |
| `决赛论文-moo-罗马花椰菜.pdf` | 当前导出的 PDF 版本。 |
| `picture/generate_figures.py` | 论文图表生成脚本。 |
| `picture/answer_round_sweep.png` | 官方原始方案 1 至 6 轮 round sweep 图。 |
| `picture/coord_round_sweep.png` | 多轮协调设计 1 至 6 轮 round sweep 图。 |
| `picture/main2_runtime_comparison.png` | `main2` 三方案 large case 平均耗时对比图。 |

## 6. 代码环境

本地实验环境如下：

| 项目 | 版本/说明 |
| --- | --- |
| Conda 环境 | `huawei_2026` |
| Python | `3.9.7` |
| MindSpore | `2.8.0` |
| MindSpore Quantum | `0.12.0` |
| NumPy | `1.24.2` |
| SciPy | `1.10.1` |
| pygmo | `2.19.5`，由 Conda 环境提供，可能不出现在 `pip freeze` 中 |
| Numba | `0.60.0` |
| Matplotlib | `3.9.4`，辅助分析依赖 |
| pandas | `2.3.3`，辅助分析依赖 |

完整 pip 包版本见 `requirements.txt`。建议先进入官方 `MOO/` 根目录再运行复现实验：

```bash
conda activate huawei_2026
cd MOO
```

如需尽量复现线程设置，可保持脚本内默认限制：`OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS` 通常被设为 1 或 2，以适配 2 核 CPU 约束。

## 7. 快速运行

以下命令默认当前目录为官方 `MOO/` 根目录。复现实验有两种入口：一是使用官方 `run.py` 搭配 `PYTHONPATH` 直接评测候选 answer 模块；二是使用本目录提供的 `code/run_answer.py`，它内置了评分复现流程和候选方案 import switchboard，适合论文实验记录。

### 7.1 用官方 `run.py` 评测候选 answer 模块

通过 `PYTHONPATH` 临时加入包内 `code/`，再用官方 `--answer` 选择模块：

```bash
PYTHONPATH="Romanesco/code${PYTHONPATH:+:$PYTHONPATH}" \
python run.py --split public --answer answer_singleround \
  --out "Romanesco/results/answer_singleround_reproduced.json"
```

其他候选方案只需替换模块名：

```bash
# 强单轮朴素三轮 warm-start 对照
PYTHONPATH="Romanesco/code${PYTHONPATH:+:$PYTHONPATH}" \
python run.py --split public --answer answer_singleround_to_multi

# 多轮协同设计，默认使用 4round_2=[400,400,150,50]
PYTHONPATH="Romanesco/code${PYTHONPATH:+:$PYTHONPATH}" \
python run.py --split public --answer answer_multiround
```

快速检查建议先限制案例数：

```bash
PYTHONPATH="Romanesco/code${PYTHONPATH:+:$PYTHONPATH}" \
python run.py --split public --max-cases 1 --answer answer_multiround
```

完整 `public10` 或 `all` 评测耗时较长，请预留足够时间：

```bash
PYTHONPATH="Romanesco/code${PYTHONPATH:+:$PYTHONPATH}" \
python run.py --split all --answer answer_multiround
```

### 7.2 切换 `answer_multiround.py` 的轮数 schedule

`answer_multiround.py` 支持通过环境变量切换消融配置：

```bash
MOO_ROUND_SCHEDULE=3round PYTHONPATH="Romanesco/code${PYTHONPATH:+:$PYTHONPATH}" \
python run.py --split public --answer answer_multiround
```

可选值包括：`1round`、`2round`、`3round`、`4round`、`4round_2`、`5round`、`6round`。

### 7.3 使用包内评分入口

`run_answer.py` 是除官方 `run.py` 之外的包内复现入口，当前默认加载 `answer_multiround.py`。它会从脚本路径定位官方 `MOO/`，并把结果写入本目录 `results/`。若当前目录为官方 `MOO/` 根目录，可运行：

```bash
conda run -n huawei_2026 python \
  "Romanesco/code/run_answer.py" \
  --split public --max-cases 1
```

若要改评测对象，请编辑 `code/run_answer.py` 顶部 import switchboard，仅切换 `import answer_multiround as answer` 等候选导入；不要改官方 `MOO/answer.py`。

### 7.4 官方原始方案 round sweep

该脚本运行时覆盖原始模块内轮数参数，不改写官方 `MOO/answer.py` 文件：

```bash
python "Romanesco/code/run_original_answer_rounds.py" \
  --rounds 5 --max-cases 10 \
  --out "Romanesco/results/original_answer_round_sweep/round_5_reproduced.json"
```

### 7.5 `main2` 三方案速度对照

快速验证一个 large case：

```bash
python "Romanesco/code/main2_compare.py" \
  --no-prompt --max-cases 1 --shots 200000 \
  --out "Romanesco/results/main2_compare_reproduced.json"
```

默认数据目录由脚本相对定位为官方 `MOO/data/large/`。也可用 `--data-dir` 显式指定其它兼容数据目录。

### 7.6 public10 经典全枚举上界

该命令枚举每个 20 比特 public case 的全部 `2^20` 个自旋配置，仅用于论文分析，耗时明显高于快速检查：

```bash
python "Romanesco/code/classical_compute.py" \
  --with-baseline \
  --out "Romanesco/results/classical_compute_reproduced.json"
```

仅验证一个案例：

```bash
python "Romanesco/code/classical_compute.py" \
  --max-cases 1 --chunk-size 65536
```
