# 奶龙：量子多目标组合优化赛道

本目录收录团队“奶龙”在 2026 MindSpore Quantum 量子多目标组合优化赛道中的参赛代码与论文。第一任务在固定量子采样预算内，以 QAOA 试探采样取得证据，再依据非支配 archive 的贡献与覆盖情况分配后续预算；第二任务保持官方随机自旋流不变，对经典 Pareto 前沿后处理进行分块与并行加速。

## 文件

| 文件 | 说明 |
| --- | --- |
| `answer.py` | 参赛代码；官方评测入口为 `main1` 与 `main2` |
| `utils.py` | 数据结构、QAOA 线路、目标计算、非支配筛选与 HV 工具 |
| `paper.pdf` | 方法、实验与复现说明 |
| `requirements.txt` | 复验使用的主要 Python 依赖版本 |

官方数据集与评测脚本未随本目录再分发。

## 环境

复验环境使用 Python 3.11、NumPy 1.26.4、SciPy 1.11.4、MindQuantum 0.12.0 与 pygmo 2.19.5。NumPy 固定为 1.26.4，以匹配本次 pygmo 前沿复验环境。建议在 Python 3.11 环境中安装：

```bash
python -m pip install -r requirements.txt
```

## 运行

1. 将 `answer.py` 与 `utils.py` 放入赛事官方模板根目录，与 `run.py` 同级。
2. 保持模板中的 `data/`、评测脚本与缓存路径不变。
3. 按官方模板运行：

```bash
python run.py --split public
python run.py --split all
```

`main1(problem_input, sample_budget=100000, rng_seed=None)` 返回采样使用量和量子采样自旋；`main2(problem_input, shots=200000, rng_seed=None, chunk_size=4096)` 返回归一化非支配前沿、HV、采样数与运行时间。两个入口在未传入 `rng_seed` 时均使用固定种子 `2026`。

## 来源与许可

代码基于赛事官方模板接口开发，不包含官方数据集或评测脚本。本目录随父仓库按 Apache-2.0 许可证发布；第三方依赖遵循各自许可证。
