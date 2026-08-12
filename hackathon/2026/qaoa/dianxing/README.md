# 点星：量子多目标组合优化赛道

本目录收录团队“点星”在 2026 MindSpore Quantum 量子多目标组合优化赛道中的参赛代码与论文。第一任务在固定量子采样预算内，以多 P 层数、多轮 warm-start QAOA 集成采样取得证据，再依据非支配前沿的 HV 贡献与覆盖情况分配后续采样种子；第二任务保持官方随机自旋流不变，对经典 Pareto 前沿后处理进行分块与线程并行加速。

## 文件

| 文件 | 说明 |
| --- | --- |
| `answer.py` | 参赛代码；官方评测入口为 `main1` 与 `main2` |
| `baseline.py` | 官方基线实现（单轮固定预算 QAOA / 串行前沿后处理），用于对照计分 |
| `utils.py` | 数据结构、QAOA 线路、目标计算、非支配筛选与 HV 工具 |
| `run.py` | 本地评测脚本；调用 `answer.main1` / `answer.main2`，输出 HV 增益与加速比并生成得分报告 |
| `transfer_data.csv` | 迁移参数列表（各 P 层的 β/γ 角度） |
| `data/public/` | 小规模公开数据集（k=5，4x5 网格，10 例） |
| `data/large/` | 大规模数据集（k=5，40x50 网格，10 例），用于任务二评测 |
| `决赛论文-量子组合优化-点星.md` | 方法、实验与复现说明（Markdown 版） |
| `决赛论文-量子组合优化-点星.pdf` | 方法、实验与复现说明（PDF 版） |
| `grid_topology.png` | 网格拓扑示意图 |
| `requirements.txt` | 复验使用的主要 Python 依赖版本 |



## 环境

 conda 环境 `pygmo_env`：Python 3.10.20、NumPy 2.2.6、SciPy 1.15.2、MindQuantum 0.12.0 与 pygmo 2.19.8。建议在 Python 3.10 环境中安装：

```bash
python -m pip install -r requirements.txt
```

或直接复用 conda 环境：

```bash
conda activate pygmo_env
```

## 运行

在本目录下运行评测脚本，将依次完成小规模数据集（任务一）与大规模数据集（任务二）的评测，

```bash
conda activate pygmo_env
python run.py --split all
```

常用参数：

```bash
python run.py --split public      # 仅评测公开小规模数据集
python run.py --split all         # 评测全部数据集（默认）
python run.py --large-shots 200000  # 任务二随机采样次数（默认 200000）
```

运行结束后终端会打印每个用例的 HV、基线 HV、HV 增益（任务一）与加速比（任务二），最终汇总得分见 `SUMMARY` 部分。
