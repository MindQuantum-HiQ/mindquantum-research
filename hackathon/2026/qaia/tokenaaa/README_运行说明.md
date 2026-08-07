# README_运行说明

## 1. 提交文件说明

本材料面向 2026 CCF 量子计算挑战赛决赛“量子启发算法赛道”。最终 zip 根目录仅保留以下 4 个正式提交文件：

- `决赛论文-量子启发算法赛道-TokenAAA.docx`
- `决赛论文-量子启发算法赛道-TokenAAA.pdf`
- `main.py`
- `README_运行说明.md`

本文和论文中统一将最终算法命名为“签名森林热启动与修复增强的离散模拟分岔算法”，英文缩写为 `SF-DSB-R`。`Baseline-SB` 指官方样例代码复制得到的基线程序；`SF-DSB-R` 指最终提交文件 `main.py` 中实现的改进算法。

论文中的实验图表数据由作者在公开 `Graph_data` 上本地运行生成；正式提交包仅包含上列 4 个文件，不包含中间实验材料。

## 2. 环境依赖

推荐环境：

- Python 3.9 或以上；
- `numpy`；
- `scipy`；
- `torch`，仅官方 `judger.py` 验证时需要；
- 赛题官方提供的 `qaia/` 包；
- 赛题官方提供的 `Graph_data/` 数据。

安装示例：

```bash
python3 -m pip install numpy scipy torch
```

如果赛题运行环境已内置上述依赖，无需重复安装。

## 3. 推荐目录结构

运行验证时，请将 `main.py` 与赛题官方 `judger.py`、`qaia/`、`Graph_data/` 放在同一目录，或保持 Python 可导入路径一致。

```text
submission_root/
├── main.py
├── README_运行说明.md
├── judger.py
├── qaia/
└── Graph_data/
    ├── block5.txt
    ├── block6.txt
    └── ...
```

## 4. 运行命令

直接运行求解器：

```bash
python3 main.py
```

使用官方判题器验证：

```bash
python3 judger.py
```

评委复现提交代码时，只需将 `main.py` 放入赛题官方环境，并按上文运行 `main.py` 或官方 `judger.py`。

## 5. 输入输出说明

输入图文件位于 `Graph_data/`。每个图文件首行包含节点数、边数、baseline cut value 和 baseline time，后续每行为一条带权无向边。`main.py` 会读取图并构造 scipy CSR 稀疏矩阵。

`maxcut_solver(G, max_iterations, baseline)` 返回长度为节点数的 `numpy.ndarray`，元素为 `-1` 或 `1`，表示 Max-Cut/Ising 二值自旋解。官方 `judger.py` 会根据该自旋向量计算 cut value、AccRatio、运行时间和最终 score。

## 6. 算法概要

`main.py` 的正式求解流程如下：

1. 读取稀疏图并提取唯一边表；
2. 根据图规模和边密度筛选强边；
3. 用强边构造签名森林，传播边符号关系生成热启动；
4. 对不连通森林分量执行压缩图符号对齐；
5. 以热启动初始化 QAIA DSB，执行量子启发式精炼；
6. 达到公开样例 baseline 时提前返回；
7. 必要时执行单点翻转局部下降；
8. 执行基于树路径差分的子树级块翻转修复；
9. 再次用 DSB 从修复状态出发精炼并输出最终解。

论文中的“签名森林热启动”“分量符号对齐”“DSB-局部修复-DSB 两阶段精炼”和“子树块翻转修复”均对应 `main.py` 中真实存在的实现。代码未实现 tabu/refill、多轮显式重加权、完整社区分块，也未复现 QHap 原始真实测序数据集。

论文图 1 的修复增强模块是条件分支：只有当初始 DSB 未达到公开基线时，才触发 single-spin descent、subtree refinement 和最终 DSB 重启。该图仅描述赛题 Max-Cut/Ising 图实例上的 `SF-DSB-R` 求解流程。

论文参考文献后新增“附录A 算法实现与补充材料”，用于汇总 `main.py` 函数对应关系、关键公式与参数、复现命令、消融边界和 QHap 公平对比条件。附录不新增未运行实验结果，也不改变正文实验数据和结论。


## 7. 本地公开样例验证

本次论文整理了公开 `Graph_data` 的 12 个图全量验证。官方 `judger.py` 对最终 `main.py` 的本地输出为：

- Weighted Acc Score: `1.0001`
- Total Solve Time: `0.73 s`
- Time Efficiency Score: `0.9985`
- Final Score: `99.8599`

该值为作者使用官方 `judger.py` 在公开 `Graph_data` 上的本地验证输出。本地多次复跑的 total solve time 在约 `0.73-0.75 s` 小范围波动，因此论文正文按约值描述为“约 99.86、约 0.73 秒”。

论文图 3 的代表性收敛过程覆盖 `block6.txt`、`block18.txt`、`block23.txt`、`block31.txt` 四个公开图，以及 `DSB`、`BSB`、`SimCIM`、`NMFA`、`LQA`、`SF-DSB-R` 六种方法。该结果是单次运行过程示例，用于说明趋势，不作为统计显著性结论。

上述结果仅为本地公开样例验证，不能等同于官方隐藏评测成绩。不同硬件、Python 版本和依赖版本会导致时间分数略有波动。

本地实验环境记录：

- macOS 15.6.1 arm64；
- Apple M4 Pro；
- Python 3.13.0；
- numpy 2.2.6；
- scipy 1.15.2；
- torch 2.7.0。

## 8. 常见问题

### 8.1 `ModuleNotFoundError: No module named 'qaia'`

请确认赛题官方 `qaia/` 文件夹与 `main.py` 在同一目录，或已加入 `PYTHONPATH`。

### 8.2 `Graph_data` 找不到

请确认运行命令所在目录包含 `Graph_data/`。若数据在其他位置，请先切换到包含 `Graph_data/` 的目录。

### 8.3 `judger.py` 缺少 torch

`main.py` 不直接依赖 torch，但官方 `judger.py` 使用 torch 计算评分。可执行：

```bash
python3 -m pip install torch
```

### 8.4 本地结果与论文表格略有不同

论文表格来自同一批本地公开样例实验，但运行时间会随系统负载、Python 版本和依赖版本变化。若重新验证，应以同一硬件和同一依赖版本下的 `judger.py` 输出为准。
