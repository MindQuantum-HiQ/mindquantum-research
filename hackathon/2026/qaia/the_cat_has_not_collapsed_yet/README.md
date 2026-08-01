# The Cat Hasn't Collapsed Yet

“猫还没坍缩”是 2026 QAIA 黑客松作品，使用 QAIA 与经典局部搜索相结合的方法求解由单倍型定相问题转换得到的 Max-Cut 问题。

## 方法概述

求解器以 MindQuantum QAIA 算法族（CAC、DSB、NMFA 和 SimCIM）为主要搜索方法，并结合以下经典策略提高解的质量和运行效率：

- MST、强边同步和谱方法生成 warm start；
- 单点翻转、边二点翻转和相位块翻转进行稀疏局部优化；
- 精英候选共识、路径重联和近基准重启处理难例；
- 固定随机种子，保证相同环境中的结果可复现；
- 可选使用 Numba 加速局部搜索，不可用时自动回退到 NumPy 实现。

## 环境

- Python 3.9 或更高版本
- 依赖见 `requirements.txt`

安装依赖：

```bash
python -m pip install -r requirements.txt
```

## 运行

将赛事提供的图数据放入 `Graph_data/`。每个 `.txt` 文件的第一行为节点数、边数、baseline cut 和 baseline time，后续每行为一条从 1 开始编号的边：

```text
<num_nodes> <num_edges> <baseline_cut> <baseline_time>
<u> <v> <weight>
```

目录结构如下：

```text
the_cat_has_not_collapsed_yet/
├── Graph_data/
│   └── *.txt
├── README.md
├── main.py
└── requirements.txt
```

运行公开训练集评分：

```bash
python main.py
```

程序逐个输出 cut value、运行时间，并在最后打印加权准确率、时间效率和总分。图数据属于赛事输入，未包含在本提交中。

## 提交者

- GitHub: [YusenTan](https://github.com/YusenTan)
