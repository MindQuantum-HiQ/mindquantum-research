# 奶龙：量子启发算法赛道

本目录收录团队“奶龙”在 2026 MindSpore Quantum 量子启发算法赛道中的加权 Max-Cut 求解代码与论文。方法以块相位表示压缩搜索变量，使用 MindQuantum QAIA 的 BSB 搜索块间相位，再通过展开、单调 one-flip 修复与基线门控路线得到最终划分。

## 文件

| 文件 | 说明 |
| --- | --- |
| `main.py` | 参赛代码；官方评测入口为 `maxcut_solver` |
| `paper.pdf` | 方法、实验与复现说明 |
| `requirements.txt` | 复验使用的主要 Python 依赖版本 |

官方数据集与评测脚本未随本目录再分发。

## 环境

复验环境使用 Python 3.11.14、NumPy 1.26.4、SciPy 1.13.1、PyTorch 2.3.1（CPU）与 MindQuantum 0.12.0。建议在 Python 3.11 环境中安装：

```bash
python -m pip install -r requirements.txt
```

## 运行

1. 将 `main.py` 放入赛事官方模板根目录，与 `judger.py`、`Graph_data/` 同级。
2. 保持官方图数据的文件名与格式不变。
3. 运行官方评测；如需使用代码内的批量检查入口，可执行：

```bash
python main.py
```

官方评测以 `judger.py` 调用 `maxcut_solver` 为准。函数返回长度与图顶点数一致、元素仅为 `-1` 或 `+1` 的一维划分。BSB 路线使用代码中固定的随机种子，便于复现。

## 来源与许可

代码基于赛事官方模板接口开发，不包含官方数据集或评测脚本。本目录随父仓库按 Apache-2.0 许可证发布；第三方依赖遵循各自许可证。
