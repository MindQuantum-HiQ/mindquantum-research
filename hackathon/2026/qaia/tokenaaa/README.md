# TokenAAA / 量子启发算法赛道作品

本目录提交 2026 CCF 量子计算挑战赛“量子启发算法赛道”的参赛代码与论文。

## 文件清单

- `main.py`：参赛求解器代码。
- `README_运行说明.md`：原始提交说明，包含算法说明、依赖、运行命令与 FAQ。
- `requirements.txt`：运行代码所需依赖（官方判题脚本 `judger.py` 如需可再安装 `torch`）。
- `决赛论文-量子启发算法赛道-TokenAAA.pdf`：决赛论文 PDF。

## 运行方式

- Python 3.9+；先安装依赖：

  ```bash
  pip install -r requirements.txt
  ```

- 放置到官方环境（与 `judger.py`、`qaia/`、`Graph_data/` 可访问的同一目录），按提交说明执行：

  ```bash
  python3 main.py
  ```

或使用官方判题器：

  ```bash
  python3 judger.py
  ```

