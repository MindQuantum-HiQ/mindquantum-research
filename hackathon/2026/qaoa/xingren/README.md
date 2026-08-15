# 星韧｜量子组合优化赛道

本目录为 **星韧** 队参加 2026 CCF
量子计算编程挑战赛（昇思杯）**量子组合优化赛道**的决赛提交材料，包括算法代码、运行环境说明及决赛论文。

## 1. 提交内容

``` text
xingren/
├── README.md
├── requirements.txt
├── code/
│   └── answer.py
├── 决赛论文-量子组合优化赛道-星韧.pdf
└── 决赛论文-量子组合优化赛道-星韧.docx
```

其中：

-   `code/answer.py`：本队最终提交的核心算法代码；
-   `requirements.txt`：代码运行所需 Python 依赖；
-   `决赛论文-量子组合优化赛道-星韧.pdf`：决赛论文 PDF 版本；
-   `决赛论文-量子组合优化赛道-星韧.docx`：决赛论文 Word 版本。

## 2. 环境与依赖

代码基于 **MindSpore Quantum 0.12.0** 实现，运行时需要 NumPy、pygmo
等依赖。

推荐按照本目录提供的 `requirements.txt` 安装依赖：

``` bash
pip install -r requirements.txt
```

评测环境以赛方提供的 **2 核 CPU、4 GB 内存**环境为准。

## 3. 评测文件放置

本仓库仅提交本队需要提交的核心算法文件
`answer.py`，赛方提供的原始评测文件不重复提交。

进行复现或评测时，请将：

``` text
code/answer.py
```

复制至赛方提供的评测目录中，与赛方评测文件放在同一目录。

最终评测目录示例如下：

``` text
赛题目录/
├── answer.py
├── run.py
├── utils.py
├── transfer_data.csv
├── baseline.py          # 赛方评测包中包含时保留
├── data/
└── results/
```

其中，`run.py`、`utils.py`、`transfer_data.csv`
和测试数据均使用赛方原始文件，无需修改。

## 4. 运行方法

### 公开测试集

``` bash
python run.py --split public --out results/latest_score_public.json
```

运行结果保存至：

``` text
results/latest_score_public.json
```

### 完整评测

``` bash
python run.py --split all --large-shots 200000 --out results/latest_score_all.json
```

运行结果保存至：

``` text
results/latest_score_all.json
```

## 5. 算法接口与预算

### `main1()`

`main1()` 用于小规模五目标 Ising 问题，总量子采样预算为 **100000
shots**。

返回的 `sample_spins` 形状为：

``` text
(100000, n)
```

其中元素取值为 `-1` 或
`+1`。代码中的多轮采样和特征门控均保持总预算不变。

### `main2()`

`main2()` 用于大规模 Ising 问题，默认随机样本数为 **200000**。

评测时需保证返回的 HyperVolume、非支配前沿和非支配点数量正确。

## 6. 说明

`answer.py` 仅调用赛方提供的评测接口和辅助文件，不修改官方评测流程。

经典部分主要用于：

-   样本评价；
-   样本去重；
-   非支配筛选；
-   warm-start 种子选择。

最终提交样本仍由量子线路采样产生。

更完整的算法设计、实验结果与分析请参阅本目录中的决赛论文。
