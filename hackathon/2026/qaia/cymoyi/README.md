# 代码一行没看：AQMF 量子启发式 Max-Cut 求解器

本目录收录团队“代码一行没看”在 2026 量子启发算法赛道使用的最终求解器。提交版本为 v1-aqmf-fastclean，核心方法为自写的异步量子均值场（Asynchronous Quantum Mean Field，AQMF）连续态退火求解器。

## 方法概述

AQMF 将 Max-Cut 写为 Ising 型二值优化问题，并以连续振幅状态和局部有效场进行异步更新。主搜索包含退火泵浦、阻尼演化、非线性饱和及 tanh 二值测量，因此候选划分由量子启发式连续动力学产生，而非由纯经典贪心搜索直接构造。

对于主路径未达到目标阈值的风险样本，程序会在官方模板提供的 qaia 环境中调用 NMFA、BSB 或 DSB 进行子图量子启发式细化。经典逻辑仅用于图窗口组织、候选打分和有严格上限的 one-flip 收尾，不作为独立主求解器。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| main.py | 参赛求解器，官方调用入口为 maxcut_solver。 |
| requirements.txt | 本地复现所用的核心依赖版本。 |
| 决赛论文-量子启发算法-代码一行没看.pdf | 与本提交版对应的赛题论文。 |

本目录不包含赛题的 Graph_data、judger.py 或模板自带的 qaia 目录。它们应使用组织方发放的原始模板版本，且不应改动模板中标记为“禁止改动”的函数。

## 环境与运行

推荐 Python 3.11，并先执行 python -m pip install -r requirements.txt。将本目录的 main.py 放入组织方原始模板根目录，与 judger.py、Graph_data 和模板的 qaia 同级，然后执行 python judger.py。官方评分以 judger.py 调用 maxcut_solver 的结果为准。

## 合规与复核说明

- main.py 中的 read_graph_file、scipy_to_torch_sparse、update_sparse_matrix_weights、get_smaller_subset 和 calculate_cut_value 与原始模板保持一致。
- 所有求解相关计算均在 maxcut_solver 内进行；代码不在模块导入或读图阶段提前读取图、求解或缓存 partition。
- 不含答案表、文件名分支、基于节点数/边数/baseline 的实例识别、网络访问、Numba 预编译或外部数据下载逻辑。cut_value_baseline 只用作早停阈值。
- 返回结果为长度与图节点数一致的一维 int8 数组，元素严格属于 {-1, +1}。
- 不随提交携带训练数据、判题器、日志、缓存、模型权重或其他可再生中间产物。

本代码随仓库以 Apache-2.0 许可证发布。官方模板和 MindQuantum/QAIA 组件的许可证与使用条件遵循其各自的上游说明。
