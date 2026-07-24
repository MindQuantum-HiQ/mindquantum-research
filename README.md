# MindQuantum Research

本仓库是 [MindQuantum](https://atomgit.com/mindspore/mindquantum) 社区的研究内容仓库，用于集中存放与 MindQuantum 相关的研究性内容，包括：

- 基于 MindQuantum 的论文复现
- 黑客松 / 挑战赛选手的参赛作品
- 开源之夏（OSPP）等实习项目的成果归档

将这些体积较大、更新频率较低的研究资产与主仓库分离，可以保持 `mindquantum` 主仓库的克隆体积较小。

## 目录结构

| 目录 | 内容 | 命名规范 |
| --- | --- | --- |
| [`papers/`](papers/) | 论文复现项目 | `papers/<年份>_<论文简称>/` |
| [`hackathon/`](hackathon/) | 黑客松 / 挑战赛作品 | `hackathon/<年份>/<赛道名>/<队伍或选手>/` |
| [`summer_ospp/`](summer_ospp/) | 开源之夏项目成果 | `summer_ospp/<年份>/<项目编号>/` |

每个子目录内的 `README.md` 提供了该类别的项目索引与详细提交规范。

## 如何贡献

欢迎通过 Pull Request 提交论文复现或参赛作品，具体流程与要求请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

基本要求：

1. 按上表的命名规范创建目录，不要修改他人目录下的内容。
2. 每个项目目录必须包含一份 `README.md`（可参考各类别下的 `_template/` 模板）。
3. 提供 `requirements.txt` 或等价的依赖说明，保证代码可复现运行。
4. 不要提交敏感信息（账号、密钥、个人隐私数据等）。

## 相关链接

- MindQuantum 主仓库：<https://atomgit.com/mindspore/mindquantum>
- MindQuantum 文档：<https://www.mindspore.cn/mindquantum/docs/zh-CN/stable/index.html>

## 许可证

本仓库采用 [Apache-2.0](LICENSE) 许可证。向本仓库提交内容即表示你同意以该许可证发布你的代码。
