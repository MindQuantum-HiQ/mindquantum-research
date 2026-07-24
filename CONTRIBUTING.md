# 贡献指南

感谢你为 MindQuantum Research 仓库贡献内容。本仓库接收两类主要贡献：论文复现（`papers/`）与黑客松参赛作品（`hackathon/`）。

## 提交流程

1. Fork 本仓库并克隆到本地。
2. 从 `main` 创建分支，例如 `add-papers-2024-adapt-qaoa`。
3. 按下述目录规范添加你的项目内容。
4. 提交 Pull Request，标题注明类别与项目名，例如：
   - `papers: add 2024_adapt_qaoa reproduction`
   - `hackathon: add 2025/quantum-hackathon/team_foo submission`
5. 等待维护者审核合并。

## 目录规范

### 论文复现（papers/）

```
papers/<年份>_<论文简称>/
├── README.md            # 必需，参考 papers/_template/README.md
├── requirements.txt     # 必需，依赖列表（含 mindquantum 版本）
├── src/ 或 *.ipynb      # 复现代码
└── results/             # 复现结果（图表、数据）
```

- `<论文简称>` 使用小写字母、数字和下划线，例如 `2024_adapt_qaoa`。
- README 中必须给出原论文链接（arXiv / DOI）以及复现结果与原文的对比。
- 添加完成后，请同步更新 `papers/README.md` 中的索引表。

### 黑客松作品（hackathon/）

```
hackathon/<年份>/<赛道名>/<队伍或选手>/
├── README.md
├── requirements.txt
└── src/ 或 *.ipynb      # 参赛代码
```

- `<赛事名>` 与 `<队伍或选手>` 使用小写字母、数字、连字符或下划线。
- 只在自己的队伍目录内提交内容，不要改动其他队伍的目录。

### 开源之夏（summer_ospp/）

OSPP 项目成果由项目开发者或维护者按 `summer_ospp/<年份>/<项目编号>/` 归档，要求与论文复现一致。

## 内容要求

- 代码可以基于仓库内声明的依赖版本运行；涉及随机性的实验请固定随机种子。
- 单个文件尽量不超过 50 MB。超大数据集或模型权重请提供外部下载链接（如 Release、云盘），不要直接提交。
- 不要提交敏感信息：账号、密钥、token、个人隐私数据等。
- 不要提交可由代码重新生成的中间产物（缓存、日志、checkpoint 等）。

## 许可证

本仓库采用 Apache-2.0 许可证。提交 Pull Request 即表示你同意你的贡献以该许可证发布。若你的作品包含第三方代码，请确保其许可证与 Apache-2.0 兼容，并在 README 中注明来源。
