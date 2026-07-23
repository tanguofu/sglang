# Wiki Schema

本 wiki 由 `/teamai-wiki` skill 自动维护。

## 页面分类

| 分类 | 目录 | 说明 | 示例 |
|------|------|------|------|
| entity | entities/ | 专有名词：具体的模块、服务、产品、项目、硬件 | tos-44, bcm57608-nic, amdgpu-driver |
| concept | concepts/ | 通用名词：设计模式、架构原则、技术概念 | dkms-module-signing, initramfs-rebuild |
| comparison | comparisons/ | 两个或多个事物的对比分析 | tos-3-vs-tos-4-rdma |
| person | people/ | 团队成员、专长领域、负责模块 | _暂无_ |
| decision | decisions/ | 架构决策记录、技术选型、变更原因 | use-bnxt-re-238-on-tos44 |
| process | processes/ | 工作流、SOP、部署/发布/on-call 流程 | install-amdgpu-driver-tos44, install-bnxt-re-rdma-driver-tos44 |
| source | sources/ | 源文件/文档的结构化摘要 | _暂无_ |
| query | queries/ | 有价值的查询结果 | _暂无_ |

## 命名规则

- 文件名使用 kebab-case：`install-amdgpu-driver-tos44.md`
- 标题使用人类可读的格式：`Install amdgpu Driver on TOS 4.4`
- Wiki 链接使用文件名（不含 .md）：`[[install-amdgpu-driver-tos44]]`

## 链接格式

- 内部引用：`[[page-name]]` — 仅使用文件名，不含目录前缀
- 带描述：`[[page-name]] — 简要说明`
- 跨分类引用同样有效，链接是全局唯一的

## 页面模板

每个页面必须包含：
1. YAML frontmatter（title, category, tags, sources, created, updated）
2. 正文内容
3. Related 段落（出链）
4. Backlinks 段落（入链，由系统自动维护）

## 遗留文档

`wiki/` 根目录下的 `glm52-0702-amd-optimization.md` 和 `glm52-0702-amd-optimization-next-steps.md` 是本 wiki 初始化前已存在的独立文档，未纳入分类目录。后续如需引用，可迁移至 `sources/` 或 `decisions/`。
