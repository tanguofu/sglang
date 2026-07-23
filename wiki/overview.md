---
title: Overview
category: entity
tags: [overview, index]
sources: []
created: 2026-07-19
updated: 2026-07-19
---

# Overview

本 wiki 记录 TI-Cloud 团队在 GPU 推理服务、操作系统、驱动安装、网络配置等方面的工程知识。内容由 `/teamai-wiki` skill 维护，通过 `teamai push/pull` 与团队同步，可在 Obsidian 中浏览。

## 主要主题

### 1. TOS 4.4 节点驱动安装

TencentOS Server 4.4 (kernel 6.6) 相比 TOS 3.1 (kernel 5.4) 在第三方驱动兼容性上有显著差异。配备 AMD MI308X GPU 和 Broadcom BCM57608 RoCE NIC 的节点需要完整重建驱动栈：

- **GPU 驱动** — 装 DKMS amdgpu 6.16.13 + MOK 签名 + 移除 `module.sig_enforce=1` + ROCm 7.2.4。参见 [[install-amdgpu-driver-tos44]]。
- **NIC/RDMA 驱动** — 装 DKMS bnxt_en/bnxt_re 238 + 源码编译 libbnxt_re + niccli + 升级 NIC 固件 + 重建 initramfs。参见 [[install-bnxt-re-rdma-driver-tos44]]。
- **版本决策** — 用 bnxt_re 238 而不是与 TOS 3.1 一致的 233，因为 233 源码不兼容 kernel 6.6 API。参见 [[use-bnxt-re-238-on-tos44]]。

### 2. 关键硬件

- [[bcm57608-nic]] — Broadcom BCM57608 双口 200G RoCE NIC（P2200G 标准版，非 PTP）
- [[amdgpu-driver]] — AMD MI308X (gfx942) 配套内核驱动

### 3. 关键概念

- [[dkms-module-signing]] — 在 Secure Boot 系统上为 DKMS 编译的 out-of-tree 模块签名
- [[initramfs-rebuild]] — DKMS 升级模块后必须 `dracut -f` 重建 initramfs，否则旧版优先加载

## 验证节点

| 节点 | 操作系统 | GPU 驱动 | NIC 驱动 | RDMA 带宽 |
|---|---|---|---|---|
| 170.159 | TOS 4.4 | amdgpu 6.16.13 | bnxt_re 238 | 183 Gb/sec |
| 171.87 | TOS 4.4 | amdgpu 6.16.13 | bnxt_re 238 | 183 Gb/sec |
| 170.19 | TOS 3.1 | （已存在） | bnxt_re 233 | （未测） |
| 170.32 | TOS 3.1 | （已存在） | bnxt_re 233 | （未测） |

## 遗留文档

wiki 根目录下的 `glm52-0702-amd-optimization.md` 和 `glm52-0702-amd-optimization-next-steps.md` 是 wiki 初始化前已存在的独立文档，记录 GLM-5.2 在 AMD MI355X 上的优化工作。后续如需引用，可迁移至 `sources/` 或 `decisions/`。

## Related

- [[tos-44]] — TOS 4.4 操作系统实体
- [[install-amdgpu-driver-tos44]] — GPU 驱动安装流程
- [[install-bnxt-re-rdma-driver-tos44]] — NIC 驱动安装流程
- [[use-bnxt-re-238-on-tos44]] — 版本选择决策

## Backlinks
