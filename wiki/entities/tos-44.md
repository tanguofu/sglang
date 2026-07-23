---
title: TencentOS Server 4.4
category: entity
tags: [os, tencent, kernel-6.6, tos-44]
sources: []
created: 2026-07-19
updated: 2026-07-19
---

# TencentOS Server 4.4

TencentOS Server 4.4 (TOS 4.4) 是腾讯内部基于 Linux 6.6 内核的服务器操作系统，主要部署在 GPU/加速器节点。相比 TOS 3.1（kernel 5.4.119-19.0009.60），TOS 4.4 升级到 kernel 6.6.110-42.4.tl4.x86_64，带来更新的调度器、io_uring、BPF 等特性，但对第三方内核模块（amdgpu、bnxt_re 等）的兼容性需要重新编译。

## 关键特性

- **内核版本**: `6.6.110-42.4.tl4.x86_64`（LTS 6.6 系列）
- **Secure Boot**: 默认开启，out-of-tree 内核模块必须用 MOK 签名才能加载
- **module.sig_enforce**: 部分构建默认开启 `module.sig_enforce=1`，需要从 GRUB 启动参数移除
- **in-tree bnxt_en**: 233.0.152.14（旧版，不支持 BNXT RoCE API 新符号）
- **in-tree amdgpu**: 不含 MI308X 的 ip discovery binary，导致 `invalid ip discovery binary checksum`
- **MLNX OFED**: 24.10（rdma-core 提供 `IBVERBS_PRIVATE_34`，不是 v57）
- **腾讯 RDMA 脚本**: `/usr/local/qcloud/rdma/` 全部存在并启用 `rdma-agent.service`、`RDMA_FastStart.service`

## 已知问题

1. **in-tree amdgpu 不支持 MI308X** — 需装 DKMS 6.16.13（参见 [[install-amdgpu-driver-tos44]]）
2. **完全没有 bnxt_re 模块** — 需装 DKMS 238（参见 [[install-bnxt-re-rdma-driver-tos44]]）
3. **旧版 bnxt_re 233 源码不兼容 kernel 6.6 API** — 必须用 238 或更新版本
4. **libbnxt_re 预编译 RPM 依赖 IBVERBS_PRIVATE_57** — 必须从源码编译生成 `rdmav34.so`

## 与 TOS 3.1 对比

| 维度 | TOS 3.1 | TOS 4.4 |
|---|---|---|
| 内核 | 5.4.119-19.0009.60 | 6.6.110-42.4.tl4.x86_64 |
| bnxt_en | out-of-tree 233 | in-tree 233（需升级到 238） |
| bnxt_re | out-of-tree 233 可用 | 不存在（必须装 DKMS 238） |
| amdgpu | 5.x 系列 | in-tree 不支持 MI308X |
| Secure Boot | 部分开启 | 默认开启 + sig_enforce |

## 部署节点

- 170.159（2026-07-19 完成 GPU + NIC 驱动修复）
- 171.87（2026-07-19 完成 GPU + NIC 驱动修复）

## Related

- [[install-amdgpu-driver-tos44]] — GPU 驱动安装流程
- [[install-bnxt-re-rdma-driver-tos44]] — NIC/RDMA 驱动安装流程
- [[amdgpu-driver]] — AMD GPU 驱动
- [[bnxt-re-driver]] — Broadcom RoCE 驱动
- [[bcm57608-nic]] — Broadcom 200G NIC 硬件

## Backlinks
