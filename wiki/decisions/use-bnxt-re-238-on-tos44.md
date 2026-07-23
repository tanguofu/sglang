---
title: Use bnxt_re 238 on TOS 4.4 (not 233)
category: decision
tags: [decision, driver-version, bnxt-re, tos-44, kernel-6.6]
sources: []
status: accepted
date: 2026-07-19
created: 2026-07-19
updated: 2026-07-19
---

# Use bnxt_re 238 on TOS 4.4 (not 233)

在 TOS 4.4 (kernel 6.6.110) 节点上安装 bnxt_re 238.1.138.5 而不是与 TOS 3.1 正常节点一致的 233.0.152.14。

## 背景

170.159 和 171.87 两台节点是 TOS 4.4 (kernel 6.6)，配备 BCM57608 RoCE NIC。其他中卫节点（170.19、170.32）是 TOS 3.1 (kernel 5.4)，装的 bnxt_re 是 233.0.152.14，工作正常。

按照"与其他节点保持版本一致"的运维原则，最初尝试在 TOS 4.4 上装 233.0.152.14。

## 决策

使用 bnxt_re/bnxt_en **238.1.138.5**（从 Broadcom 官网下载）而非 233.0.152.14。

## 原因

1. **233 源码不兼容 kernel 6.6 API**
   - kernel 6.6 相比 5.4 修改了多个 net_device_ops、ethtool_ops 接口
   - 233 的源码中 `ndo_*` 函数签名与 6.6 不匹配，DKMS 编译直接失败
   - 233 的 `bnxt_en_main.c` 用了 `pci_enable_msix_range()`，6.6 已废弃

2. **238 是 Broadcom 官方支持的版本**
   - 238.1.138.5 源码支持 kernel 6.6（甚至 6.12+）
   - 通过 DKMS 自动适配当前内核版本编译

3. **RoCE 协议是标准协议，版本差异不影响互操作**
   - bnxt_re 238 和 233 都实现 RoCE v2 (RFC 7306)
   - 跨节点 RDMA 通信只依赖 RoCE 协议层，不依赖驱动版本
   - 实测 159 (238) ↔ 87 (238) 的 `ib_send_bw` 达到 183 Gb/sec

4. **TOS 3.1 节点不需要升级**
   - 170.19/170.32 的 233 在 kernel 5.4 上工作良好
   - 没有必要为了版本对齐而升级正在运行的节点
   - 升级需要 reboot，影响业务

## 备选方案

- **方案 A**: 强行装 233 — 编译失败，不可行
- **方案 B**: 等腾讯云提供 TOS 4.4 官方 bnxt_re 包 — 不可控时间，阻塞节点交付
- **方案 C**: 装 238 — ✅ 选中

## 影响

### 正面

- 170.159 和 171.87 的 RDMA 正常工作，183 Gb/sec
- 节点可投入业务使用
- 为后续 TOS 4.4 节点的同款修复提供参考流程

### 中性

- 驱动版本与 TOS 3.1 节点不一致（238 vs 233），但 RoCE 协议层互操作无问题
- NIC 固件也升级到 238.1.138.6（in-tree 233 配套固件是 233.0.152.11）

### 需关注

- 后续腾讯云若发布 TOS 4.4 官方驱动包，需评估是否迁移
- `libbnxt_re` 必须从源码编译（预编译 RPM 依赖 IBVERBS_PRIVATE_57，节点只有 v34）
- DKMS 升级内核时需要重新编译 bnxt_en/bnxt_re，且 MOK 密钥必须保留

## 相关记忆

- `project_tos44_bnxt_re_rdma_install.md` — 完整安装流程记忆

## Related

- [[install-bnxt-re-rdma-driver-tos44]] — 执行的安装流程
- [[bnxt-re-driver]] — bnxt_re 驱动实体
- [[bcm57608-nic]] — NIC 硬件
- [[tos-44]] — 目标操作系统

## Backlinks
