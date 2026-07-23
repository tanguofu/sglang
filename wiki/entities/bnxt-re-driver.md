---
title: bnxt_re Driver
category: entity
tags: [driver, broadcom, roce, rdma, kernel-module]
sources: []
created: 2026-07-19
updated: 2026-07-19
---

# bnxt_re Driver

`bnxt_re`（Broadcom NetXtreme-E RoCE Driver）是 Broadcom BCM57608 系列 NIC 的 RoCE v2 RDMA 内核驱动，提供 InfiniBand verbs verbs 接口，使用户态应用可以通过 RDMA over Converged Ethernet 进行远程直接内存访问。

## 模块依赖

```
bnxt_re  →  bnxt_en  (L2 Ethernet driver)
        →  ib_core   (InfiniBand core)
        →  rdma_ucm  (RDMA user-space Connection Manager)
```

`bnxt_re` 通过 `bnxt_register_dev` 等 exported symbol 与 `bnxt_en` 通信。**两者的版本必须匹配**，否则会出现 `disagrees about version of symbol` 错误。

## 关键符号

- `bnxt_register_dev` — 注册 RDMA 设备
- `bnxt_set_soft_roce_icrc` — 软件 RoCE ICRC 计算
- `bnxt_re_bondX` — 创建的 RDMA 设备名（X 对应 bond 索引）

## 版本

| 版本 | 来源 | 适用内核 |
|---|---|---|
| 233.0.152.14 | out-of-tree (TOS 3.1) | 5.4.x ✅, 6.6.x ❌（API 不兼容） |
| 238.1.138.5 | DKMS (Broadcom 官网) | 6.6.x ✅ |

## 用户态库

`bnxt_re` 内核模块需要配套 `libbnxt_re` 用户态 verbs 库：

- 预编译 RPM (`libbnxt_re-238.1.138.5-rhel9u7.x86_64.rpm`) 依赖 `IBVERBS_PRIVATE_57`
- 节点装的 mlnx-ofed 24.10 的 rdma-core 只提供 `IBVERBS_PRIVATE_34`
- **必须从源码编译**，编译时自动检测 libibverbs 版本，生成 `libbnxt_re-rdmav34.so`

## 加载验证

```bash
lsmod | grep bnxt_re
cat /sys/module/bnxt_re/version  # 238.1.138.5
ls /sys/class/infiniband/  # bnxt_re_bond0 ~ bnxt_re_bond7
ibstat  # 8 CA, Active, Rate 400
```

## 常见加载失败原因

1. **`Unknown symbol bnxt_set_soft_roce_icrc (err -2)`** — `bnxt_en` 版本与 `bnxt_re` 不匹配，可能 initramfs 还在加载旧版 in-tree bnxt_en 233。修复：`dracut -f` 重建 initramfs。
2. **`disagrees about version of symbol bnxt_register_dev`** — 同上，符号表 version mismatch。
3. **`No IB devices found`** — `bnxt_re` 模块没装，需要装 DKMS 238。

## Related

- [[bcm57608-nic]] — 配套 NIC 硬件
- [[install-bnxt-re-rdma-driver-tos44]] — 完整安装流程
- [[bnxt-en-driver]] — 配套 L2 驱动（实际上由 bnxt_en 实体页承载，本 wiki 暂合并于此）
- [[dkms-module-signing]] — DKMS 签名概念
- [[initramfs-rebuild]] — initramfs 重建概念
- [[tos-44]] — 目标操作系统

## Backlinks
