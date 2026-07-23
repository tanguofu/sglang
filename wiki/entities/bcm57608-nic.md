---
title: Broadcom BCM57608 NIC
category: entity
tags: [hardware, nic, broadcom, roce, 200g, pcie, p2200g]
sources: []
created: 2026-07-19
updated: 2026-07-19
---

# Broadcom BCM57608 NIC

Broadcom NetXtreme-E BCM57608 是一款双口 200GbE PCIe 以太网卡，支持 RoCE v2。腾讯 GPU 节点使用的 SKU 为 `BCM957608-P2200GQF00`（P2200G 标准版，**非 PTP 版本**）。

## 硬件标识

- **PCI Device ID**: `14e4:1760` rev 11
- **SKU**: BCM957608-P2200GQF00
- **型号**: P2200G（标准版，2×200G）
- **变体**: P2200G-PTP（带精密时间协议，**不要装错固件**）

## 软件 Stack

| 层 | 组件 | 说明 |
|---|---|---|
| L2 Ethernet | `bnxt_en` | Linux 以太网驱动，TOS 4.4 in-tree 233，需升级到 DKMS 238 |
| RoCE RDMA | `bnxt_re` | RoCE v2 RDMA 驱动，**TOS 4.4 缺失**，必须装 DKMS 238 |
| User-space verbs | `libbnxt_re` | 用户态 RoCE verbs 库，必须从源码编译生成 rdmav34.so |
| 管理 CLI | `niccli` | 固件升级、NIC 配置工具，238.1.138.6 |
| 固件 | `.pkg` 文件 | `BCM957608-P2200GQF00.pkg`（238.1.138.6） |

## 每节点 8 张卡

每个 GPU 节点配 8 张 BCM57608（每张 2 口 200G），PCI BDF 分布：

```
0000:06:00.0  0000:15:00.0  0000:1e:00.0  0000:31:00.0
0000:83:00.0  0000:9f:00.0  0000:b5:00.0  0000:c1:00.0
```

每张物理卡有 2 个 PF (`.0` 和 `.1`) 共享一个 NVM，固件升级只需升 `.0`。

## 腾讯 RDMA 网络拓扑

腾讯自动化脚本 (`/usr/local/qcloud/rdma/bnxt_service.sh`) 配置：

- 8 张 NIC → 8 个 bond（`bond0`~`bond7`），802.3ad LACP
- MTU 9100
- 每个 bond 是独立 /30 P2P 子网，DHCP 自动获取 IP
- 跨节点同 bond 通信走 bond7 网关三层路由（`29.198.0.0/15 via <bond7_gw> dev bond7`）
- QoS: ROCE_DSCP=40, ROCE_PRI=5, CNP_DSCP=48, CNP_PRI=6

## 固件版本

- 出厂: 233.0.152.11 / pkg 233.1.135.14
- 升级后: 238.1.138.6（参见 [[install-bnxt-re-rdma-driver-tos44]]）

## 实测带宽

`ib_send_bw` between 170.159 and 171.87 on `bnxt_re_bond7`: **183.11 Gb/sec**

## Related

- [[bnxt-re-driver]] — RoCE RDMA 驱动
- [[install-bnxt-re-rdma-driver-tos44]] — 完整安装流程
- [[tos-44]] — TOS 4.4 操作系统
- [[use-bnxt-re-238-on-tos44]] — 为什么选 238 版本

## Backlinks
