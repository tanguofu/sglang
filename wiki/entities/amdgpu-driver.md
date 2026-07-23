---
title: amdgpu Driver
category: entity
tags: [driver, amd, gpu, kernel-module, mi308x, rocm]
sources: []
created: 2026-07-19
updated: 2026-07-19
---

# amdgpu Driver

`amdgpu` 是 Linux 内核中 AMD GPU 的统一驱动，管理 Radeon 系列 GPU 和 Instinct 系列 compute GPU（如 MI308X）。它提供显存管理、GPU 调度、KFD (Kernel Fusion Driver) 接口，是 ROCm 运行时栈的内核态基础。

## 支持的硬件

- AMD Instinct MI308X (gfx942, device ID `1002:74a2`) — 8 GPU/节点，每卡 256G BAR 内存
- AMD Instinct MI355X (gfx942)
- AMD Radeon RX 7000 系列
- APU 集成显卡

## 版本

| 版本 | 来源 | 说明 |
|---|---|---|
| in-tree（TOS 4.4） | kernel 6.6 自带 | **不支持 MI308X**，缺 ip discovery binary |
| 6.16.13 DKMS | AMD 官方 | 支持 MI308X，配套 ROCm 7.2.4 |

## 关键子系统

- **KFD (Kernel Fusion Driver)** — 用户态提交 compute queue 的接口
- **TTM (Translation Table Maps)** — GPU 显存管理
- **DCN (Display Core Next)** — 显示核心（compute GPU 不用）
- **SDMA** — System DMA engine
- **PSP (Platform Security Processor)** — GPU 固件加载和安全启动

## 加载流程

```
amdgpu.ko  →  drm_kms_helper
          →  drm
          →  ttm
          →  scheduler
          →  amdkfd (KFD)
```

启动时 amdgpu 从 `/lib/firmware/amdgpu/` 加载 GPU 固件（MC、ME、MEC、PFP、RLC、SDMA 等），然后初始化 KFD。如果 ip discovery binary 缺失或不匹配，会报 `invalid ip discovery binary checksum` 并放弃初始化。

## Secure Boot 签名

TOS 4.4 默认开启 Secure Boot，out-of-tree amdgpu 必须用 MOK 签名：

```bash
/usr/src/linux-headers-$(uname -r)/scripts/sign-file \
  sha256 /etc/mok/amdgpu-private.key /etc/mok/amdgpu-public.crt \
  /lib/modules/$(uname -r)/extra/amdgpu/amdgpu.ko
```

并且必须从 GRUB 启动参数中移除 `module.sig_enforce=1`，否则即使签名也会被拒绝。

## 验证

```bash
lsmod | grep amdgpu
cat /sys/module/amdgpu/version  # 6.16.13
rocm-smi  # 8 GPU, Healthy
rocminfo | grep "Name:" | grep -c gfx942  # 8
```

## 相关 ROCm 组件

- **ROCm 7.2.4** — HIP runtime, MIOpen, RCCL, rocFFT 等
- **HSA_OVERRIDE=gfx942** — 必须设置，否则 ROCm 可能错误识别架构
- **HSA_ENABLE_SDMA=0** — MI308X XGMI 单 hive 场景下避免 SDMA 路径

## 已知问题

- **`invalid ip discovery binary checksum`** — in-tree amdgpu 不含 MI308X ip discovery binary，需装 DKMS 6.16.13
- **`Key was rejected`** — out-of-tree 模块未签名，需 MOK 签名（参见 [[dkms-module-signing]]）
- **EAGLE speculative decode 触发 GPU coredump** — 见 project memory `project_mi308x_gpu_coredump_firmware.md`

## Related

- [[install-amdgpu-driver-tos44]] — 完整安装流程
- [[tos-44]] — 目标操作系统
- [[dkms-module-signing]] — DKMS 签名概念
- [[initramfs-rebuild]] — initramfs 重建概念
- [[bnxt-re-driver]] — 同节点配套 NIC 驱动（独立但协同工作）

## Backlinks
