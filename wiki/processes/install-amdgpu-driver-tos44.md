---
title: Install amdgpu Driver on TOS 4.4
category: process
tags: [tos-44, amdgpu, mi308x, dkms, mok-signing, rocm, gpu-driver]
sources: []
created: 2026-07-19
updated: 2026-07-19
---

# Install amdgpu Driver on TOS 4.4

在 TencentOS Server 4.4 (kernel 6.6.110-42.4.tl4.x86_64) 节点上为 AMD MI308X (gfx942, device ID `1002:74a2`) 安装 amdgpu 内核驱动并配置 ROCm 运行时。本流程于 2026-07-19 在 170.159 和 171.87 两台节点上验证通过。

## 适用场景

- TOS 4.4 节点配备 MI308X GPU，但 `rocm-smi` 报错或找不到 GPU
- dmesg 出现 `amdgpu: invalid ip discovery binary checksum`（in-tree 驱动问题）
- dmesg 出现 `Key was rejected` / `required key not available`（out-of-tree 驱动未签名）
- 需要从头重建 amdgpu 驱动栈

## 前置条件

- [ ] 节点运行 TencentOS Server 4.4，内核 `6.6.110-42.4.tl4.x86_64`
- [ ] 已安装 `dkms`、`kernel-devel`、`gcc`、`make`
- [ ] 已下载 amdgpu 6.16.13 DKMS 源码包
- [ ] 已下载 ROCm 7.2.4 安装包（或配置好 amdgpu repo）
- [ ] 通过 debug pod 进入节点主机命名空间（`chroot /host nsenter -t 1 -m -u -i -n --`）

## 步骤

### 1. 检查现状

```bash
# 确认内核版本
uname -r  # 应为 6.6.110-42.4.tl4.x86_64

# 检查 amdgpu 模块状态
lsmod | grep amdgpu
modinfo amdgpu | head -5

# 检查 GPU 是否被识别
lspci -d 1002:74a2 | wc -l  # 应为 8

# 检查 dmesg 报错
dmesg | grep -i amdgpu | tail -20
```

### 2. 卸载冲突的旧驱动

```bash
# 如果有旧的 out-of-tree amdgpu（未经 DKMS 管理），手动卸载
rmmod amdgpu 2>/dev/null

# 清理旧残留
find /lib/modules/$(uname -r) -name "amdgpu*" -not -path "*/extra/*" -delete
```

### 3. 安装 amdgpu 6.16.13 DKMS

```bash
# 解压源码包
tar xzf amdgpu-6.16.13.tar.gz
cd amdgpu-6.16.13

# 通过 DKMS 安装（自动 add/build/install）
dkms add .
dkms build amdgpu/6.16.13 -k $(uname -r)
dkms install amdgpu/6.16.13 -k $(uname -r)
```

### 4. 生成 MOK 密钥并签名模块

TOS 4.4 默认开启 Secure Boot，未签名的 out-of-tree 模块无法加载。

```bash
# 生成 MOK 密钥对
mokutil --export-key  # 如果已有密钥跳过
openssl req -new -x509 -newkey rsa:2048 \
  -keyout /etc/mok/amdgpu-private.key \
  -out /etc/mok/amdgpu-public.crt \
  -subj "/CN=amdgpu module signing/" -days 3650

# 签名 amdgpu 模块
/usr/src/linux-headers-$(uname -r)/scripts/sign-file \
  sha256 /etc/mok/amdgpu-private.key /etc/mok/amdgpu-public.crt \
  /lib/modules/$(uname -r)/extra/amdgpu/amdgpu.ko

# 将公钥注册到 MOK 队列（下次启动时需要输入密码确认）
mokutil --import /etc/mok/amdgpu-public.crt
# 提示输入密码，记住此密码，重启时要用
```

### 5. 移除 `module.sig_enforce=1`

如果内核启动参数包含 `module.sig_enforce=1`，即使签名也会拒绝加载未内置到内核的模块。

```bash
# 检查当前启动参数
grep -E "sig_enforce|module.sig" /proc/cmdline

# 编辑 GRUB 配置
vi /etc/default/grub
# 从 GRUB_CMDLINE_LINUX 中移除 module.sig_enforce=1

# 重新生成 grub.cfg
grub2-mkconfig -o /boot/grub2/grub.cfg
# 或 grub2-mkconfig -o /boot/efi/EFI/centos/grub.cfg (UEFI)
```

### 6. 安装 ROCm 7.2.4

```bash
# 配置 amdgpu repo (或使用本地包)
rpm --import https://repo.radeon.com/RPM-GPG-KEY-amdgpu
yum install -y amdgpu-dkms rocm-dev rocm-libs rocm-utils \
  hip-runtime-amd miopen-hip rccl rocfft rocsparse rocrand

# 或从本地 tar 包安装
tar xzf rocm-7.2.4.tar.gz -C /opt
ln -s /opt/rocm-7.2.4 /opt/rocm
```

### 7. 重建 initramfs

```bash
dracut -f /boot/initramfs-$(uname -r).img $(uname -r)
```

### 8. 重启节点

```bash
reboot
```

重启过程中会进入 MOK Manager 蓝色界面：
- 选择 "Enroll MOK"
- 选择之前导入的公钥
- 输入步骤 4 设置的密码
- 选择 "Reboot"

### 9. 验证

```bash
# 加载状态
lsmod | grep amdgpu
cat /sys/module/amdgpu/version  # 应为 6.16.13

# GPU 可见性
rocm-smi  # 应显示 8 个 GPU，状态 Healthy
rocminfo | grep "Name:" | grep -c gfx942  # 应为 8

# BAR 内存
dmesg | grep -i "bar memory" | head -8  # 每个 256G

# HSA 状态
echo $HSA_OVERRIDE  # 应为 gfx942
```

## 两种典型故障场景对照

| 场景 | 症状 | 根因 | 修复 |
|---|---|---|---|
| 170.159 (in-tree) | `invalid ip discovery binary checksum` | TOS 4.4 内置 in-tree amdgpu 不含 MI308X ip discovery binary | 装 DKMS 6.16.13 out-of-tree |
| 171.87 (out-of-tree 未签名) | `Key was rejected` | 已装 out-of-tree 但未用 MOK 签名，Secure Boot 阻止加载 | DKMS 重新编译 + MOK 签名 |

## 回滚方案

```bash
# 卸载 DKMS 模块
dkms remove amdgpu/6.16.13 -k $(uname -r)

# 恢复 GRUB 启动参数（重新加回 module.sig_enforce=1）
vi /etc/default/grub
grub2-mkconfig -o /boot/grub2/grub.cfg

# 移除 MOK 公钥
mokutil --delete /etc/mok/amdgpu-public.crt

# 重建 initramfs
dracut -f /boot/initramfs-$(uname -r).img $(uname -r)

# 卸载 ROCm
yum remove -y amdgpu-dkms rocm-*

reboot
```

## 常见坑

1. **MOK 密钥丢失** — `/etc/mok/` 下的 `.key` 和 `.crt` 文件必须持久保存，下次升级驱动还要用同一个密钥签名，否则需要重新 enroll。
2. **sig_enforce 漏改** — 即使签名正确，`module.sig_enforce=1` 仍会阻止加载，必须从 GRUB 启动参数中移除。
3. **initramfs 未重建** — DKMS 安装的模块在 `/lib/modules/.../extra/` 下，但启动时可能从 initramfs 加载旧版本，必须 `dracut -f`。
4. **ROCm 版本不匹配** — ROCm 7.2.4 配套 amdgpu 6.16.13，混搭版本可能造成 `hipRuntimeVersion` 不一致。

## Related

- [[install-bnxt-re-rdma-driver-tos44]] — 同一批 TOS 4.4 节点的网卡/RDMA 驱动安装流程
- [[tos-44]] — TencentOS Server 4.4 操作系统概述
- [[amdgpu-driver]] — AMD GPU 内核驱动实体
- [[dkms-module-signing]] — DKMS 模块签名概念
- [[initramfs-rebuild]] — initramfs 重建概念

## Backlinks
