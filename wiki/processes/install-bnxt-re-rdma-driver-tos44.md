---
title: Install bnxt_re RDMA Driver on TOS 4.4
category: process
tags: [tos-44, bnxt-re, bnxt-en, rdma, roce, bcm57608, dkms, firmware-upgrade, niccli, libbnxt-re]
sources: []
created: 2026-07-19
updated: 2026-07-19
---

# Install bnxt_re RDMA Driver on TOS 4.4

在 TencentOS Server 4.4 (kernel 6.6.110-42.4.tl4.x86_64) 节点上为 Broadcom BCM57608 (BCM957608-P2200GQF00, 2×200G PCIe RoCE NIC) 安装 bnxt_en/bnxt_re DKMS 驱动、libbnxt_re 用户态库、niccli 工具，并升级 NIC 固件。本流程于 2026-07-19 在 170.159 和 171.87 验证通过，`ib_send_bw` 实测 183 Gb/sec。

## 适用场景

- TOS 4.4 节点配备 Broadcom BCM57608 NIC，但 `ibstat` / `ibv_devinfo` 报 "No IB devices found"
- `/sys/class/infiniband/` 目录为空
- `lsmod | grep bnxt_re` 无输出（TOS 4.4 内核只有 in-tree bnxt_en 233，完全没有 bnxt_re）
- 腾讯 RDMA 脚本 (`/usr/local/qcloud/rdma/`) 已就绪但缺少 `bnxt_re.ko` 模块

## 前置条件

- [ ] 节点运行 TOS 4.4，内核 `6.6.110-42.4.tl4.x86_64`
- [ ] 已安装 `dkms`、`kernel-devel`、`gcc`、`make`、`autoconf`、`automake`、`libtool`
- [ ] 已从 Broadcom 官网下载 `bcm_238.1.138.6a.tar.gz` (337MB)
- [ ] 已从 Broadcom 官网下载 `niccli-238.1.138.6_linux.zip` (31MB)
- [ ] 已通过 `kubectl cp` 将包传到节点 `/host/tmp/`
- [ ] 通过 debug pod 进入节点：`chroot /host nsenter -t 1 -m -u -i -n --`

## 步骤

### 1. 传输驱动包到节点

本机不能直连节点 IP，需通过 debug pod 中转：

```bash
# 本机 -> debug pod
kubectl cp /Users/guofutan/Downloads/bcm_238.1.138.6a.tar.gz \
  kube-system/debug-ds-<pod-id>:/tmp/

kubectl cp /Users/guofutan/Downloads/niccli-238.1.138.6_linux.zip \
  kube-system/debug-ds-<pod-id>:/tmp/

# 在 debug pod 内 -> 宿主机 /tmp/
cp /tmp/bcm_238.1.138.6a.tar.gz /host/tmp/
cp /tmp/niccli-238.1.138.6_linux.zip /host/tmp/

# 切到宿主机命名空间
chroot /host /usr/bin/nsenter -t 1 -m -u -i -n --
cd /tmp
tar xzf bcm_238.1.138.6a.tar.gz
unzip niccli-238.1.138.6_linux.zip
```

### 2. 安装 bnxt_en DKMS 238

```bash
cd /tmp/bcm_238.1.138.6a/drivers_linux/bnxt_en/dkms
rpm -ivh bnxt_en-1.10.3.238.1.138.5-1dkms.noarch.rpm
# DKMS 自动 add/build/install，并自动用 MOK 签名（若已 enroll）
```

### 3. 安装 bnxt_re DKMS 238

```bash
cd /tmp/bcm_238.1.138.6a/drivers_linux/bnxt_re/dkms
rpm -ivh bnxt_re-238.1.138.5-1dkms.noarch.rpm

cd ../bnxt_re_conf
rpm -ivh bnxt_re_conf-238.1.138.5-1.noarch.rpm
```

### 4. 编译 libbnxt_re 用户态库（关键！）

**不能直接装 rhel9.7 预编译 RPM**，否则会因 libibverbs 版本不匹配而失败。

```bash
# 预编译 RPM 的依赖问题：
# libbnxt_re-238.1.138.5-rhel9u7.x86_64.rpm 需要 libibverbs.so.1(IBVERBS_PRIVATE_57)
# 但节点装的 mlnx-ofed 24.10 的 rdma-core 只提供 IBVERBS_PRIVATE_34
# 报错: version not found

# 必须从源码编译：
cd /tmp/bcm_238.1.138.6a/drivers_linux/bnxt_rocelib
tar xzf libbnxt_re-238.1.138.5.tar.gz
cd libbnxt_re-238.1.138.5
./configure --prefix=/usr --libdir=/usr/lib64
make -j8
make install

# 编译时自动检测 libibverbs 版本，生成 libbnxt_re-rdmav34.so (不是 rdmav57)

# 如果之前装过预编译 RPM，必须删除 v57 版本
rm -f /usr/lib64/libbnxt_re-rdmav57.so
ldconfig
```

### 5. 安装 niccli

```bash
cd /tmp/niccli-238.1.138.6_linux/linux_x86_64
rpm -ivh niccli-238.1.138.6-1.x86_64.rpm
# 注意: 如果 rpm -ivh 报 "No such file" 但文件确实存在，
# 是 nsenter 路径解析问题，先 cd 进目录再执行
```

### 6. 升级 NIC 固件

每张物理卡有 2 个 PF (`.0` 和 `.1`) 共享一个 NVM，**只需升级 `.0`**。

```bash
# 列出所有 Broadcom NIC 的 BDF
lspci -d14e4: | awk '{print $1}' | grep '\.0$'
# 输出: 0000:06:00.0 0000:15:00.0 0000:1e:00.0 0000:31:00.0
#       0000:83:00.0 0000:9f:00.0 0000:b5:00.0 0000:c1:00.0

# 升级每张卡（固件包路径在解压目录的 board_sku_files/ 下）
for bdf in 0000:06:00.0 0000:15:00.0 0000:1e:00.0 0000:31:00.0 \
           0000:83:00.0 0000:9f:00.0 0000:b5:00.0 0000:c1:00.0; do
  echo "=== Upgrading $bdf ==="
  niccli --pci $bdf fw -u -f \
    /tmp/bcm_238.1.138.6a/board_sku_files/BCM957608-P2200GQF00.pkg -y
done

# 固件版本: 233.0.152.11 → 238.1.138.6
```

### 7. 重建 initramfs（关键！容易遗漏）

DKMS 装 bnxt_en 238 后，旧 in-tree 233 仍嵌入在 initramfs 里，启动时优先加载旧版会导致 bnxt_re 符号表不匹配。

```bash
dracut -f /boot/initramfs-$(uname -r).img $(uname -r)

# 验证 initramfs 内嵌的是新版 bnxt_en
lsinitrd /boot/initramfs-$(uname -r).img | grep bnxt
# 期望: extra/bnxt_en.ko.xz  (DKMS 路径)
# 不要: kernel/drivers/net/ethernet/broadcom/bnxt/bnxt_en.ko  (in-tree 旧版)
```

**不重建的后果**：启动后 `dmesg` 会出现：
```
bnxt_re: Unknown symbol bnxt_set_soft_roce_icrc (err -2)
bnxt_re: disagrees about version of symbol bnxt_register_dev
```

### 8. 重启节点

```bash
reboot
```

启动后 `rdma-agent.service` 和 `RDMA_FastStart.service` 自动配置 bond0~7（802.3ad LACP, MTU 9100, DHCP）。等待约 90 秒让 8 个 bond 全部起来。

### 9. 验证

```bash
# 模块版本
cat /sys/module/bnxt_en/version  # 1.10.3-238.1.138.5
cat /sys/module/bnxt_re/version  # 238.1.138.5

# RDMA 设备
ls /sys/class/infiniband/  # bnxt_re_bond0 ~ bnxt_re_bond7

ibstat  # 8 个 CA, State: Active, Rate: 400, Firmware: 238.1.138.0
rdma link  # 8 条链路全部 ACTIVE

# 用户态
ibv_devinfo | grep -c "hca_id"  # 8
```

### 10. 带宽测试

腾讯 RDMA 网络拓扑：每个 bond 是独立 /30 子网，**跨节点同 bond 通信走 bond7 网关三层路由**。

```bash
# 在 171.87 上启动 server
ib_send_bw -d bnxt_re_bond7 --report_gbits

# 在 170.159 上启动 client（用 87 的 bond7 IP）
ib_send_bw -d bnxt_re_bond7 --report_gbits -F 29.199.73.46

# 实测结果: 183.11 Gb/sec
```

**路由注意**：`ping -I bond0 <对端 bond0 IP>` 可能失败，因为两个节点的 bond0 不在同一 /30 子网。测试前先用 `ping -I bond7 <对端 bond7 IP>` 确认路由可达。

## 驱动版本对照

| 组件 | 修复前 | 修复后 |
|---|---|---|
| bnxt_en | 1.10.3-233.0.152.14 (in-tree) | 1.10.3-238.1.138.5 (DKMS) |
| bnxt_re | **未安装** | 238.1.138.5 (DKMS) |
| NIC 固件 | 233.0.152.11 / pkg 233.1.135.14 | 238.1.138.6 |
| /sys/class/infiniband/ | 空 | bnxt_re_bond0~7 |
| ibv_devinfo | No IB devices found | 8 HCA, Active |
| ib_send_bw | N/A | 183 Gb/sec |

## 回滚方案

```bash
# 卸载 DKMS
dkms remove bnxt_re/238.1.138.5 -k $(uname -r)
dkms remove bnxt_en/1.10.3.238.1.138.5 -k $(uname -r)

# 移除 libbnxt_re
rm -f /usr/lib64/libbnxt_re*.so*
ldconfig

# 卸载 niccli
rpm -e niccli-238.1.138.6-1.x86_64

# 重建 initramfs（恢复 in-tree bnxt_en 233）
dracut -f /boot/initramfs-$(uname -r).img $(uname -r)

reboot
```

## 常见坑

1. **预编译 RPM 的 IBVERBS_PRIVATE_57 依赖** — 必须从源码编译 libbnxt_re，否则用户态 verbs 调用失败。
2. **initramfs 旧 bnxt_en 233 优先加载** — 必须 `dracut -f` 重建，否则 bnxt_re 因符号表不匹配无法加载。
3. **固件升级只升 .0 不升 .1** — 同一物理卡的两个 PF 共享 NVM，升级 `.0` 即可，重复升 `.1` 会报错或无效。
4. **niccli 在 nsenter 下路径解析异常** — `rpm -ivh /abs/path/foo.rpm` 可能报 "No such file"，先 `cd` 进目录再 `rpm -ivh foo.rpm`。
5. **bond0 跨节点不通** — 腾讯 RDMA 每个 bond 是 /30 P2P 子网，跨节点同 bond 走 bond7 网关三层路由，测试用 bond7。
6. **rdma-agent.service 未就绪** — 重启后立即测试可能只有 3~4 个 bond，等 90 秒让 `rdma-agent.service` 完成 `activating` 状态。

## Related

- [[install-amdgpu-driver-tos44]] — 同一批 TOS 4.4 节点的 GPU 驱动安装流程
- [[tos-44]] — TencentOS Server 4.4 操作系统概述
- [[bcm57608-nic]] — Broadcom BCM57608 NIC 硬件实体
- [[bnxt-re-driver]] — bnxt_re RoCE 驱动实体
- [[dkms-module-signing]] — DKMS 模块签名概念
- [[initramfs-rebuild]] — initramfs 重建概念
- [[use-bnxt-re-238-on-tos44]] — 为什么选 238 而不是与正常节点一致的 233

## Backlinks
