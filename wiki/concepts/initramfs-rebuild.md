---
title: initramfs Rebuild
category: concept
tags: [initramfs, dracut, boot, kernel-module]
sources: []
created: 2026-07-19
updated: 2026-07-19
---

# initramfs Rebuild

`initramfs`（Initial RAM Filesystem）是启动时加载到内存的临时根文件系统，包含内核启动早期需要的模块和工具。当通过 DKMS 安装新版内核模块时，旧版模块可能仍嵌入在 initramfs 里，启动时会被优先加载，导致新版模块失效。

## 为什么需要重建

Linux 启动模块加载顺序：

```
1. 内核启动 → 加载 initramfs 作为临时 rootfs
2. 从 initramfs 加载早期模块（存储、网络、文件系统驱动）
3. 挂载真实根文件系统
4. 切换到真实 rootfs
5. udev 加载 /lib/modules/$(uname -r)/ 下的其他模块
```

**问题**：DKMS 装的新版模块在 `/lib/modules/$(uname -r)/extra/` 下，但 initramfs 里仍然嵌入着旧版（如 in-tree 233 的 `kernel/drivers/net/.../bnxt_en.ko`）。启动时 step 2 优先加载旧版，step 5 加载的 DKMS 新版因同名模块已存在而被忽略。

## dracut 命令

```bash
# 重建当前内核的 initramfs
dracut -f /boot/initramfs-$(uname -r).img $(uname -r)

# -f  强制覆盖已有 initramfs
# 第一个参数  输出文件路径
# 第二个参数  内核版本

# 验证 initramfs 内嵌的模块
lsinitrd /boot/initramfs-$(uname -r).img | grep <module>
```

## 何时必须重建

| 场景 | 必须重建？ |
|---|---|
| DKMS 安装新版内核模块（bnxt_en、amdgpu 等） | ✅ 必须 |
| 修改 `/etc/dracut.conf.d/` 配置 | ✅ 必须 |
| 升级内核 | ✅ 自动重建（kernel 包的 %post 脚本） |
| 仅修改用户态工具（niccli、libbnxt_re） | ❌ 不需要 |
| 仅修改 `/etc/modprobe.d/` 配置 | ❌ 不需要（modprobe 配置不在 initramfs） |

## bnxt_en 233→238 升级案例

**症状**：DKMS 装 bnxt_en 238 后重启，`dmesg` 报：
```
bnxt_re: Unknown symbol bnxt_set_soft_roce_icrc (err -2)
bnxt_re: disagrees about version of symbol bnxt_register_dev
```

**根因**：
- initramfs 里嵌入的是 in-tree bnxt_en 233 (`srcversion BC0135C10A1B8CD2E045F50`)
- bnxt_re 238 期望的符号表 srcversion 是 `24436B2B5C386D69ABF66C6`
- 启动时先加载 initramfs 里的 233，导致 bnxt_re 238 符号表不匹配

**修复**：
```bash
dracut -f /boot/initramfs-$(uname -r).img $(uname -r)
lsinitrd /boot/initramfs-$(uname -r).img | grep bnxt
# 期望: extra/bnxt_en.ko.xz  (DKMS 路径)
# 不要: kernel/drivers/net/ethernet/broadcom/bnxt/bnxt_en.ko  (in-tree 旧版)
reboot
```

## 验证方法

```bash
# 重启后检查运行中的模块版本
cat /sys/module/bnxt_en/version  # 应为 1.10.3-238.1.138.5
cat /sys/module/bnxt_re/version  # 应为 238.1.138.5

# 检查 srcversion 是否一致
cat /sys/module/bnxt_en/srcversion
cat /sys/module/bnxt_re/srcversion
```

## 何时容易被遗漏

DKMS 安装时通常会提示 `dracut: rebuild initramfs`，但：
1. 如果是通过 `rpm -ivh` 装 DKMS 包，可能跳过 dracut 钩子
2. 在 debug pod 内通过 `nsenter` 执行命令时，可能不在 chroot 环境完整执行
3. 第一次重启后才出现符号表错误，需要第二次重启才能验证修复

**最佳实践**：DKMS 安装新版内核模块后，**总是手动执行 `dracut -f`**，不要依赖包管理器自动触发。

## Related

- [[install-bnxt-re-rdma-driver-tos44]] — 必须重建 initramfs 的典型场景
- [[install-amdgpu-driver-tos44]] — 同上
- [[dkms-module-signing]] — 配套的签名概念
- [[bnxt-re-driver]] — 符号表不匹配的受害者
- [[amdgpu-driver]] — 同样需要重建

## Backlinks
