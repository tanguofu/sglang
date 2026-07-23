---
title: DKMS Module Signing
category: concept
tags: [dkms, mok, secure-boot, kernel-module, signing]
sources: []
created: 2026-07-19
updated: 2026-07-19
---

# DKMS Module Signing

DKMS (Dynamic Kernel Module Support) 是一种将 out-of-tree 内核模块以源码形式打包、在内核升级时自动重新编译的机制。在开启 Secure Boot 的系统上，DKMS 编译出的 `.ko` 模块必须用 Machine Owner Key (MOK) 签名才能被内核加载。

## 为什么需要

1. **Secure Boot 链** — 内核启动时校验所有加载的模块签名，未签名的 out-of-tree 模块被拒绝
2. **module.sig_enforce=1** — 部分发行版默认开启此启动参数，比 Secure Boot 更严格，连内置密钥都不放过
3. **DKMS 升级路径** — 每次内核升级，DKMS 自动重新编译模块，重新签名才能继续加载

## MOK 密钥生命周期

```
1. 生成密钥对
   openssl req -new -x509 -newkey rsa:2048 \
     -keyout /etc/mok/<module>-private.key \
     -out /etc/mok/<module>-public.crt \
     -subj "/CN=<module> module signing/" -days 3650

2. 签名 .ko 模块
   /usr/src/linux-headers-$(uname -r)/scripts/sign-file \
     sha256 /etc/mok/<module>-private.key /etc/mok/<module>-public.crt \
     /lib/modules/$(uname -r)/extra/<module>/<module>.ko

3. 注册公钥到 MOK 队列
   mokutil --import /etc/mok/<module>-public.crt
   # 提示输入密码（一次性，重启时用）

4. 重启 → 进入 MOK Manager 蓝色界面
   - 选 "Enroll MOK"
   - 选刚才导入的公钥
   - 输入步骤 3 的密码
   - 选 "Reboot"

5. 重启后公钥被写入内核 MOK 数据库
   mokutil --list-enrolled | grep "CN=<module>"
```

## 关键命令

```bash
# 查看已 enroll 的密钥
mokutil --list-enrolled

# 查看待 enroll 队列
mokutil --list-new

# 删除已 enroll 的密钥
mokutil --delete /etc/mok/<module>-public.crt

# 验证模块签名
modinfo <module> | grep "^sig"
# 应输出 signer, key, sig_id, sig_hashalgo
```

## 注意事项

1. **密钥文件持久保存** — `/etc/mok/*.key` 和 `/etc/mok/*.crt` 必须备份，下次升级驱动还要用同一密钥
2. **sig_enforce 必须移除** — 即使签名正确，`module.sig_enforce=1` 仍会拒绝加载，需从 `/etc/default/grub` 的 `GRUB_CMDLINE_LINUX` 移除并 `grub2-mkconfig -o /boot/grub2/grub.cfg`
3. **DKMS 自动签名** — 部分 DKMS 包（如 amdgpu 6.16.13）在 `dkms.conf` 中配置了 `POST_BUILD` 钩子，会自动用 `/etc/mok/` 下的密钥签名，无需手动 sign-file
4. **MOK 密码** — `mokutil --import` 时设置的密码只在 enroll 阶段用一次，enroll 完成后不再需要

## 在 TOS 4.4 上的应用

- **amdgpu 6.16.13** — 需 MOK 签名 + 移除 sig_enforce（参见 [[install-amdgpu-driver-tos44]]）
- **bnxt_en/bnxt_re 238** — DKMS 包自动签名，但首次需 enroll MOK 公钥（参见 [[install-bnxt-re-rdma-driver-tos44]]）

## 排错

| 症状 | 根因 | 修复 |
|---|---|---|
| `Key was rejected` | 模块未签名 | MOK 签名 + enroll |
| `required key not available` | MOK 公钥未 enroll | `mokutil --import` + 重启 |
| 即使签名仍被拒绝 | `module.sig_enforce=1` | 从 GRUB 启动参数移除 |
| 升级内核后模块加载失败 | DKMS 重编译但未重新签名 | 检查 `dkms.conf` 的 `POST_BUILD` 钩子 |

## Related

- [[install-amdgpu-driver-tos44]] — GPU 驱动安装流程
- [[install-bnxt-re-rdma-driver-tos44]] — NIC 驱动安装流程
- [[initramfs-rebuild]] — 配套的 initramfs 重建概念
- [[amdgpu-driver]] — 需要 MOK 签名的内核模块示例
- [[bnxt-re-driver]] — 同上

## Backlinks
