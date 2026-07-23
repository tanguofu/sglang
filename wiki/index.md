# Wiki Index

> Auto-maintained by /teamai-wiki. Last updated: 2026-07-19
> Pages: 10 | Links: 28 | Sources: 0

## Entities

- [[tos-44]] — TencentOS Server 4.4 (kernel 6.6) GPU 节点操作系统
- [[bcm57608-nic]] — Broadcom BCM57608 双口 200G RoCE NIC (P2200GQF00)
- [[bnxt-re-driver]] — Broadcom RoCE v2 RDMA 内核驱动
- [[amdgpu-driver]] — AMD GPU 统一内核驱动（MI308X 需要 DKMS 6.16.13）

## Concepts

- [[dkms-module-signing]] — DKMS + MOK 签名机制（Secure Boot 环境下必备）
- [[initramfs-rebuild]] — DKMS 升级内核模块后必须 `dracut -f` 重建 initramfs

## Comparisons

_No pages yet._

## People

_No pages yet._

## Decisions

- [[use-bnxt-re-238-on-tos44]] — TOS 4.4 用 bnxt_re 238 而非与 TOS 3.1 一致的 233（2026-07-19）

## Processes

- [[install-amdgpu-driver-tos44]] — TOS 4.4 上安装 MI308X amdgpu 驱动 + ROCm 完整流程
- [[install-bnxt-re-rdma-driver-tos44]] — TOS 4.4 上安装 BCM57608 bnxt_re RDMA 驱动 + 固件升级完整流程

## Sources

_No pages yet._

## Queries

_No pages yet._

## 遗留文档（未纳入分类）

- `glm52-0702-amd-optimization.md` — GLM-5.2 on AMD MI355X 优化记录
- `glm52-0702-amd-optimization-next-steps.md` — 后续优化计划
