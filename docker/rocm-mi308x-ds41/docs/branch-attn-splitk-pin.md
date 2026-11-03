# 分支说明：decode attention split_k 钉常量

- **分支**：`fix/ds41-attn-splitk-pin`
- **提交**：`ada08d2653`
- **基线**：`ds41-flash` @ `49aef68124`（分支只有 1 个提交，merge-base 即基线 HEAD）
- **改动文件**：`python/sglang/kernels/ops/attention/nsa_triton_decode/triton_mla_kernels_decode_fused.py`（+42 / −2）
- **当前状态**：**未合入 `ds41-flash`**。已决定不上线，保留分支待评估。

## 解决什么

双域（dual-scope）decode attention 的 few-heads 分支在 `total_tokens <= 8` 时返回
`split_k=8`、在其上返回 `split_k=4`（见 `_decide_splitk_dual_scope_ungated`）。
`split_k` 会重排 fp32 归约顺序，所以**同一批 token 的 bit 取决于这个 batch 恰好有多少行**。

投机解码开着时（DSPARK，gamma=3）`T = 4 * bs`，切换点正好落在
`conc=2`（T ∈ {4, 8}）与 `conc=4`（T 到 16）之间 —— 也正是实测分歧从
2.4% 跳到 15.2%（Fisher p=5.1e-5）、HumanEval+ 从 150 掉到 146 的位置。

## 开关与默认值

| 项 | 值 |
|---|---|
| 环境变量 | `SGLANG_DS41_ATTN_SPLITK_PIN` |
| 默认 | `"0"` = **今天的行为**（保留 T 相关的切换） |
| 接受值 | `2` / `4` / `8`；其他值（含非法字符串）静默回落 `0`，不会发出没人处理的 `split_k` |

两个约束：

1. pin 只作用于 **few-heads decode 这一个分支内部**，所以它不会扩大或缩小哪些
   `(T, h_q, topk)` 能走到 split-K，只是把该分支里那个随 T 变的值换成常量。
   `SMALL_BATCH_TOKEN_THRESHOLD`(8) 与 `SMALL_HQ_DECODE_TOKEN_THRESHOLD`(128) 未动。
2. pin 生效的前提是 `SGLANG_DS41_ATTN_SPLITK=1`（现网值）且
   `SGLANG_DS41_ATTN_P_HILO=1`（现网值）。`SGLANG_DS41_ATTN_SPLITK=0` 时
   few-heads 分支根本不进入，pin 无作用。

## 已自证什么

在 m17（`df8cd6cf`）、gfx942、`h_q=8`、`topk=640`、真实 ue8m0 KV 布局下：

1. **pin=8 抹平 T 的影响**：`T = 8, 9, 12, 16, 32, 64, 128` 的 output 与 LSE 与
   `T <= 8`（split_k=8）基准 `torch.equal`，maxdiff `0.000e+00`，**7/7**。
   作为对照，今天的 `T >= 9` 行为：output maxdiff `3.906e-03`（0.07% 元素）、
   LSE maxdiff `9.537e-07`（25% 元素）。
2. **pin=8 对 `T <= 8` 是恒等变换**：`T=4` 与 `T=8` 在开关开/关下逐字节相同。
   这是构造性的保证 —— `conc <= 2` 完全不动，因此可以在**已经在跑的配置上**
   做零风险的非回归门禁。

## 代价

仅微基准，**不是端到端结论**（空闲卡，两轮给出波动）。pin=8 相对 pin=4，ms/call 比值：

| T | 第 1 轮 | 第 2 轮 |
|---|---|---|
| 8 | 1.00 | 1.10 |
| 16 | 1.07 | 1.16 |
| 32 | 1.08 | 1.08 |
| 64 | 1.03 | 1.11 |
| 128 | 1.16 | 1.16 |

最坏是 `T=128` 时 attention kernel +16%（+38 µs/call）。端到端代价必须由
质量/吞吐门禁给出，不能由这个数推断。

## 上线前还需要什么

- 端到端门禁（质量 + 吞吐），在**同一窗口**内做 A/B，不能只信微基准。
- 确认目标环境的 `SGLANG_DS41_ATTN_SPLITK=1` 与 `SGLANG_DS41_ATTN_P_HILO=1` 都在。
- 镜像需包含本提交（m17 之后构建的镜像）。
- 回滚方式：把 `SGLANG_DS41_ATTN_SPLITK_PIN` 去掉/置 0 并重启，回到今天的行为。
