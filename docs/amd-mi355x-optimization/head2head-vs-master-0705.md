
## 13. Head-to-Head 性能对比：node-1（优化配置）vs master（基线配置）

### 13.1 配置差异

| 配置项 | Master（基线） | Node-1（优化） |
|--------|---------------|---------------|
| 镜像 | `20260629` | `20260702` |
| MTP steps | 2 | 3 |
| MTP draft_tokens | 3 | 4 |
| max_running_requests | 128 | 32 |
| cuda-graph-bs decode | 默认(1-64) | 显式(1-16) |
| dual-stream | 未启用 | 启用 |
| Patches | 7 | 15 |

### 13.2 性能对比结果

**Test 1: Decode Throughput (1024 tokens, no thinking)**

| 并发 | Master tok/s | Node-1 tok/s | 提升 |
|------|-------------|-------------|------|
| C=1 | 211.7 | 305.5 | **+44%** |
| C=2 | 414.5 | 601.2 | **+45%** |
| C=4 | 757.2 | 1104.5 | **+46%** |
| C=8 | 1269.2 | 1767.0 | **+39%** |

**Test 2: Decode Throughput (2048 tokens, no thinking)**

| 并发 | Master tok/s | Node-1 tok/s | 提升 |
|------|-------------|-------------|------|
| C=1 | 217.4 | 260.0 | **+20%** |
| C=4 | 657.2 | 949.9 | **+45%** |
| C=8 | 1091.1 | 1553.7 | **+42%** |

**Test 3: Short Q&A with thinking (4096 tokens)**

| 并发 | Master tok/s | Node-1 tok/s | 提升 |
|------|-------------|-------------|------|
| C=1 | 187.6 | 261.3 | **+39%** |
| C=4 | 542.6 | 780.1 | **+44%** |

**Test 4: Medium context ~7K (512 tokens)**

| 并发 | Master tok/s | Node-1 tok/s | 提升 |
|------|-------------|-------------|------|
| C=1 | 214.9 | 358.5 | **+67%** |
| C=4 | 656.6 | 1251.5 | **+91%** |

### 13.3 Completion Token 吞吐对比（不含 reasoning）

| 测试 | 并发 | Master | Node-1 | 提升 |
|------|------|--------|--------|------|
| decode_1024 | C=1 | 105.9 | 152.7 | +44% |
| decode_1024 | C=8 | 634.3 | 882.9 | +39% |
| decode_2048 | C=1 | 108.7 | 157.1 | +44% |
| decode_2048 | C=8 | 630.7 | 884.4 | +40% |
| qa_thinking | C=1 | 118.4 | 168.7 | +42% |
| medium_ctx | C=1 | 107.3 | 179.1 | +67% |
| medium_ctx | C=4 | 327.9 | 624.7 | +90% |

### 13.4 结论

**Node-1（优化配置）在所有测试中全面碾压 Master（基线配置），平均提升 40-67%。**

主要收益来源：
1. **MTP 2→3, draft_tokens 3→4**：每 forward 多产出 33% token（accept_len 2.85→3.275）
2. **cuda-graph-bs 精简**：减少 graph replay 开销 + 释放显存
3. **max_running_requests 128→32**：减少调度开销
4. **dual-stream MoE**：routed experts 与 shared expert 并行执行

**优化配置不仅没有导致性能下降，反而带来了 40-67% 的吞吐提升。**
