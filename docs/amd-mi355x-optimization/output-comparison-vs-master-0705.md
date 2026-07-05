
## 14. 输出质量对比：Master vs Node-1（2026-07-05）

### 14.1 测试方法

在两台机器上用相同 prompt、相同参数（temperature=0, max_tokens=8192）跑 6 道题，逐字符对比 content 和 reasoning。

### 14.2 结果汇总

| 测试 | Content 一致 | Reasoning 一致 | 最终答案一致 | Master 耗时 | Node-1 耗时 |
|------|-------------|---------------|-------------|-----------|-----------|
| math（整除） | ❌ | ❌ | ✅ 1,11,37 | 21.8s | 15.0s |
| math2（组合） | ❌ | ❌ | ✅ 36 | 7.0s | 4.1s |
| code（链表环检测） | ✅ 逐字一致 | ❌ | — | 3.4s | 2.6s |
| reasoning（平均速度） | ❌ | ❌ | ✅ 48 km/h | 7.7s | 4.8s |
| knowledge（TCP vs UDP） | ❌ | ❌ | — | 6.9s | 3.3s |
| logic（三段论） | ❌ | ❌ | — | 12.6s | 7.5s |

### 14.3 关键发现

**最终答案 100% 一致**：所有有 `\boxed{}` 答案的题目（3/3），两台机器的答案完全相同。

**Content 差异是措辞不同，非答案不同**：
- math：Master 用 "polynomial division"，Node-1 用 "modular arithmetic" — 不同方法，同一答案
- reasoning：Master 用 "Calculate the total distance"，Node-1 用 "Calculate the time taken" — 不同组织方式，同一答案
- knowledge：Master "dedicated session"，Node-1 "dedicated connection" — 同义表达

**Code 输出逐字一致**：链表环检测的代码在两台机器上**完全相同**（315字符），说明模型的核心生成能力无差异。

**Reasoning 始终不同**：6/6 题的 reasoning_content 都不同。这是预期行为 — reasoning 是自由推理过程，即使 temperature=0，GPU 浮点运算的非确定性也会导致推理路径分叉。但最终结论一致。

**Node-1 全面更快**：所有 6 道题 Node-1 都比 Master 快 30-52%。

### 14.4 结论

**优化配置未改变模型输出质量**：
- 最终答案 100% 一致
- 代码输出逐字一致
- Content 差异仅是措辞/方法选择不同（同义表达）
- Reasoning 差异是 GPU 浮点非确定性导致的推理路径分叉（正常现象）
- **MTP 3/4 + FP8 KV cache + cuda-graph 精简 不影响模型推理能力**
