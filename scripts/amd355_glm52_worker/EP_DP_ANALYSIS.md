# EP+DP 部署测试结果 (MoriEP)

## 配置
- DP=8 + EP=8 + MoriEP low_latency + FP8 dispatch
- HSA_ENABLE_SDMA=1, MORI_ENABLE_SDMA=1, --enable-p2p-check
- KV Cache: 115.44 GB (2.24M tokens), 支持 ~2.6 个 1M context 并发
- CUDA graph: 启用, MTP: 正常工作 (accept rate 0.55-0.89)

## 性能对比
| Test | TP=8 baseline | EP+DP=8 | 变化 |
|------|-------------|---------|------|
| decode_short c1 | 170.5 | 61.6 | -64% |
| decode_short c8 | 921.4 | 500 error (timeout) | N/A |

## 根因分析
EP A2A 在低并发 (batch_size=1) 下比 TP all-reduce 慢得多:
1. **A2A 延迟主导**: batch_size=1 + top-8 experts, 每层需 16 次 P2P 传输 (dispatch+combine), 75 MoE 层 = 1200 次传输/forward
2. **TP all-reduce 是单次优化集合通信**: NCCL/AITER ring/tree 对小数据高度优化
3. **EP 设计目标是大 batch 高吞吐**: A2A 的 setup overhead 需要大 batch 来摊销

## 结论
- **低并发场景 (Codex/Claude Code)**: TP=8 是最优, EP 反而更慢
- **高并发场景 (32+ 并发)**: EP 可能更优, 需要进一步测试
- **通信瓶颈是 TP=8 的固有代价**: 但已经是低并发下的最优解
