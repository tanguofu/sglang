# amd-355-worker GLM-5.2-FP8 部署与验证脚本集

> **来源**: 从 `amd-355-worker`（`144.202.61.0`）容器 `glm52_prof_tilelang` 的 `/data/` 目录拉取
> **采集时间**: 2026-06-27
> **镜像**: `lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260626`
> **模型**: GLM-5.2-FP8（`GlmMoeDsaForCausalLM`，704G）
> **硬件**: 8 × AMD Instinct MI355X（gfx950，309GB VRAM/卡，full mesh XGMI）
> **部署形态**: TP8 / PP1 + MTP（NEXTN/EAGLE steps=2）+ DSA attention（tilelang）+ FP8 KV + 1M context

本目录保存了 worker 上用于**部署、patch、调优、benchmark、精度验证**的全部运维脚本（共 43 个），是 GLM-5.2-FP8 在 AMD MI355X 上 SGLang 部署的完整工具链备份。这些脚本以**容器内 in-place patch** 方式修改 SGLang 源码（`/sgl-workspace/sglang/python/...`），属于部署期 hack，**尚未全部上游化**。

> **相关文档**: [amd-355-worker 部署配置详解](https://iwiki.woa.com/p/4023020023)（iWiki）
> **配套 PD 部署脚本**: `scripts/pd_single_node/`（单机 PD 分离部署）
> **benchmark 总结**: `glm-5.2-amd355-benchmark.md`（仓库根目录）

---

## 一、脚本分类总览

| 类别 | 数量 | 说明 |
|------|------|------|
| A. 源码 Patch | 10 | in-container 修改 SGLang 源码（`patch_*.py` + `tilelang_tune.py`） |
| B. 一次性 Bug Fix | 9 | 针对特定 bug 的修复脚本（`fix_*.py`，多版本演进） |
| C. AITER 调优配置生成 | 4 | 生成密集 GEMM tuned config（`gen_*.py` + `add_*.py`） |
| D. Benchmark / 测量 / 精度 | 11 | 压测、测量、预热、精度验证 |
| E. 启动脚本 | 7 | 各部署形态的 docker 启动脚本（`start_*.sh`） |
| **合计** | **43** | （已排除 `dsa_backend_patched.py`、`model_config_patched.py` 两个运行时产物副本） |

---

## 二、A. 源码 Patch（`patch_*.py` + `tilelang_tune.py`）

这些脚本在容器启动后、`sglang.launch_server` 之前执行，直接修改 `/sgl-workspace/sglang/python/...` 下的源码。幂等设计，可重复运行。

| 脚本 | 修改文件 | 作用 | worker 是否应用 |
|------|----------|------|-----------------|
| `patch_glm_config.py` | `transformers` `GlmMoeDsaConfig.attribute_map` | 移除 `"head_dim": "qk_rope_head_dim"` 映射，防止 `head_dim=192` 覆盖 `qk_rope_head_dim=64` 导致 fused QKV 尺寸错误 | ✅ |
| `patch_dsa_backend.py` | `dsa_backend.py` | 用 `self.qk_rope_head_dim` / `self.qk_nope_head_dim` 替代 `layer.head_dim - layer.v_head_dim`（GLM-5.2 `v_head_dim=256=head_dim` 导致 0 维）+ 9× `.view()→.reshape()` | ✅ |
| `patch_dsa_backend_v2.py` | `dsa_backend.py` | **仅** `.view()→.reshape()`，保留原维度（v1 改维度，v2 是回退保守版） | — |
| `patch_aiter_decode.py` | `dsa_backend.py` | 修 AITER decode backend 3 bug：`q_scale=None→ones`、`o=new_empty` fp8→bfloat16（4 处） | ❌（worker 用 tilelang decode，未启用 aiter decode） |
| `patch_aiter_runtime.py` | `/tmp/aiter_configs/bf16_tuned_gemm.csv` | 运行时从 docker logs 读缺失 shape，找最近模板追加 | — |
| `patch_aiter_runtime_v2.py` | `/tmp/aiter_configs/bf16_tuned_gemm.csv` | 从 `/tmp/missing_shapes.txt` 读缺失 shape 追加 | — |
| `patch_eplb.py` | `expert_distribution.py` | EPLB `torch.distributed.reduce` 用 `moe_ep_group` 替代 world group（PP>1 死锁） | ❌（worker 无 EPLB） |
| `patch_model_config.py` | — | **no-op** 占位（修复在 dsa_backend.py，非 model_config.py） | — |
| `patch_mori_pp_kv_slices.py` | `disaggregation/mori/conn.py` | mori PP KV mem-desc 切片对齐（PD-1b 必需） | ❌（worker 非 PD） |
| `patch_pp_missing_layer.py` | `models/deepseek_v2.py` | PP>1 时 `PPMissingLayer.embedding_dim` AttributeError | ❌（worker PP1） |
| `tilelang_tune.py` | `dsa/tilelang_kernel.py` | 调 gfx950 tuning 参数（`block_I, threads, num_stages, block_per_cu, cu`），默认 `64,512,0,2,256` | — |

### 关键 patch 详解

**`patch_glm_config.py`（attribute_map 修复）**：上游 `GlmMoeDsaConfig.attribute_map` 含 `"head_dim": "qk_rope_head_dim"`，transformers 加载 config 时把 `head_dim=192` 写入 `qk_rope_head_dim`，覆盖正确的 64。修复后 `qk_rope_head_dim=64`、`qk_nope_head_dim=192`、`qk_head_dim=256`。

**`patch_dsa_backend.py`（GLM-5.2 v_head_dim 修复）**：GLM-5.2 config `v_head_dim=256`（= `qk_head_dim` 总维度），但 DSA backend 用 `layer.head_dim - layer.v_head_dim` 算 rope 维度会得 0。改用 `self.qk_rope_head_dim`（64）和 `self.qk_nope_head_dim`（192）。同时 9 处 `.view()→.reshape()`（q tensor 来自 MLA absorb 路径常非连续）。

> **注意**：`patch_dsa_backend.py`（v1，改维度）和 `patch_dsa_backend_v2.py`（v2，仅 reshape 保留维度）是两种方案。worker 实际跑的是 v1。上游分支 `fix/glm52-dsa-reshape-no-vhead-override` 是 v1 的进一步上游化（用 `self.qk_rope_head_dim` 替代 `layer.head_dim - layer.v_head_dim` + `o=bfloat16`）。

---

## 三、B. 一次性 Bug Fix（`fix_*.py`）

针对特定 bug 的修复脚本，部分已被 `patch_*.py` 或上游覆盖，保留作历史记录。

| 脚本 | 修改文件 | Bug | 状态 |
|------|----------|-----|------|
| `fix_bug_b.py` | `dsa_backend.py:~2604` | Bug B: `steps>3` 触发 multi-backend fused-copy JIT 路径，ROCm 找不到 `cuda_runtime.h`。加 `_USE_FUSED_METADATA_COPY` guard | worker 用 steps=2 规避 |
| `fix_dsa_page_size.py` | `server_args.py` | DSA 无条件强制 `page_size=64`，即使显式设 `--page-size 1`。改为仅未显式设置时覆盖（EAGLE3+topk>1+page_size=1） | — |
| `fix_eagle3_set_embed.py` | `deepseek_v2.py` | EAGLE3 worker 调 `set_embed`，NextN draft 无此方法 | v1 |
| `fix_eagle3_set_embed_v2.py` | `glm4_moe_nextn.py` | 同上，改 `Glm4MoeForCausalLMNextN` 加 `load_lm_head_from_target=True` | v2 |
| `fix_eagle3_set_embed_v3.py` | `deepseek_v2.py` | GLM-5.2 draft 映射到 `DeepseekV3ForCausalLMNextN`（非 Glm4Moe） | v3 |
| `fix_eagle3_set_embed_v4.py` | `deepseek_nextn.py` | `DeepseekV3ForCausalLMNextN` 实际在 `deepseek_nextn.py`（非 v2.py） | v4 |
| `fix_eagle3_set_embed_v5.py` | `deepseek_nextn.py` | 加 `load_lm_head_from_target=True` **且** `hot_token_id=None` | v5（最终） |
| `fix_fp8_all_gather.py` | `communicator.py` | NCCL 不支持 `float8_e4m3fn`，FP8 hidden_states cast bf16 再 all_gather | v1（cast 后未 cast 回） |
| `fix_fp8_all_gather_v2.py` | `communicator.py` | 同上 v2：cast bf16 all_gather 后 cast 回 FP8（attention 需 FP8 匹配 FP8 权重） | v2 |
| `fix_fp8_nccl.py` | NCCL `from_torch()` | **根因修复**：加 `float8_e4m3fn → ncclUint8` 映射，FP8 按 uint8 传输（lossless memcpy，仅 all_gather 非 reduce） | 根因 |

### fix_eagle3_set_embed 演进链

v1（deepseek_v2.py）→ v2（glm4_moe_nextn.py，找错 draft 类）→ v3（deepseek_v2.py，DeepseekV3NextN）→ v4（deepseek_nextn.py，正确文件）→ v5（+ `hot_token_id=None`，最终）。体现了逐步定位 draft model 类归属和所需属性的过程。

### fix_fp8 演进链

v1（cast bf16，未 cast 回）→ v2（cast 回 FP8）→ `fix_fp8_nccl.py`（根因：NCCL dtype 映射，uint8 lossless 传输）。v1/v2 是 workaround，`fix_fp8_nccl.py` 是根因修复。

---

## 四、C. AITER 调优配置生成（`gen_*.py` + `add_*.py`）

AITER（AMD GPU kernel 库）的 GEMM tuned config 原始仅覆盖少量 M 值，GLM-5.2 MoE expert GEMM 产生大量动态 M 值导致回退 PyTorch native。这些脚本生成密集调优条目。

| 脚本 | 目标 CSV | 覆盖范围 | 条目数 |
|------|----------|----------|--------|
| `gen_aiter_dense.py` | `glm5_bf16_tuned_gemm.csv` | K=6144, N=32/256, M=1–50000（步长 1） | +99,978 |
| `gen_a8w8_dense.py` | `glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv` | K=6144, N=128/2624/3072/6144, M=1–65536 | +262,096 |
| `add_aiter_glm5_shapes.py` | `glm5_bf16_tuned_gemm.csv` | 补缺失 M: 264,288,312,336,360,384,434,1953,3648（N=32/256） | 少量 |
| `add_aiter_shapes.py` | `/tmp/aiter_configs/bf16_tuned_gemm.csv` | 同上，通用 AITER config | 少量 |

**效果**：AITER "not found tuned config" 警告从 27,408 → 0。

---

## 五、D. Benchmark / 测量 / 精度（11 个）

| 脚本 | 用途 | 依赖 |
|------|------|------|
| `bench_glm52.py` | 统一 benchmark（asyncio + urllib，无外部依赖） | stdlib |
| `bench_unified.py` | 6 suite 并发 benchmark（对照 `glm-5.2-amd355-benchmark.md`） | stdlib |
| `bench_notok.py` | 无 tokenizer benchmark | stdlib |
| `simple_bench.py` | 简单流式 benchmark（测 TTFT/ITL） | aiohttp |
| `measure_decode.py` | decode 吞吐测量（`/generate`，长 prompt） | requests |
| `measure_4k.py` | 4k prompt 测量 | requests |
| `warmup_server.py` | 预热：预跑推理 JIT 编译 AITER kernel + 暖 cuda graph，消除冷启动 | requests |
| `accuracy_test.py` | 精度测试 v1（coding/math/reasoning，`/generate`） | requests |
| `accuracy_v2.py` | 精度测试 v2（raw generate API） | requests |
| `accuracy_chat.py` | 精度测试 v3（`/v1/chat/completions`，chat template） | requests |
| `accuracy_final.py` | 精度测试 v4（thinking mode max_tokens，更好抽取） | requests |

### Benchmark suite 定义（`bench_unified.py`）

| Suite | 用途 | 形状 |
|-------|------|------|
| `short_c32` | decode-heavy | 短 prompt, c=32, max_tok=128 |
| `short_c128` | 高并发 decode | 短 prompt, c=128, max_tok=128 |
| `mid_c32` | 混合 | 中 prompt, c=32, max_tok=512 |
| `prefill16k_c32` | 长 prompt prefill | ~16k, c=32, max_tok=32 |
| `prefill64k_c4` | 大 prompt 稳定性 | ~64k, c=4, max_tok=32 |
| `prefill128k_c1` | 1M context 检查 | ~128k, c=1, max_tok=32 |

### 冷启动 vs warm

`warmup_server.py` 必须在 benchmark 前运行。冷启动首请求 EAGLE draft cuda graph 内联捕获，慢 19×（11.5s vs warm 0.6s）。**MTP benchmark 必须 warm-run，冷启动数据无效。**

---

## 六、E. 启动脚本（`start_*.sh`）

| 脚本 | 部署形态 | 镜像 |
|------|----------|------|
| `start_sglang_glm52_tp8mtp.sh` | **TP8 + MTP**（worker 主部署） | v0.5.13.post1-20260623 |
| `start_baseline.sh` | TP4/PP2 基线 | v0.5.13.post1-20260623 |
| `start_dp8_ep8_mtp.sh` | DP8 + EP8 + MTP | v0.5.13.post1-20260623 |
| `start_eagle3.sh` / `v2` | EAGLE3 | v0.5.13.post1-20260623 |
| `start_mem090.sh` | mem_fraction 0.90 | v0.5.13.post1-20260623 |
| `start_test.sh` | 测试 | v0.5.13.post1-20260623 |

> **注意**：启动脚本里写的是 `v0.5.13.post1-rocm720-mi35x-20260623`，但 worker 实际跑的是更新的 `v0.5.14-rocm720-mi35x-20260626`（`glm52_prof_tilelang` 容器）。`v0.5.14` 已内置部分 patch，但启动仍跑 patch 脚本做幂等加固。

---

## 七、标准启动流程（worker 实测）

```bash
# 1. 进入容器（或 start_*.sh 内联）
docker exec -it glm52_prof_tilelang bash

# 2. 应用 patch（幂等，顺序执行）
python3 /data/patch_glm_config.py          # attribute_map 修复
python3 /data/patch_dsa_backend.py         # view->reshape + qk_rope_head_dim
python3 /data/gen_aiter_dense.py           # BF16 dense 调优配置
python3 /data/gen_a8w8_dense.py             # a8w8 调优配置

# 3. 启动 server（TP8/PP1 + MTP）
python3 -m sglang.launch_server \
  --model-path /data/models/GLM-5.2-FP8 \
  --tp-size 8 --pp-size 1 \
  --context-length 1048576 --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.88 \
  --enable-aiter-allreduce-fusion --enable-fused-qk-norm-rope \
  --enable-mixed-chunk --chunked-prefill-size 32768 \
  --schedule-conservativeness 0.5 \
  --prefill-max-requests 32 --max-prefill-tokens 32768 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 2 --speculative-num-draft-tokens 3 \
  --speculative-eagle-topk 1 \
  --max-running-requests 128 \
  --tool-call-parser glm47 --reasoning-parser glm45 \
  --watchdog-timeout 3600 --log-level info

# 4. 等待健康（约 8-10 分钟加载）
curl -sf http://127.0.0.1:30000/health

# 5. 预热（消除冷启动）
python3 /data/warmup_server.py

# 6. Benchmark
python3 /data/bench_glm52.py --url http://localhost:30000
```

---

## 八、与上游 SGLang 的关系

| Patch | 上游状态 |
|-------|----------|
| `patch_glm_config.py`（attribute_map） | 部分上游化（分支 `fix/glm52-dsa-reshape-no-vhead-override`） |
| `patch_dsa_backend.py`（view->reshape + qk_rope_head_dim） | 上游化中（同上分支，dsa_backend.py 已有未提交修改） |
| `patch_aiter_decode.py` | 未上游（aiter decode backend 未启用） |
| `fix_bug_b.py` | 未上游（steps>3 规避） |
| `fix_fp8_nccl.py` | 根因修复，待评估上游 |
| `fix_eagle3_set_embed_v5.py` | EAGLE3 相关，待上游 |
| `gen_*_dense.py` | 调优配置，非源码，不入上游 |
| `patch_mori_pp_kv_slices.py` | PD 场景，分支 `fix/mori-pp-kv-slices-pd-validation` |
| `patch_pp_missing_layer.py` | PP>1 场景 |

---

## 九、注意事项

1. **路径假设**：脚本内硬编码 `/sgl-workspace/sglang/python/...` 和 `/data/...`，仅在 SGLang ROCm 镜像内有效。
2. **幂等性**：patch 脚本可重复运行，会检测已应用状态。
3. **版本敏感**：patch 基于特定 SGLang commit，镜像升级后可能需调整匹配模式。
4. **排除文件**：`dsa_backend_patched.py`（110K）和 `model_config_patched.py`（79K）是运行时产物副本，未纳入本目录。
5. **worker 当前状态**：8 卡 VRAM ~302–307GB used（权重 + KV pool），compute 0% 热待命。不可再起第二个 TP8。
6. **冷启动**：MTP benchmark 必须 warm-run，否则数据无效（19× 慢）。
7. **tilelang vs aiter asm_mla**：worker 用 tilelang decode（稳定）。aiter `asm_mla` decode backend 在 cuda graph capture 时崩溃（`asm_mla.cu` persistent path `hipMalloc` 非法），见 [iWiki 4022990943](https://iwiki.woa.com/p/4022990943)。

---

## 十、相关链接

- **iWiki 部署详解**: https://iwiki.woa.com/p/4023020023
- **iWiki 父页（sglang-kernel 优化）**: https://iwiki.woa.com/p/4022910540
- **iWiki 部署父页**: https://iwiki.woa.com/p/4022389950
- **单机 PD 部署脚本**: `scripts/pd_single_node/`
- **benchmark 总结**: `glm-5.2-amd355-benchmark.md`（仓库根目录）
- **上游分支**: `fix/glm52-dsa-reshape-no-vhead-override`（dsa_backend.py 修复）
- **mori PP 修复分支**: `fix/mori-pp-kv-slices-pd-validation`
