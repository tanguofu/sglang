# Stage C 执行 runbook（CK GEMM decode-1 PoC + 灰度）

> 前置：镜像 `v0918-src-ckgemm` build + push 完成（后台任务 bnr7ajndp 通知）。
> 所有命令从 ops box（本机）执行。每步有门禁，不过即停。

## C.0 预检（已完成 2026-09-18 14:40）

- [x] decode-0 / decode-1 token_usage = 0.0，running = 0（空载窗口）
- [x] bench_poc_suite.py 已放 decode-1 /data
- [x] sts_poc_mode.py 从本机跑（kubectl 可用）
- [x] helper 逻辑在 decode-1 真实 CSV 上 5/5 验证通过

## C.1 PoC 基线（triton，现图 v0916-src-projfuse）

```bash
# 切 PoC 模式（:30000 -> :31000，router 自动摘除 decode-1）
python3 docker/rocm-mi308x-glm52-pd/scripts/sts_poc_mode.py poc statefulset/sglang-1p1d-decode-1

# 等 health（约 10min 重建 + 模型加载）
until curl -s -m 5 http://21.151.225.172:31000/health | grep -q OK; do sleep 30; done

# 基线 ×3（在 decode-1 pod 内跑，直连 :31000）
for i in 1 2 3; do
  kubectl exec -n kube-system sglang-1p1d-decode-1-0 -c sglang -- \
    python3 /data/bench_poc_suite.py --url http://21.151.225.172:31000 --label ckgemm-baseline-r$i
done
# 记录：needle ok / decode tps / accept_len（三个文件的中位）
```

## C.2 PoC CK（同机，只差 image + env）

```bash
# 导出 -> 改 image + env -> apply（PoC spec 下改，重建仍是 PoC）
kubectl get sts sglang-1p1d-decode-1 -n kube-system -o json > /tmp/decode-1-poc.json
python3 - <<'EOF'
import json
spec = json.load(open('/tmp/decode-1-poc.json'))
c = spec['spec']['template']['spec']['containers'][0]
c['image'] = 'mirrors.tencent.com/ti-platform/sglang-glm52-308x:v0918-src-ckgemm'
envs = c['env']
for e in envs:
    if e['name'] == 'SGLANG_ROCM_USE_CK_GEMM_GFX942':
        e['value'] = '1'
        break
else:
    envs.append({'name': 'SGLANG_ROCM_USE_CK_GEMM_GFX942', 'value': '1'})
for k in ('resourceVersion', 'generation', 'managedFields'):
    spec['metadata'].pop(k, None)
json.dump(spec, open('/tmp/decode-1-poc.json', 'w'))
print('patched')
EOF
kubectl replace -f /tmp/decode-1-poc.json
kubectl delete pod sglang-1p1d-decode-1-0 -n kube-system

# 等 health 后验证激活（关键证据：a8w8 CSV tuned 行，此前 4 pod 全为 0 条）
until curl -s -m 5 http://21.151.225.172:31000/health | grep -q OK; do sleep 30; done
kubectl logs -n kube-system sglang-1p1d-decode-1-0 -c sglang | \
  grep -c "is tuned on cu_num = 80 in /data/aiter_configs/a8w8_blockscale_tuned_gemm_glm5_1_cu80"
# 期望 > 0（bf16 行不算——grep 串含 a8w8_blockscale_tuned_gemm_glm5_1）

# PoC ×3
for i in 1 2 3; do
  kubectl exec -n kube-system sglang-1p1d-decode-1-0 -c sglang -- \
    python3 /data/bench_poc_suite.py --url http://21.151.225.172:31000 --label ckgemm-poc-r$i
done
```

## C.3 门禁（全过才继续）

| 检查 | 命令 | 阈值 |
|---|---|---|
| needle | poc-results-ckgemm-poc-r*.json 的 needle.ok | 3/3 True |
| decode tps | poc vs baseline 中位 | ≥ +2% |
| accept_len | poc vs baseline | \|Δ\| < 0.05 |
| 数值抽测 | pod 内跑 numcheck2.py | max rel ≤ 3% |
| 稳定 | 无 CrashLoop / NaN | - |

## C.4 恢复生产 + 灰度

```bash
# 恢复生产 spec（image 已是 v0918）
python3 docker/rocm-mi308x-glm52-pd/scripts/sts_poc_mode.py restore statefulset/sglang-1p1d-decode-1
kubectl delete pod sglang-1p1d-decode-1-0 -n kube-system
# 等 health + router 纳管，观察 2h（gen_throughput / accept_len / 5xx）

# decode-0（decode-1 稳定 2h 后）
kubectl get sts sglang-1p1d-decode -n kube-system -o json > /tmp/decode-0.json
# 同样 patch image + env，排空确认 token_usage < 0.55 后 delete pod
```

## 回滚（任一门禁不过）

```bash
# env 级（首选，10min）
kubectl get sts sglang-1p1d-decode-1 -n kube-system -o json | \
  python3 -c "import json,sys; s=json.load(sys.stdin); [e.update(value='0') for e in s['spec']['template']['spec']['containers'][0]['env'] if e['name']=='SGLANG_ROCM_USE_CK_GEMM_GFX942']; [s['metadata'].pop(k,None) for k in ('resourceVersion','generation','managedFields')]; json.dump(s,open('/tmp/rollback.json','w'))"
kubectl replace -f /tmp/rollback.json
kubectl delete pod sglang-1p1d-decode-1-0 -n kube-system
```
