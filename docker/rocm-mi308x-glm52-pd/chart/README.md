# GLM-5.2 MI308X 3P1D Helm chart

3 prefill + 1 decode（GDR PD 传输 + DSA tilelang + NEXTN 2/3）。Prefill 开 HiCache：L1 GPU radix、L2 host（ratio 1）、L3 Mooncake DRAM 池（三台 P 共享前缀 KV）。

Live STS 曾被 `kubectl apply/patch` 改过字段 manager，升级必须带 `--force-conflicts`：

```bash
helm upgrade --install sglang-1p1d docker/rocm-mi308x-glm52-pd/chart \
  -n kube-system --force-conflicts
```

改镜像或参数只动 `values.yaml` 后再 upgrade。不要 `helm install` 第二个 PD release。

节点：P `144/152/172`，D `132`，Mooncake master 在 `132:50051`。不要把 `--hicache-ratio` 调到 4。公网 HTTPRoute 直连 router，无 LiteLLM。
