{{- /*
Common environment variables for prefill and decode pods (ROCm/SGLang optimizations).
Expects dict with `ctx` (root $) and `hostIP`.
*/ -}}
{{- define "sglang.env.common" -}}
- name: SGLANG_HOST_IP
  value: "{{ .hostIP }}"
- name: MOONCAKE_PROTOCOL
  value: "{{ .ctx.Values.rdma.protocol }}"
- name: MC_GID_INDEX
  value: "{{ .ctx.Values.rdma.gidIndex }}"
- name: MC_DISABLE_HIP_TRANSPORT
  value: "1"
- name: HIP_VISIBLE_DEVICES
  value: "0,1,2,3,4,5,6,7"
- name: NCCL_DEBUG
  value: "WARN"
- name: HSA_ENABLE_SDMA
  value: "0"
- name: HIP_FORCE_DEV_KERNARG
  value: "1"
- name: NCCL_CUMEM_ENABLE
  value: "0"
- name: NCCL_MIN_NCHANNELS
  value: "112"
- name: NCCL_NVLS_ENABLE
  value: "0"
- name: PYTORCH_CUDA_ALLOC_CONF
  value: "expandable_segments:True"
- name: PYTORCH_ROCM_ARCH
  value: "gfx942"
- name: ROCM_QUICK_REDUCE_QUANTIZATION
  value: "INT8"
- name: SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN
  value: "1"
- name: SGLANG_DISABLE_CUDNN_CHECK
  value: "1"
- name: SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM
  value: "1"
- name: SGLANG_INT4_WEIGHT
  value: "0"
- name: SGLANG_MOE_PADDING
  value: "1"
- name: SGLANG_ROCM_DISABLE_LINEARQUANT
  value: "0"
- name: SGLANG_ROCM_FUSED_DECODE_MLA
  value: "1"
- name: SGLANG_SET_CPU_AFFINITY
  value: "1"
- name: SGLANG_USE_AITER
  value: "1"
- name: SGLANG_USE_ROCM700A
  value: "1"
- name: SGLANG_OPT_USE_TOPK_V2
  value: "0"
{{- end -}}

{{- /*
Common RDMA + GPU hostPath volumes for prefill and decode pods.
The patched sglang image ships a rebuilt libbnxt_re-rdmav34.so (238.1.138.5
compiled against Ubuntu 22.04 glibc 2.35 + rdma-core v34 API) that matches
the bnxt_re 238.x kernel module ABI on ts4 nodes. No host lib override needed.
*/ -}}
{{- define "sglang.volumes.rdma" -}}
- name: shm
  emptyDir:
    medium: Memory
    sizeLimit: 64Gi
- name: data
  hostPath:
    path: /data
    type: Directory
- name: dev-kfd
  hostPath:
    path: /dev/kfd
    type: CharDevice
- name: dev-dri
  hostPath:
    path: /dev/dri
    type: Directory
- name: dev-infiniband
  hostPath:
    path: /dev/infiniband
    type: Directory
{{- end -}}

{{- /* Common volume mounts for RDMA + GPU. */ -}}
{{- define "sglang.volumeMounts.rdma" -}}
- mountPath: /dev/shm
  name: shm
- mountPath: /data
  name: data
- mountPath: /dev/kfd
  name: dev-kfd
- mountPath: /dev/dri
  name: dev-dri
- mountPath: /dev/infiniband
  name: dev-infiniband
{{- end -}}

{{- /*
Init container that installs per-bond /30 static routes on the host so each
bondN's RDMA traffic to the peer node's bondN IP stays on bondN. Without this,
Linux picks the catch-all route `29.198.0.0/15 via bond7` for inter-node traffic
and RDMA QP RTR handshake times out because the QP is bound to bondN but packets
egress through bond7. Expects dict with `routes` (list of "dst via gw dev bondN"
strings). The pod runs with hostNetwork, so ip route commands hit the host netns.
*/ -}}
{{- define "sglang.init.routes" -}}
- name: setup-routes
  image: "{{ .ctx.Values.image.sglang }}:{{ .ctx.Values.image.tag }}"
  imagePullPolicy: {{ .ctx.Values.image.pullPolicy }}
  command: ["/bin/bash", "-c"]
  args:
  - |
    set -e
    echo "=== Installing per-bond static routes ==="
  {{- range .routes }}
    # Route: {{ . }}
    ip route replace {{ . }} 2>&1 || echo "WARN: route add failed: {{ . }}"
  {{- end }}
    echo "=== Routes installed ==="
    ip route | grep "29.199.73" | sort -t. -k4 -n
  securityContext:
    privileged: true
    capabilities:
      add: ["NET_ADMIN"]
{{- end -}}
