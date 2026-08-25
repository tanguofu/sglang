{{- define "sglang-1p1d.name" -}}
{{- default "sglang-1p1d" .Values.nameOverride -}}
{{- end -}}

{{- define "sglang-1p1d.labels" -}}
app.kubernetes.io/name: {{ include "sglang-1p1d.name" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "sglang-1p1d.ekletAnnotations" -}}
eks.tke.cloud.tencent.com/cluster-ip-switch: cluster
eks.tke.cloud.tencent.com/resolv-conf: |
  nameserver 9.137.197.120 9.137.192.34
tke.cloud.tencent.com/pod-type: eklet
{{- end -}}

{{- define "sglang-1p1d.mooncakeMasterAddr" -}}
{{ .Values.mooncake.master.ip }}:{{ .Values.mooncake.master.port }}
{{- end -}}

{{- define "sglang-1p1d.workerEnv" -}}
{{- $extra := .extra | default dict -}}
{{- range $k, $v := .root.Values.workerEnv }}
- name: {{ $k }}
  value: {{ $v | quote }}
{{- end }}
- name: SGLANG_HOST_IP
  value: {{ .hostIP | quote }}
{{- if .mooncakeLocal }}
- name: MOONCAKE_LOCAL_HOSTNAME
  value: {{ .hostIP | quote }}
- name: MOONCAKE_MASTER
  value: {{ include "sglang-1p1d.mooncakeMasterAddr" .root | quote }}
{{- end }}
{{- range $k, $v := $extra }}
- name: {{ $k }}
  value: {{ $v | quote }}
{{- end }}
{{- end -}}

{{- define "sglang-1p1d.prefillStsName" -}}
{{- if eq (int .index) 0 -}}
{{ include "sglang-1p1d.name" .root }}-prefill
{{- else -}}
{{ include "sglang-1p1d.name" .root }}-prefill-{{ .index }}
{{- end -}}
{{- end -}}

{{- define "sglang-1p1d.decodeStsName" -}}
{{- if eq (int .index) 0 -}}
{{ include "sglang-1p1d.name" .root }}-decode
{{- else -}}
{{ include "sglang-1p1d.name" .root }}-decode-{{ .index }}
{{- end -}}
{{- end -}}

{{- define "sglang-1p1d.podFQDN" -}}
{{ .pod }}.{{ .svc }}.{{ .ns }}.svc.cluster.local
{{- end -}}

{{- define "sglang-1p1d.prefillFQDN" -}}
{{- $sts := include "sglang-1p1d.prefillStsName" . -}}
{{- include "sglang-1p1d.podFQDN" (dict "pod" (printf "%s-0" $sts) "svc" $sts "ns" .root.Release.Namespace) -}}
{{- end -}}

{{- define "sglang-1p1d.decodeFQDN" -}}
{{- $sts := include "sglang-1p1d.decodeStsName" . -}}
{{- include "sglang-1p1d.podFQDN" (dict "pod" (printf "%s-0" $sts) "svc" $sts "ns" .root.Release.Namespace) -}}
{{- end -}}

{{- define "sglang-1p1d.mooncakeMasterDNS" -}}
{{ include "sglang-1p1d.name" . }}-mooncake-master.{{ .Release.Namespace }}.svc.cluster.local:{{ .Values.mooncake.master.port }}
{{- end -}}

{{- define "sglang-1p1d.hicacheArgs" -}}
{{- if .Values.hicache.enabled }} --enable-hierarchical-cache --hicache-ratio {{ .Values.hicache.ratio }} --hicache-storage-backend {{ .Values.hicache.storageBackend }} --hicache-storage-prefetch-policy {{ .Values.hicache.prefetchPolicy }} --hicache-mem-layout {{ .Values.hicache.memLayout }} --hicache-write-policy {{ .Values.hicache.writePolicy }}{{- end -}}
{{- end -}}

{{- define "sglang-1p1d.launchArgs" -}}
{{- $root := .root -}}
{{- range $root.Values.commonArgs }} {{ . }}{{- end -}}
{{- range .extra }} {{ . }}{{- end }} --port {{ .port }} --disaggregation-ib-device {{ $root.Values.ibDevice | quote }} --disaggregation-bootstrap-port {{ .bootstrapPort }}
{{- end -}}

{{- define "sglang-1p1d.workerVolumes" -}}
- name: shm
  emptyDir:
    medium: Memory
    sizeLimit: 64Gi
- name: data
  hostPath:
    path: {{ .Values.hostPathData }}
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

{{- define "sglang-1p1d.workerVolumeMounts" -}}
- name: shm
  mountPath: /dev/shm
- name: data
  mountPath: /data
- name: dev-kfd
  mountPath: /dev/kfd
- name: dev-dri
  mountPath: /dev/dri
- name: dev-infiniband
  mountPath: /dev/infiniband
{{- end -}}

{{- define "sglang-1p1d.securityContext" -}}
allowPrivilegeEscalation: true
privileged: true
readOnlyRootFilesystem: false
seccompProfile:
  type: Unconfined
{{- end -}}

{{- define "sglang-1p1d.preflight" -}}
set -euo pipefail
echo "========== Environment Pre-Flight Check =========="
echo "Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Hostname:  $(hostname)"
GPU_COUNT=$(rocminfo 2>/dev/null | grep -c "^  Name:.*gfx" || true)
echo "GPU count: ${GPU_COUNT}"
if [ "${GPU_COUNT}" -lt 1 ]; then
  echo "FATAL: No GPU detected via rocminfo" >&2
  exit 1
fi
MODEL_PATH="{{ .Values.modelPath }}"
if [ ! -d "${MODEL_PATH}" ] || [ ! -f "${MODEL_PATH}/config.json" ]; then
  echo "FATAL: Model ${MODEL_PATH} missing" >&2
  exit 1
fi
echo "Model path: ${MODEL_PATH} size=$(du -sh "${MODEL_PATH}" 2>/dev/null | awk '{print $1}')"
IB_ACTIVE=$(ibv_devinfo 2>&1 | grep -c "PORT_ACTIVE" || true)
echo "Active IB ports: ${IB_ACTIVE}"
if [ "${IB_ACTIVE}" -lt 1 ]; then
  echo "WARNING: No active IB ports — PD may fall back to TCP" >&2
fi
echo "========== Pre-Flight Check Complete =========="
{{- end -}}
