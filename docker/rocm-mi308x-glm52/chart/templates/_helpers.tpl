{{/*
Common labels
*/}}
{{- define "sglang.labels" -}}
app.kubernetes.io/name: sglang-glm52-308x
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
app: sglang
accelerator: amd-gpu
{{- end -}}

{{/*
Full image reference
*/}}
{{- define "sglang.image" -}}
{{ .Values.image }}:{{ .Values.tag }}
{{- end -}}
