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
Selector labels — minimal labels for pod selection (shared across components)
*/}}
{{- define "sglang.selectorLabels" -}}
app.kubernetes.io/name: sglang-glm52-308x
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Full image reference
*/}}
{{- define "sglang.image" -}}
{{ .Values.image }}:{{ .Values.tag }}
{{- end -}}
