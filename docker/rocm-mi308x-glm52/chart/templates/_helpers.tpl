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
Note: Kept empty to match live StatefulSet/Deployment selectors which were
simplified to just `app: sglang` / `app: sglang-router` (immutable fields).
Templates add the `app` key explicitly after this helper.
Service selectors are mutable and use `sglang.serviceSelectorLabels` instead
to keep per-release isolation via app.kubernetes.io/instance.
*/}}
{{- define "sglang.selectorLabels" -}}
{{- end -}}

{{/*
Service selector labels — includes app.kubernetes.io/instance for per-release
isolation (each Service only selects pods of its own release).
*/}}
{{- define "sglang.serviceSelectorLabels" -}}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Full image reference — worker (StatefulSet) container.
*/}}
{{- define "sglang.image" -}}
{{ .Values.image }}:{{ .Values.tag }}
{{- end -}}

{{/*
Router image reference — uses router.image/router.tag when set, otherwise
falls back to the worker image. Lets the chart deploy a separate sgl-model-gateway
image (e.g. one with /v1/messages support) without patching the live Deployment.
*/}}
{{- define "sglang.router.image" -}}
{{- if and .Values.router.image .Values.router.tag -}}
{{ .Values.router.image }}:{{ .Values.router.tag }}
{{- else -}}
{{ include "sglang.image" . }}
{{- end -}}
{{- end -}}
