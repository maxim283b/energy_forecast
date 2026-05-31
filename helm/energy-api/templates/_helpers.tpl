{{/*
Expand the name of the chart.
*/}}
{{- define "energy-api.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "energy-api.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "energy-api.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "energy-api.labels" -}}
helm.sh/chart: {{ include "energy-api.chart" . }}
{{ include "energy-api.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "energy-api.selectorLabels" -}}
app.kubernetes.io/name: {{ include "energy-api.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "energy-api.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "energy-api.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the MLflow component name.
*/}}
{{- define "energy-api.mlflowName" -}}
{{- printf "%s-mlflow" (include "energy-api.name" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create the MLflow fully qualified name.
*/}}
{{- define "energy-api.mlflowFullname" -}}
{{- printf "%s-mlflow" (include "energy-api.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
MLflow selector labels.
*/}}
{{- define "energy-api.mlflowSelectorLabels" -}}
app.kubernetes.io/name: {{ include "energy-api.mlflowName" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
MLflow common labels.
*/}}
{{- define "energy-api.mlflowLabels" -}}
helm.sh/chart: {{ include "energy-api.chart" . }}
{{ include "energy-api.mlflowSelectorLabels" . }}
app.kubernetes.io/component: mlflow
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}
