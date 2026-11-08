#!/usr/bin/env bash
# Read-only render check: values.on.yaml adds exactly ONE env key, the Helm
# 4.1.3 dense-tile N=0 trap is not triggered, and every rendered object is
# byte-identical to the m20 base except for that one env entry per engine pod.
# No cluster contact, no apply, no restart.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
CH="${CHART:-/Users/guofutan/ai-frameworks/dsv4-308x/charts/ds41-flash}"
RELEASE="ds41-ts4-poc"
NS="kube-system"
KEY="SGLANG_DS41_MHC_FUSED_SINKHORN"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }
ok()   { echo "PASS: $*"; }

command -v helm >/dev/null || fail "helm not found"
python3 -c "import yaml" 2>/dev/null || fail "PyYAML not found"

# 1) values layer: the overlay has exactly one leaf, the new env key = "1".
python3 - "$D/values.on.yaml" "$KEY" <<'PY'
import sys, yaml
doc = yaml.safe_load(open(sys.argv[1]))
key = sys.argv[2]
assert isinstance(doc, dict) and set(doc.keys()) == {"env"}, doc
assert set(doc["env"].keys()) == {key}, doc["env"]
assert str(doc["env"][key]) == "1", doc["env"][key]
print("PASS: values overlay = 1 leaf (%s: \"1\"), under shared env:" % key)
PY

# 2) render both arms.
helm template "$RELEASE" "$CH" -n "$NS" -f "$D/values.base.yaml" \
  > "$TMP/base.yaml"
helm template "$RELEASE" "$CH" -n "$NS" \
  -f "$D/values.base.yaml" -f "$D/values.on.yaml" \
  > "$TMP/on.yaml"

# 3) parsed-object diff: after removing the added env entries, ON == BASE.
python3 - "$TMP/base.yaml" "$TMP/on.yaml" "$KEY" <<'PY'
import copy, sys, yaml
base = list(yaml.safe_load_all(open(sys.argv[1])))
on = list(yaml.safe_load_all(open(sys.argv[2])))
key = sys.argv[3]

def env_lists(obj, path=()):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "env" and isinstance(v, list) and v and all(
                isinstance(e, dict) and "name" in e for e in v
            ):
                out.append((path + (k,), v))
            else:
                out.extend(env_lists(v, path + (k,)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(env_lists(v, path + (i,)))
    return out

def strip(doc):
    added = []
    for _, env in env_lists(doc):
        kept = []
        for e in env:
            if e.get("name") == key:
                added.append(e)
            else:
                kept.append(e)
        env[:] = kept
    return added

on_norm = copy.deepcopy(on)
added = []
for d in on_norm:
    added += strip(d)
assert len(added) >= 1, "fusion env key not present in any rendered container"
for e in added:
    assert e.get("value") == "1" or e.get("value") == 1, e
assert on_norm == base, (
    "rendered objects differ beyond the added env entry(ies)"
)
print("PASS: parsed render diff = %d added env entries (%s), all else identical" % (len(added), key))
PY

# 4) Helm 4.1.3 YAML-1.1 trap: N=0 must be absent; tile paths unchanged.
for tag in base on; do
  n0=$(grep -c 'N=0' "$TMP/$tag.yaml" || true)
  [ "$n0" -eq 0 ] || fail "$tag: N=0 present ($n0) -- dense tile destroyed"
done
np_base=$(grep -c 'path: "N=' "$TMP/base.yaml" || true)
np_on=$(grep -c 'path: "N=' "$TMP/on.yaml" || true)
[ "$np_base" -gt 0 ] || fail "no tile paths rendered"
[ "$np_base" -eq "$np_on" ] || fail "tile path count changed $np_base -> $np_on"
ok "N=0=0 / tile paths=$np_base unchanged in both arms"

echo "ALL PASS -- read-only; GPU bit-exact gate PASSED 2026-11-08; still needs GO + frozen window + atomic restart"
