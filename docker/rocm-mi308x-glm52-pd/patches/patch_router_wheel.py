#!/usr/bin/env python3
"""Patch sgl-model-gateway for /v1/responses on HTTP PDRouter + accept codex
non-standard tool types (custom, namespace, ...).

Run AFTER cargo fetch has downloaded the openai-protocol crate to the registry.
"""
import os, shutil, subprocess, sys

GW = "/sgl-workspace/sglang/sgl-model-gateway"

# 1. route_responses on PDRouter (batch_size: None — string bootstrap, not list)
p = f"{GW}/src/routers/http/pd_router.rs"
s = open(p).read()
imp = "use super::pd_types::api_path;"
if "ResponsesRequest" not in s:
    assert imp in s, "api_path import not found"
    s = s.replace(imp, imp + "\nuse crate::protocols::responses::ResponsesRequest;", 1)
anchor = "    async fn route_completion("
method = '''    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream.unwrap_or(false);
        let context = PDRequestContext {
            route: "/v1/responses",
            batch_size: None,
            is_stream,
            return_logprob: false,
            request_text: None,
            model_id,
            headers: headers.cloned(),
        };
        self.execute_dual_dispatch(headers, body, context).await
    }

'''
if "async fn route_responses(" not in s:
    assert s.count(anchor) == 1, "route_completion anchor not found"
    s = s.replace(anchor, method + anchor, 1)
    open(p, "w").write(s)
    print("[OK] route_responses added to PDRouter (batch_size: None)")
else:
    print("[OK] route_responses already present")

# 2. Patch openai-protocol crate: add ALL tool types that codex + the Python
#    worker accept, so the Rust router doesn't reject or mangle them.
#    Python worker (protocol.py) accepts: function, web_search, web_search_preview,
#    code_interpreter, file_search, image_generation, computer_use_preview,
#    local_shell, mcp, custom, namespace, tool_search.
#    We must NOT use #[serde(other)] because it serializes unknown types as
#    "other", which the Python worker rejects. List all types explicitly.
import glob
crate_files = glob.glob("/root/.cargo/registry/src/*/openai-protocol-1.0.0/src/responses.rs")
assert crate_files, "openai-protocol crate not found in registry (run cargo fetch first)"
crate_src = crate_files[0]
crate_dir = os.path.dirname(os.path.dirname(crate_src))  # .../openai-protocol-1.0.0/
patched_dir = "/sgl-workspace/openai-protocol-patched"
if os.path.exists(patched_dir):
    shutil.rmtree(patched_dir)
shutil.copytree(crate_dir, patched_dir)
rp = f"{patched_dir}/src/responses.rs"
s = open(rp).read()
old = """pub enum ResponseToolType {
    Function,
    WebSearchPreview,
    CodeInterpreter,
    Mcp,
}"""
new = """pub enum ResponseToolType {
    Function,
    WebSearch,
    WebSearchPreview,
    CodeInterpreter,
    FileSearch,
    ImageGeneration,
    ComputerUsePreview,
    LocalShell,
    Mcp,
    Custom,
    Namespace,
    ToolSearch,
}"""
if "Custom" not in s:
    assert old in s, "ResponseToolType enum not found"
    s = s.replace(old, new, 1)
    open(rp, "w").write(s)
    print("[OK] openai-protocol patched: 12 tool types")
else:
    print("[OK] openai-protocol tool types already patched")

# 2b. ResponseInput: codex may send input formats the crate doesn't accept.
#     Make ResponseInputOutputItem accept unknown item types via #[serde(other)].
#     This lets the Items variant match codex's items without breaking matches.
import re
s2 = open(rp).read()
# Find ResponseInputOutputItem enum and add #[serde(other)] Other if not present
m = re.search(r'(pub enum ResponseInputOutputItem \{[^}]+)\}', s2)
if m and "Other" not in m.group(0) and "#[serde(other)]" not in m.group(0):
    old_item = m.group(0)
    new_item = old_item[:-1] + "    #[serde(other)]\n    UnknownItem,\n}"
    s2 = s2.replace(old_item, new_item, 1)
    open(rp, "w").write(s2)
    print("[OK] ResponseInputOutputItem: added #[serde(other)] UnknownItem catch-all")
elif "#[serde(other)]" in s2:
    print("[OK] ResponseInputOutputItem already has catch-all")
else:
    print("[WARN] ResponseInputOutputItem enum not found (may be a struct)")

# 3. [patch.crates-io] in Cargo.toml
ct = f"{GW}/Cargo.toml"
s = open(ct).read()
if "patch.crates-io" not in s:
    s += '\n[patch.crates-io]\nopenai-protocol = { path = "/sgl-workspace/openai-protocol-patched" }\n'
    open(ct, "w").write(s)
    print("[OK] [patch.crates-io] added to Cargo.toml")

# 4. Add match arms for all new ResponseToolType variants in builder.rs
bp = f"{GW}/src/routers/grpc/harmony/builder.rs"
s = open(bp).read()
if "ResponseToolType::ToolSearch" not in s:
    s = s.replace(
        'ResponseToolType::Mcp => "mcp",',
        'ResponseToolType::Mcp => "mcp",\n'
        '                            ResponseToolType::WebSearch => "web_search",\n'
        '                            ResponseToolType::FileSearch => "file_search",\n'
        '                            ResponseToolType::ImageGeneration => "image_generation",\n'
        '                            ResponseToolType::ComputerUsePreview => "computer_use_preview",\n'
        '                            ResponseToolType::LocalShell => "local_shell",\n'
        '                            ResponseToolType::Custom => "custom",\n'
        '                            ResponseToolType::Namespace => "namespace",\n'
        '                            ResponseToolType::ToolSearch => "tool_search",',
        1)
    open(bp, "w").write(s)
    print("[OK] builder.rs match arms added (all 12 tool types)")

print("\n=== All router patches applied ===")
