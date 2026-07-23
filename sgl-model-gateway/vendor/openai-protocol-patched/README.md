# openai-protocol-patched

Vendored copy of `openai-protocol` 1.0.0 with patches for Codex compatibility.

## Baseline

- Source: crates.io `openai-protocol` 1.0.0
- Imported via `[patch.crates-io]` in the parent `Cargo.toml`

## Patches applied

### 1. `ResponseToolType` enum expanded (src/responses.rs)

Upstream only defines 4 variants (`Function`, `WebSearchPreview`, `CodeInterpreter`,
`Mcp`). Codex sends additional tool types that would cause deserialization failures
at the gateway. Expanded to 12 variants:

- `Function`, `WebSearch`, `WebSearchPreview`, `CodeInterpreter`
- `FileSearch`, `ImageGeneration`, `ComputerUsePreview`, `LocalShell`
- `Mcp`, `Custom`, `Namespace`, `ToolSearch`

### 2. Lenient input deserializer (src/responses.rs)

Added `deserialize_input_lenient` function and `#[serde(deserialize_with)]` on
`ResponsesRequest.input`. When the typed `ResponseInput` parse fails, falls back
to `ResponseInput::Text(json_string)` so the request reaches the Python worker
(which handles arbitrary input formats) instead of being rejected at the Rust
gateway.

## Maintenance

When upgrading the upstream `openai-protocol` version, re-apply these two
patches to the new version. Track upstream changes to `ResponseToolType` and
`ResponseInput` — if upstream adds the missing tool types, patch #1 can be
dropped.
