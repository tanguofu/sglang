# Bug: `/v1/responses` non-streaming returns 400 through Rust router

## Status
- **Open** (not fixed; requires upstream router image rebuild)
- Discovered: 2026-07-18
- Workaround: Use streaming (`"stream": true`) for `/v1/responses` through the router.

## Summary
Non-streaming POST requests to `/v1/responses` through the sglang Rust router
return HTTP 400 with a Pydantic validation error. Streaming requests to the same
endpoint work fine. The endpoint works correctly when hit directly on the worker
(bypassing the router).

## Environment
- Router image: `messages-0717c` (revision 28 of `deployment/sglang-glm52-2tp8-router`)
- Router binary: `/opt/venv/lib/python3.10/site-packages/sglang_router/sglang_router_rs.abi3.so` (v0.3.2, 42MB)
- Worker: sglang server on port 30000 (works correctly for both streaming and non-streaming)

## Reproduction

### Failing request (non-streaming, through router)
```bash
curl -X POST "https://glm52-2tp8.jmpti.woa.com/v1/responses" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.2[1M]","input":"hi","max_output_tokens":10}'
```

### Response
```json
{
  "error": {
    "message": "1 validation error for ChatCompletionRequest\nstream\n  Input should be a valid boolean [type=bool_type, input_value=None, input_type=NoneType]\n    For further information visit https://errors.pydantic.dev/2.13/v/bool_type None",
    "type": "invalid_request_error",
    "param": null,
    "code": 400
  }
}
```

### Working request (streaming, through router)
Same request with `"stream": true` returns 200.

### Working request (non-streaming, direct to worker)
Bypassing the router and hitting the worker directly on port 30000 works for
both streaming and non-streaming — confirming the bug is in the router, not the
worker's `/v1/responses` handler.

## Root Cause
The Rust router (`sglang_router_rs`) has a hardcoded path whitelist that includes
`/v1/responses`. When a request hits this path, the router validates the request
body against a `ChatCompletionRequest` schema that requires `stream` to be a
boolean (`true` or `false`). Non-streaming OpenAI Responses API clients omit the
`stream` field entirely (or send `null`), which the router's validator rejects.

The `/v1/responses` endpoint is a distinct OpenAI API that uses different
request/response shapes than `/v1/chat/completions`, but the router appears to
validate both with the same `ChatCompletionRequest` schema — which is too strict
for the Responses API.

## Why Not Fixed via HTTPRoute Bypass
The other router-unsupported paths (`/v1/messages/count_tokens`, `/metrics`,
`/get_server_info`, `/get_model_info`, `/flush_cache`, `/engine_metrics`) were
fixed by adding HTTPRoute rules that route those path prefixes directly to the
worker service, bypassing the router entirely. This works because those paths
are not in the router's whitelist at all (router returns 404).

For `/v1/responses` non-streaming, an HTTPRoute bypass is NOT viable because:
1. The same path `/v1/responses` must serve both streaming (works on router) and
   non-streaming (broken on router) requests — HTTPRoute can't distinguish them
   by request body.
2. Routing all `/v1/responses` traffic to the worker would lose the router's
   load balancing, cache-aware routing, and worker health checking for Responses
   API traffic.

## Recommended Fix
Rebuild the router image with one of:
1. **Relax the validation**: Make `stream` accept `None`/omitted in the router's
   `ChatCompletionRequest` schema (treat as `false`).
2. **Separate schemas**: Use a separate `ResponsesRequest` schema for
   `/v1/responses` validation instead of reusing `ChatCompletionRequest`.
3. **Skip validation for Responses**: Don't validate `/v1/responses` requests in
   the router — pass them through to the worker unchanged (the worker already
   validates them correctly).

Option 3 is the lowest-risk fix and matches the router's treatment of unknown
paths (which are passed through without validation).

## Impact
- Claude Code (Anthropic Messages API): NOT affected — uses `/v1/messages`, not
  `/v1/responses`.
- OpenAI Codex CLI and other Responses API clients that use non-streaming
  requests: affected. Must use streaming mode through the gateway.
- OpenAI Chat Completions API clients: NOT affected — use `/v1/chat/completions`.

## Verification
After the router image is rebuilt, run:
```bash
curl -X POST "https://glm52-2tp8.jmpti.woa.com/v1/responses" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.2[1M]","input":"hi","max_output_tokens":10}'
```
Should return 200 with a `ResponsesResponse` JSON body.
