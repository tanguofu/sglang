#!/usr/bin/env python3
"""Extract server tunables relevant to Codex (Responses API streaming) performance."""
import json, subprocess, sys

cmd = [
    "kubectl", "exec", "-n", "kube-system", "sglang-glm52-2tp8-sglang-0",
    "--", "/usr/bin/curl", "-sS", "--max-time", "15",
    "-H", "Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}",
    "http://127.0.0.1:30000/get_server_info",
]
out = subprocess.check_output(cmd, text=True)
info = json.loads(out)

# Group tunables by relevance to Codex streaming latency/throughput.
groups = {
    "Context & capacity": [
        "context_length", "max_total_num_tokens", "max_req_input_len",
        "max_prefill_tokens", "chunked_prefill_size", "max_running_requests",
        "max_queued_requests", "mem_fraction_static",
    ],
    "Scheduling": [
        "schedule_policy", "schedule_conservativeness", "retraction_policy",
        "prefill_max_requests", "enable_priority_scheduling",
        "priority_scheduling_preemption_threshold", "num_continuous_decode_steps",
        "scheduler_recv_interval", "enable_mixed_chunk",
    ],
    "Streaming (Codex uses stream=true)": [
        "stream_interval", "batch_notify_size",
        "stream_response_default_include_usage",
        "incremental_streaming_output",
        "enable_streaming_session",
        "enable_cache_report",
    ],
    "Speculative decoding (latency)": [
        "speculative_algorithm", "speculative_num_steps",
        "speculative_num_draft_tokens", "speculative_eagle_topk",
        "speculative_adaptive",
    ],
    "KV cache & hierarchy": [
        "kv_cache_dtype", "disable_radix_cache", "radix_eviction_policy",
        "swa_full_tokens_ratio", "disable_hybrid_swa_memory",
        "enable_hierarchical_cache", "hicache_ratio", "hicache_io_backend",
        "hicache_mem_layout", "hicache_write_policy",
        "enable_session_radix_cache",
    ],
    "Tool/reasoning parsing (Codex uses tools)": [
        "tool_call_parser", "reasoning_parser",
        "strip_thinking_cache", "enable_strict_thinking",
    ],
    "Concurrency / parallelism": [
        "tp_size", "dp_size", "pp_size", "ep_size", "moe_dp_size",
        "enable_dp_attention", "enable_two_batch_overlap",
        "enable_torch_compile", "torch_compile_max_bs",
    ],
    "CUDA graph (decode latency)": [
        "cuda_graph_bs_decode", "cuda_graph_max_bs_decode",
        "cuda_graph_bs_prefill", "cuda_graph_max_bs_prefill",
        "disable_decode_cuda_graph", "disable_prefill_cuda_graph",
        "cuda_graph_backend_decode", "cuda_graph_backend_prefill",
    ],
    "Watchdog & timeout (Codex long sessions)": [
        "watchdog_timeout", "soft_watchdog_timeout", "sleep_on_idle",
    ],
    "Defaults that affect sampling": [
        "sampling_defaults", "preferred_sampling_params",
        "allow_auto_truncate",
    ],
}

print("=" * 70)
print("Server tunables — relevance to Codex (Responses API streaming)")
print("=" * 70)
for group, keys in groups.items():
    print(f"\n## {group}")
    for k in keys:
        v = info.get(k, "<missing>")
        if isinstance(v, (list, dict)) and len(str(v)) > 80:
            v = str(v)[:77] + "..."
        print(f"  {k} = {v}")

# Also surface version + status
print("\n## Meta")
for k in ["version", "status", "served_model_name", "model_path",
          "quantization", "kv_cache_dtype", "dtype"]:
    print(f"  {k} = {info.get(k)}")
