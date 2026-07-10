#!/usr/bin/env python3
"""Long context stress test for GLM-5.2 on MI308X (v13, no MTP).
Tests gradually: 4K -> 32K -> 128K -> 512K -> 1M.
Captures both content and reasoning_content.
"""
import requests
import time
import json
import subprocess
import sys

API_URL = "http://127.0.0.1:30000"
API_KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL = "glm-5.2"
RESULTS_FILE = "/tmp/long_context_results.json"
LOG_FILE = "/tmp/long_context_test.log"

TESTS = [
    {"name": "4K",   "input_tokens": 4096,    "output_tokens": 128},
    {"name": "32K",  "input_tokens": 32768,   "output_tokens": 128},
    {"name": "128K", "input_tokens": 131072,  "output_tokens": 128},
    {"name": "512K", "input_tokens": 524288,  "output_tokens": 64},
    {"name": "1M",   "input_tokens": 1048576, "output_tokens": 32},
]

NEEDLE = "The secret number is 42."


def log(msg):
    line = "[{}] {}".format(time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def wait_health(max_wait=900):
    for i in range(max_wait // 5):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            if r.status_code == 200:
                return i * 5
        except:
            pass
        time.sleep(5)
    return -1


def ensure_server_ready():
    elapsed = wait_health(900)
    if elapsed >= 0:
        log("Server health OK after {}s".format(elapsed))
        time.sleep(10)
        return True
    log("Server not ready after timeout")
    return False


def get_vram_usage():
    try:
        proc = subprocess.Popen(
            ["amd-smi", "monitor", "--json"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stdout, _ = proc.communicate()
        data = json.loads(stdout.decode("utf-8"))
        gpus = data if isinstance(data, list) else data.get("system", {}).get("gpus", [])
        total_used = 0.0
        total_total = 0.0
        for g in gpus:
            vram_used_obj = g.get("vram_used", {})
            vram_total_obj = g.get("vram_total", {})
            used = float(vram_used_obj.get("value", 0)) if isinstance(vram_used_obj, dict) else float(vram_used_obj)
            total = float(vram_total_obj.get("value", 0)) if isinstance(vram_total_obj, dict) else float(vram_total_obj)
            total_used += used
            total_total += total
        return {"total_used_gb": round(total_used, 1), "total_total_gb": round(total_total, 1)}
    except Exception as e:
        return {"error": str(e)}


def tokenize_text(text):
    try:
        r = requests.post(f"{API_URL}/tokenize",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": MODEL, "prompt": text}, timeout=30)
        if r.status_code == 200:
            d = r.json()
            return d.get("count", d.get("len", 0))
    except:
        pass
    return 0


def generate_text(target_tokens):
    base = ("The quick brown fox jumps over the lazy dog. "
            "This is a test of the long context capability of the GLM model. ")
    cnt = tokenize_text(base)
    ratio = len(base) / cnt if cnt and cnt > 0 else 4.0
    target_chars = int(target_tokens * ratio)
    reps = target_chars // len(base) + 1
    text = (base * reps)[:target_chars]
    insert_pos = int(len(text) * 0.6)
    text = text[:insert_pos] + " " + NEEDLE + " " + text[insert_pos:]
    return text


def run_test(config, results):
    name = config["name"]
    input_tokens = config["input_tokens"]
    output_tokens = config["output_tokens"]

    log("=" * 60)
    log("Testing {} context ({} input + {} output)".format(name, input_tokens, output_tokens))
    log("=" * 60)

    if not ensure_server_ready():
        result = {"name": name, "status": "SERVER_NOT_READY", "input_tokens": input_tokens}
        results.append(result)
        save_results(results)
        return

    vram_before = get_vram_usage()
    log("VRAM before: {}GB / {}GB".format(vram_before.get("total_used_gb", "?"), vram_before.get("total_total_gb", "?")))

    text = generate_text(input_tokens)
    log("Generated {} chars for {} target tokens".format(len(text), input_tokens))

    prompt = text + "\n\nQuestion: What is the secret number mentioned in the text above? Answer with just the number."

    request_timeout = max(600, input_tokens // 100)
    start_time = time.time()
    first_token_time = None
    full_content = ""
    full_reasoning = ""

    try:
        response = requests.post(
            f"{API_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": output_tokens,
                "stream": True,
                "temperature": 0.0,
            },
            stream=True,
            timeout=request_timeout,
        )

        if response.status_code != 200:
            error_text = response.text[:500]
            log("ERROR: HTTP {}: {}".format(response.status_code, error_text))
            vram_after = get_vram_usage()
            result = {
                "name": name, "input_tokens": input_tokens, "output_tokens": output_tokens,
                "status": "FAILED",
                "error": "HTTP {}: {}".format(response.status_code, error_text[:200]),
                "oom": "out of memory" in error_text.lower() or "oom" in error_text.lower(),
                "vram_before_gb": vram_before.get("total_used_gb"),
                "vram_after_gb": vram_after.get("total_used_gb"),
            }
            results.append(result)
            save_results(results)
            return

        for line in response.iter_lines():
            if line:
                s = line.decode("utf-8", errors="replace")
                if s.startswith("data: "):
                    d = s[6:]
                    if d.strip() == "[DONE]":
                        break
                    try:
                        j = json.loads(d)
                        choices = j.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "") or ""
                            reasoning = delta.get("reasoning_content", "") or ""
                            if content or reasoning:
                                if first_token_time is None:
                                    first_token_time = time.time()
                                    ttft = first_token_time - start_time
                                    log("TTFT: {:.3f}s".format(ttft))
                                full_content += content
                                full_reasoning += reasoning
                    except:
                        pass

        end_time = time.time()
        total_time = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else None
        vram_after = get_vram_usage()
        prefill_throughput = (input_tokens / ttft) if ttft else None

        combined = full_content + " " + full_reasoning
        correct = "42" in combined.strip()

        result = {
            "name": name, "input_tokens": input_tokens, "output_tokens": output_tokens,
            "status": "PASS" if correct else "CHECK",
            "ttft_s": round(ttft, 3) if ttft else None,
            "total_time_s": round(total_time, 3),
            "prefill_throughput_tps": round(prefill_throughput, 1) if prefill_throughput else None,
            "content": full_content[:300],
            "reasoning": full_reasoning[:300],
            "correct": correct,
            "oom": False,
            "vram_before_gb": vram_before.get("total_used_gb"),
            "vram_after_gb": vram_after.get("total_used_gb"),
            "vram_total_gb": vram_after.get("total_total_gb"),
        }

        log("TTFT: {}s".format(round(ttft, 3) if ttft else "N/A"))
        log("Total time: {:.3f}s".format(total_time))
        log("Prefill: {} t/s".format(round(prefill_throughput, 1) if prefill_throughput else "N/A"))
        log("Content: {}".format(full_content[:200]))
        log("Reasoning: {}".format(full_reasoning[:200]))
        log("Correct (needle found): {}".format(correct))
        log("VRAM after: {}GB / {}GB".format(vram_after.get("total_used_gb", "?"), vram_after.get("total_total_gb", "?")))

    except requests.exceptions.Timeout:
        log("ERROR: Request timed out")
        vram_after = get_vram_usage()
        result = {
            "name": name, "input_tokens": input_tokens, "output_tokens": output_tokens,
            "status": "TIMEOUT", "oom": False,
            "vram_before_gb": vram_before.get("total_used_gb"),
            "vram_after_gb": vram_after.get("total_used_gb"),
        }
    except Exception as e:
        err_str = str(e)
        log("ERROR: {}".format(err_str[:300]))
        vram_after = get_vram_usage()
        is_conn_refused = "Connection refused" in err_str or "Connection aborted" in err_str
        result = {
            "name": name, "input_tokens": input_tokens, "output_tokens": output_tokens,
            "status": "CRASHED" if is_conn_refused else "ERROR",
            "error": err_str[:300],
            "oom": "out of memory" in err_str.lower() or "oom" in err_str.lower(),
            "vram_before_gb": vram_before.get("total_used_gb"),
            "vram_after_gb": vram_after.get("total_used_gb"),
        }

    results.append(result)
    save_results(results)


def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump({"status": "in_progress", "results": results}, f, indent=2)


def main():
    open(LOG_FILE, "w").close()
    log("Starting long context stress test for GLM-5.2 on MI308X (v13 no-MTP)")
    log("API: {}, Model: {}".format(API_URL, MODEL))

    if not ensure_server_ready():
        log("FATAL: Server not ready")
        with open(RESULTS_FILE, "w") as f:
            json.dump({"status": "server_not_ready", "results": []}, f, indent=2)
        return

    try:
        r = requests.get(f"{API_URL}/get_server_info",
            headers={"Authorization": f"Bearer {API_KEY}"}, timeout=10)
        if r.status_code == 200:
            info = r.json()
            log("Server: max_total_num_tokens={}, kv_cache_dtype={}".format(
                info.get("max_total_num_tokens"), info.get("kv_cache_dtype")))
    except:
        pass

    log("Sending warmup request...")
    try:
        r = requests.post(f"{API_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": MODEL, "messages": [{"role": "user", "content": "Say hello"}],
                  "max_tokens": 5, "temperature": 0.0}, timeout=60)
        if r.status_code == 200:
            log("Warmup OK")
        else:
            log("Warmup failed: HTTP {}".format(r.status_code))
    except Exception as e:
        log("Warmup error: {}".format(e))

    results = []
    for config in TESTS:
        run_test(config, results)
        time.sleep(5)

    log("\n" + "=" * 60)
    log("FINAL SUMMARY")
    log("=" * 60)
    log("{:>6s} | {:>10s} | {:>8s} | {:>10s} | {:>8s} | {:>5s}".format(
        "Test", "Status", "TTFT", "Prefill", "Correct", "OOM"))
    for r in results:
        name = r.get("name", "?")
        status = r.get("status", "?")
        ttft = "{}s".format(r.get("ttft_s", "?")) if r.get("ttft_s") else "N/A"
        tps = "{} t/s".format(r.get("prefill_throughput_tps", "?")) if r.get("prefill_throughput_tps") else "N/A"
        correct = str(r.get("correct", "?"))
        oom = str(r.get("oom", False))
        log("{:>6s} | {:>10s} | {:>8s} | {:>10s} | {:>8s} | {:>5s}".format(
            name, status, ttft, tps, correct, oom))

    with open(RESULTS_FILE, "w") as f:
        json.dump({"status": "complete", "results": results}, f, indent=2)
    log("\nResults saved to {}".format(RESULTS_FILE))
    log("DONE")


if __name__ == "__main__":
    main()
