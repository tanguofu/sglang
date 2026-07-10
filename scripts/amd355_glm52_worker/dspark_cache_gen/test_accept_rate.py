import requests, sys, json

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:30000"
num_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 20
api_key = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"

prompts = [
    "Write a Python function to reverse a string.",
    "Explain how quicksort works.",
    "What is the time complexity of binary search?",
    "Write a SQL query to find the second highest salary.",
    "Implement a binary tree in Python.",
    "Explain the difference between TCP and UDP.",
    "Write a function to check if a string is a palindrome.",
    "How does garbage collection work in Python?",
    "Write a regex to match email addresses.",
    "Explain the CAP theorem.",
] * (num_requests // 10 + 1)

print("Testing accept_rate against %s (%d requests)" % (url, num_requests))
print()

accept_rates = []
for i in range(num_requests):
    try:
        resp = requests.post(url + "/generate", json={
            "text": prompts[i],
            "sampling_params": {"max_new_tokens": 128, "temperature": 0},
            "stream": True,
        }, headers={"Authorization": "Bearer " + api_key}, timeout=120, stream=True)

        for line in resp.iter_lines():
            if line:
                text = line.decode("utf-8")
                if text.startswith("data: ") and text != "data: [DONE]":
                    try:
                        data = json.loads(text[6:])
                        meta = data.get("meta_info", {})
                        if "spec_accept_rate" in meta:
                            rate = meta["spec_accept_rate"]
                            accept_len = meta.get("spec_accept_length", 0)
                            accept_rates.append(rate)
                            print("  req %d: accept_len=%.2f rate=%.2f" % (i, accept_len, rate))
                            break
                    except:
                        pass
    except Exception as e:
        print("  req %d: ERROR - %s" % (i, e))

if accept_rates:
    avg = sum(accept_rates) / len(accept_rates)
    print("\n=== RESULTS ===")
    print("Average accept_rate: %.2f (over %d requests)" % (avg, len(accept_rates)))
    print("Min: %.2f  Max: %.2f" % (min(accept_rates), max(accept_rates)))
else:
    print("\nNo accept_rate data found. Check server configuration.")
