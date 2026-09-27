# Findings

## Current DFlash2 state
- Canary decode image: `20260927r`
- Prefill image: `20260927l`
- Root cause of TP7 hang: `window_kv_indptr[-1].item()` forced GPU scalar readback.
- Fix: compute window length on CPU and avoid `.item()`.
- Additional fix: replace host-to-device `torch.tensor(...)` with GPU-side `torch.full` in DSA target-verify metadata.

## Model compatibility
- `glm53-flash`: `Glm5Next`, hidden size 4096.
- `glm53-fp8`: `GlmMoeDsa`, hidden size 6144.
- Current DFlash2 draft model hidden size: 4096.
- DFlash2 borrows the target model's `lm_head`, so draft hidden size must match target hidden size.
