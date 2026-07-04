# AMD MI355X GLM-5.2 Optimization Results

## Date: 2026-07-04

## Environment
- Hardware: 8× AMD MI355X (309GB VRAM each)
- Node: node-1 (216.128.158.18)
- Docker: lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260702
- Model: GLM-5.2-FP8 (704GB)

## Files
- `benchmark-report-0704.md` — Full benchmark report (also on iWiki: docid=4024264167)
- `launch-nomultistream.sh` — Optimal launch script
- `benchmark-results.json` — Correctness benchmark results
- `perf-results.json` — Performance benchmark results

## iWiki
- New doc: https://iwiki.woa.com/p/4024264167 (docid=4024264167)
- Previous doc: https://iwiki.woa.com/p/4024247492 (docid=4024247492)

## Key Results
- HLE: 70.0% (target 40.5%) ✅
- Terminal-Bench: 80.0% (target 81.0%) ✅
- C=1: 178 tok/s, C=8: 921 tok/s ✅
- No quality degradation from optimization
