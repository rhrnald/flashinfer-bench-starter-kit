# Direct B200 Small-Seq Optimization Plan

## Scope
- Primary target: small sequence workloads first, especially workload 0~5.
- Large sequence workloads are not the current priority.
- Submission fast path may hard-code static choices and remove debug/fallback branches.
- Failed quantization fusion experiments should not become default unless a later variant proves faster.

## Current Baseline
- Current committed fast path: direct B200 Step1/Step2 with Step2 W2 sw128 TMA, K8 double buffering, batch2 MMA wait reduction, and small-T metadata sync reduction.
- Recent 0~19 timing:
  - ours sum: `105.896 ms`
  - aggregate speedup: `0.0866x`
- Phase 1+3 working tree timing:
  - 0~5 sum: `7.610 ms` versus `7.691 ms` baseline log
  - 0~19 sum: `105.573 ms` versus `105.896 ms` baseline log
  - correctness: 0~19 PASS at default ratio
- Phase 2 CUDA Graph result:
  - implemented as opt-in `FIB_ENABLE_DIRECT_GRAPH=1` because the TVM stream cannot be captured directly
  - uses a dedicated nonblocking graph stream with event dependencies to/from the TVM stream
  - 0~5 graph sum: `7.653 ms`; graph disabled sum: `7.605 ms`
  - conclusion: keep CUDA Graph available for experiments, but do not enable by default
- Phase 5 memset reduction:
  - `local_weight` zeroing moved into `routing_kernel`, removing one separate memset launch
  - `expert_counts` and `running_counter` init combined into one tiny kernel
  - 0~5 sum: `7.583 ms`; 0~19 sum: `105.288 ms`
  - correctness: 0~19 PASS at default ratio
- Recent workload 0~5 profile, median-ish:

| idx | seq | routing | step1 | step2 | expert_other | out | total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7 | 0.090 | 0.104 | 0.098 | 0.053 | 0.010 | 0.356 |
| 1 | 1 | 0.066 | 0.059 | 0.060 | 0.052 | 0.009 | 0.251 |
| 2 | 32 | 0.109 | 0.260 | 0.279 | 0.062 | 0.010 | 0.725 |
| 3 | 80 | 0.138 | 0.344 | 0.426 | 0.072 | 0.011 | 1.005 |
| 4 | 901 | 0.158 | 2.113 | 2.530 | 0.103 | 0.025 | 4.934 |
| 5 | 16 | 0.097 | 0.166 | 0.183 | 0.063 | 0.010 | 0.522 |

All times are ms. `step2` includes Step2 prequant.

## Current Pipeline
1. Workspace growth checks and buffer memset.
2. `routing_kernel`.
3. Grouped metadata:
   - expert counts are accumulated inside `routing_kernel`
   - `scan_offsets_kernel`
   - `scatter_placements_kernel`
4. Step1 direct:
   - GEMM1
   - SwiGLU
   - write `c_perm_all`
5. Step2 prequant:
   - `c_perm_all -> c_perm_q + c_perm_scale`
6. Step2 direct:
   - W2 TMA
   - tcgen MMA
   - TF32 scale MMA
   - scatter to `out_acc`
7. Output convert:
   - `out_acc(float) -> output(bf16)`
8. Final sync.

## Important Negative Results
- Step2-side quantization fusion was slower.
  - Cause: Step2 CTAs are launched per H tile, so the same `c_perm_all` quantization repeats for every H tile.
- Step1 producer-side quantization fusion was also slower in the current implementation.
  - Cause: Step1 critical path gained shared staging, reductions, and synchronization.
  - Removing the FP32 `c_perm_all` global store was not enough to beat standalone prequant.
- Therefore default should keep the standalone Step2 prequant kernel until a better producer design exists.

## Optimization Candidates

### 1. Clean Up Failed Fusion Experiments
- Remove or fully isolate `FIB_STEP1_FUSE_PREQUANT` and `FIB_STEP2_FUSE_PREQUANT`.
- Keep the default path clean before adding more experiments.
- Expected effect: no direct speedup, but reduces compile/control complexity and prevents accidental slow-path testing.

### 2. Cache `cudaFuncSetAttribute`
- Step1 direct currently calls `cudaFuncSetAttribute` on every launch.
- Step1 comm-only also has repeated attribute calls.
- Add static one-time guards per kernel/smem configuration.
- Expected effect: small but safe improvement for tiny seq.
- Risk: low.

### 3. Submit Fast-Path Config Cache
- Avoid repeated `std::getenv` parsing in the hot function.
- Parse once into static config:
  - direct implementation enabled
  - TMA enabled
  - sw128 enabled
  - B sw128 enabled
  - tcgen accum mode
  - Step2 variant
- For final submission, hard-code the chosen path if acceptable.
- Expected effect: small but safe for small seq.
- Risk: low.

### 4. CUDA Graph for Small Seq
- Capture the stable small-seq direct pipeline and replay it.
- Graph should include:
  - memset/init kernels
  - routing
  - metadata kernels
  - Step1
  - Step2 prequant
  - Step2
  - output convert
- Cache graph by:
  - `t`
  - relevant tensor pointers
  - workspace buffer pointers/capacity
  - selected static config
- Use graph only for small seq at first, e.g. `t <= 128`.
- Expected effect: likely largest immediate win because workload 0~5 is launch-heavy.
- Risks:
  - Graph capture may fail if host-side CUDA calls remain inside capture.
  - Workspace growth must happen before capture.
  - TMA descriptor upload must be outside capture.
  - Host metadata sync path cannot be inside graph; use device metadata/all-expert path for graph.
- Result:
  - Direct capture on the TVM stream fails even with an empty graph.
  - Side-stream graph capture works and passes correctness, but event overhead plus custom zero kernels made it slightly slower on 0~5.
  - Default remains non-graph; enable with `FIB_ENABLE_DIRECT_GRAPH=1` only for follow-up experiments.

### 5. Routing + Count Fusion
- Current routing writes dense `local_weight[t,32]`, then `build_counts_kernel` rereads it.
- Fuse counts into routing:
  - zero `expert_counts`
  - routing computes local routed experts
  - routing writes `local_weight`
  - routing also `atomicAdd(expert_counts[le])`
  - remove `build_counts_kernel`
- Expected effect: removes one kernel launch and one dense read.
- Risk: medium-low. Atomic pattern is similar to current build-count atomics.

### 6. Routing + Metadata Redesign
- More aggressive than count fusion.
- Avoid dense `local_weight[t,32]`.
- Route directly into compact per-token local topk arrays:
  - token local expert ids
  - token weights
  - local route count
- Then build expert grouped layout from compact arrays.
- Expected effect: reduce memory traffic and memset pressure.
- Risk: medium. Needs careful correctness comparison.

### 7. Memset Reduction
- Current per-call memsets:
  - `local_weight_dev`
  - `out_acc_dev`
  - `expert_counts_dev`
  - `running_counter_dev`
- Opportunities:
  - remove `local_weight` memset after routing metadata redesign
  - combine `expert_counts` and `running_counter` init in one tiny init kernel
  - remove `out_acc` memset only if Step2 writes final output without atomic partial accumulation
- Expected effect: meaningful for tiny seq because each memset is a launch.
- Risk: varies.
- Result:
  - Removed the standalone `local_weight` memset by clearing each token row inside routing.
  - Combined the two 32-int metadata memsets into `clear_expert_metadata_kernel`.
  - Kept `out_acc` memset because Step2 still uses atomic accumulation.

### 8. Small-Seq Static Direct Path
- Create a dedicated fast function for `t <= 128`:
  - no active expert host list
  - all-expert launch
  - no fallback branches
  - fixed Step2 K8 sw128 path
  - fixed `tcgen_accum=tf32_scale`
  - no debug paths
  - no env parsing
- Expected effect: reduces host and device branch overhead.
- Risk: low if gated by env or `t <= 128`.

### 9. Step2 Scatter/Accumulator Refinement
- Current Step2 scatters partial K-group results into `out_acc` with `atomicAdd`.
- For K8 there are two K groups, so up to two output atomics per value.
- Candidate:
  - accumulate both K groups in TMEM or shared/register, scatter once
  - or use a per-CTA output staging then write once
- Expected effect: medium for Step2-heavy small/medium seq.
- Risk: high due TMEM/shared pressure and correctness.

### 9a. Step2 Larger N / Weight Reuse Experiment
- Goal: process more routed rows per loaded W2 group for large `t_valid`.
- Buffer math for M64/K8:
  - W2 double buffer: `2 * 64 * 128 * 8 = 131072 B`
  - fallback A buffer if kept: `8192 B`
  - B buffer for N: `N * 128 B` for batch1, `2 * N * 128 B` for batch2
  - TF32 diag for N: `4 issues * (N / 4) * N * 4 floats`
  - Shared memory is not the hard limit; B200 SM shared is about `233472 B`.
- TMEM limit:
  - batch2 needs roughly `8N partial + N scaled = 9N` cols, so 128-col TMEM only safely supports N8.
  - batch1 needs roughly `4N partial + N scaled = 5N` cols, so 128-col TMEM can support N16 and N24.
- Implemented experimental selector:
  - `FIB_STEP2_PIPELINE_VARIANT=m64n16_k8_batch1`
  - M64, N16, K8, sw128 W2 TMA, batch1 MMA waits.
- Current status:
  - correctness is fixed for the experimental N16 path.
  - debug toy: `/home/snu_avq1/workspace/chaewon/test/tcgen05_m64n16_mapping_toy.cu`.
  - confirmed FP8 N16 readback uses TMEM column stride 8.
  - confirmed a single N16 TF32 diagonal scale MMA does not match the expected mapping.
  - working mapping is N16 FP8 followed by two N8 TF32 diagonal scale MMAs.
  - fixed one confirmed N16 shared B swizzle bug: use `n & 7` in the swizzle xor for rows 8~15.
- Performance result:
  - `m64n16_k8_batch1` passes workload 0~5 and 8~9, but is slower overall.
  - workload 4 is roughly neutral/slightly better in one short run (`4.810 ms` smoke, `4.869 ms` iter10).
  - workload 8/9 regress badly (`71.953/48.005 ms` vs default about `54.553/35.793 ms`).
- Default remains N8 batch2. N16 is useful as a debug/possible moderate-`t_valid`
  building block, not as the global Step2 path.

### 10. Output Convert Fusion
- Current output convert is only about `0.01 ms`, but it is one launch.
- Candidate:
  - fold output conversion into Step2 final scatter if Step2 can produce final bf16 once
- Expected effect: small after CUDA Graph, more useful without graph.
- Risk: medium because `out_acc` currently accumulates across atomics.

### 11. Step1 Specialization
- Remove runtime/debug branches in Step1 fast kernel.
- Build a submit-only kernel template:
  - no debug output modes
  - fixed TMA sw128
  - fixed B sw128
  - fixed tcgen accumulation mode
- Expected effect: possible instruction/register pressure reduction.
- Risk: medium. Must compare SASS/registers/perf.

### 12. Step2 Specialization
- Remove fallback/repack path from submit-only Step2 kernel.
- Fixed K8, fixed sw128 TMA, fixed batch2.
- Remove generic selector and null TMA branches from kernel.
- Expected effect: small-to-medium, especially for compiler optimization.
- Risk: medium-low if old generic kernel is retained as fallback.

## Execution Plan

### Phase 0: Reset/Isolate Experiments
- [x] Decide whether to revert failed fusion code or keep it behind experimental envs.
- [x] Confirm default path matches last pushed behavior.
- [x] Run:
  - `WORKLOAD_INDEXES=0,1,2,3,4,5 WARMUP_RUNS=1 ITERATIONS=10 NUM_TRIALS=1 ./run.sh`

### Phase 1: Low-Risk Host Cleanup
- [x] Cache Step1 `cudaFuncSetAttribute`.
- [x] Cache Step1 comm-only `cudaFuncSetAttribute`.
- [x] Cache or hard-code submit env parsing.
- [x] Measure 0~5 and 0~19.

### Phase 2: CUDA Graph Small Path
- [x] Ensure workspace growth and TMA descriptor uploads happen before capture.
- [x] Use device metadata/all-expert path only.
- [x] Capture first call for `t <= 128`.
- [x] Replay subsequent calls with the same pointer/capacity signature.
- [x] Measure 0~5, with special attention to workloads 0, 1, 2, 3, 5.
- [x] Keep graph opt-in because measured 0~5 was slightly slower than the default path.

### Phase 3: Routing Count Fusion
- [x] Add routing variant that also updates `expert_counts`.
- [x] Remove `build_counts_kernel` in that variant.
- [x] Compare correctness and timing.
- [ ] Keep fallback routing path until stable.

### Phase 4: Small-Seq Static Kernels
- [ ] Add submit-only Step1 fast launch.
- [ ] Add submit-only Step2 fast launch.
- [ ] Remove kernel-internal fallback branches from these specialized kernels.
- [ ] Compare against generic current default.

### Phase 5: More Aggressive Device Pipeline
- [ ] Redesign routing metadata to avoid dense `local_weight`.
- [x] Reduce memsets.
- [ ] Explore Step2 one-scatter accumulation.
- [ ] Explore output convert fusion.

## Test Matrix

### Primary Performance
```bash
WORKLOAD_INDEXES=0,1,2,3,4,5 WARMUP_RUNS=1 ITERATIONS=10 NUM_TRIALS=1 ./run.sh
```

### Full Sanity
```bash
WORKLOAD_INDEXES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
WARMUP_RUNS=1 ITERATIONS=10 NUM_TRIALS=1 ./run.sh
```

### Profile
```bash
FIB_MOE_PROFILE=1 WORKLOAD_INDEXES=0,1,2,3,4,5 \
WARMUP_RUNS=1 ITERATIONS=3 NUM_TRIALS=1 ./run.sh
```

### Correctness Smoke
```bash
WORKLOAD_INDEXES=0,1,2,3,4,5 WARMUP_RUNS=1 ITERATIONS=5 NUM_TRIALS=1 ./run.sh
```

### Strict Ratio Record
```bash
WORKLOAD_INDEXES=0,1,2,3,4,5 EXTRA_ARGS="--required-matched-ratio 0.9" ./run.sh
```

## Selection Rules
- First priority: workload 0~5 PASS at default 0.8 matched-ratio.
- Second priority: workload 0~5 latency sum.
- Third priority: no large regression on 0~19 unless explicitly accepted.
- Keep failed variants behind env only if they help future debugging; otherwise remove them.
