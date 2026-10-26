# V2 stage-one implementation

V1 inputs are read-only. The IGCRN model, loss, legacy simulator, legacy training and evaluator are unchanged. V2 is a separately configured spatial-simulation experiment, not a claim that Dual must outperform Duplicate.

## Approved parameters

`configs/v2.yaml` fixes 16 kHz, 2 s, 5×5×3 m, microphones (2.49,2.5,1.5)/(2.51,2.5,1.5), radius/height 1.5 m, independent continuous azimuths −90°..90°, RT60 requests 0.15/0.25/0.35 s, and SNR −3/0/3 dB. `pra.inverse_sabine` determines uniform wall energy absorption and ISM order. No ray tracing, randomized ISM, air absorption, or artificial AWGN. Actual measured RT60 is recorded separately from the request; no silent order cap. The fixed protocol is 3000/300/300 samples and 300/60/60 geometry-cache entries. Requested RT60 cycles across the geometry bank, but a separate seeded permutation assigns geometry to sample index so RT60 is not coupled to the repeating SNR schedule. Source selection, geometry creation and geometry assignment use separate random streams. SNR is exactly balanced by sample index. Data are generated offline on CPU.

Every geometry has two sources and two microphones. PRA RIRs are zero-padded to a common length without individually aligning their peaks. A full RIR of source history precedes the shared two-second crop; the clean target is the same crop of reverberant speech at Mic1. SNR uses Mic1 reverberant energy, one noise scaling factor for both channels and one common output gain. Short sources are rejected rather than tiled. Each sample allows 20 attempts. Split/sample/retry RNG streams are independent, and all source IDs, offsets, geometry, RIR hashes, crop and gains are recorded. WAVs use timestamp-free IEEE float32 writing. A staging-directory rename publishes only a complete dataset.

## Source preparation and V1 protection

Restore all three original V1 manifests and legally obtained raw sources. A test-only manifest cannot establish V1 train/valid protection. Do not rewrite original manifests to relocate data; restore original `/content/...` paths when necessary.

Candidate sources can be prepared into a NEW directory with the existing `scripts.prepare_data` pipeline, e.g. 24/6/6 candidate speakers, up to 40 utterances each, and 80/20/20 candidate noise originals, subject to the actual public-subset inventory. Its candidate split labels are **not authoritative** for V2: `register_v2_split` locks every original V1 speaker/noise ID to its role, then deterministically assigns unused IDs. It also checks source hashes for cross-split collisions and binds the original train/valid manifest hashes to the trusted V1 Dual checkpoint. A registry without this binding is fixture-only and cannot generate a production dataset. No source inventory count or caller-supplied metadata is treated as proof of experimental performance.

From repository root:
```bash
python -m pip install -r requirements-v2.txt
python -m scripts.register_v2_split --v1-manifests /content/v1/train.jsonl /content/v1/valid.jsonl /content/v1/test.jsonl --catalog /content/igcrn_candidates/catalog.jsonl --v1-dual /content/drive/MyDrive/IGCRN_NIO/dual/last.pt --output /content/igcrn_v2_registry.json
python -m scripts.accept_acoustics --output /content/igcrn_v2_stage1/acoustics
python -m scripts.diagnose_spatial --manifest /content/v1/valid.jsonl --dual /content/drive/MyDrive/IGCRN_NIO/dual/last.pt --output /content/igcrn_v2_stage1/v1
python -m scripts.simulate_v2 --registry /content/igcrn_v2_registry.json --output /content/igcrn_v2_stage1/small_data --cache /content/igcrn_v2_rir_cache --acceptance-count 3
python -m pytest -q
```
All output directories above must be new. The acceptance dataset is explicitly labeled `acceptance_only` and cannot be used for production training. Production counts cannot be changed through that mode. Check raw-source duration: pre-roll adds required context beyond two seconds.

## Diagnostics

`diagnose_spatial` verifies frozen input hashes and loads only a trusted Dual checkpoint. It computes inter-channel correlation/difference energy, sub-sample GCC-PHAT (32× interpolation, ±2-sample search), weighted IPD/ILD in 200–1000, 1000–4000 and 4000–7000 Hz, geometry-predicted speech/noise delays and their separation. Low-energy STFT bins are excluded. Mixture GCC-PHAT is not isolated-source ground truth. High correlation does not mean no usable spatial cue.

Frozen inference compares normal Mic1+Mic2, duplicated Mic1 and zero Mic2, reporting output differences, SI-SDRi and standard STOI. Perturbation sensitivity does not prove correct physical cue use and is not the trained Duplicate ablation. Missing dual-channel audio is UNVERIFIED; mono exports are never substituted. Prefer V1 validation for diagnostics. Any supplied test-set diagnostics are retrospective and must not tune V2 test cases.

`accept_acoustics` compares legacy, PRA direct and PRA first-order propagation. Strict TDOA/IPD tolerances apply only to free-field PRA, not to full multipath transfer functions. Three high-order probes record requested/actual RT60, actual ISM order, global fractional-filter delay and CPU time. Missing PRA produces UNVERIFIED, not PASS.

## Paired training and 300-minute budget

Use `notebooks/v2_colab.ipynb`. It defaults to `RUN_OVERNIGHT=False`; stage one does not launch production generation or training. The production CLI also defaults to PLAN_ONLY without `--execute`. First restore the actual V1 paths, supply a verified candidate catalog, and pass the physical acceptance plus actual dual-channel V1 diagnostic gates.

Production run directories must be under a distinct `.../v2/<run_id>`, never V1's `dual` or `duplicate`. Both models/optimizers remain resident; each arm has its own model, optimizer, loader generator and RNG snapshot. Same seed gives matched initial weights and data ordering; no trained weights cross arms. Order is Dual epoch, Duplicate epoch, then publication of a checkpoint pair.

Only whole epochs are resumable. A failed/partial epoch or unpaired epoch is not a final comparison; resume rolls back to the last published pair. Extra discarded work is recorded in attempted steps where possible, never hidden as equal actual compute. Checkpoints include optimizer/RNG/loader state and effective step count. Temporary saves are copied to Drive and SHA256-rechecked; `pair.json` is published last. Older immutable pairs remain available. History is recoverable from each committed epoch log. Each minute, runtime state reports phase, active epoch, observed steps/loss and remaining time. Full completion is never inferred from a lone `last.pt`.

The clock is persisted before notebook dependency installation when overnight mode is enabled. Disconnect/restart never resets the absolute deadline. Before each pair, the controller reserves evaluation (20 min), sync (7 min), safety (15 min), plus an initial 20-minute pair estimate or a conservative recent measured pair duration (including I/O). The first actual pair is also the throughput calibration; it is not thrown away. Maximum 11 paired epochs; no test-score-based stopping, dataset shrinking, or favorable metric selection. A completed lower common budget is labeled `COMPLETE_REDUCED_BUDGET` with actual epochs/updates.

At update boundaries the trainer checks its deadline. An already-started backward/clip/optimizer step finishes, CUDA is synchronized, then stopping occurs before another update. This is **not batch-level resume**; incomplete epoch progress is discarded. A partially completed pair gets `INTERRUPTED_UNCOMMITTED_PAIR`, no experiment-complete marker. Sudden platform termination or a hung external I/O call cannot be made into a guaranteed graceful stop by Python; no SIGKILL or keep-alive scripts are used.

Evaluation uses the same CPU metric definitions as V1, all 300 cases / 900 method rows, same paired checkpoints, macro/per-SNR means, seed-fixed SNR-stratified paired bootstrap intervals and four input/target/output cases. CIs are utterance-level, not multi-seed or speaker/noise-cluster claims. Test-time predictions must be finite. Incomplete evaluation writes partial metadata but no complete marker. Completion is published only after full results and checkpoint verification; listening candidates are explicitly unlistened.

## Expected cost and interpretation

Using the reported V1 55 minutes per 30 epochs, a V2 epoch is roughly 9 minutes per arm, not a guarantee. Target 11 pairs takes about 198 minutes. A planning envelope is diagnostics 15, environment/data 40, initial calibration overhead 5, training 198, evaluation 20, final synchronization 7 and safety 15 = 300 minutes. Preparation or slower pairs reduce the common budget. CPU ISM cost must first be measured; no artificial RT60/order downgrade is used to fit time. Data-generation preparation does not use GPU, though Colab may count the allocated GPU session.

Keep the 6-epoch pair (18,000 updates/arm) as an equal-update reference to V1 and the final pair (at most 33,000 updates/arm). More examples plus improved simulation are a combined recipe: V2 scores alone cannot establish which individual change caused improvement. No pass condition requires Dual to win. V1 assets, original manifest bytes and original checkpoint/evaluation hashes remain authoritative. The supplied V1 assets do not contain a verifiable training code commit or the three complete training/validation/source manifests; do not invent that linkage.
