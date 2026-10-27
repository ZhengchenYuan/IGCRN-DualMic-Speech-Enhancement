# V1 completed experiment

The summary, per-utterance metrics, provenance and listening candidates are unmodified exports from the completed Colab evaluation. The original test manifest is committed separately at [evaluation_manifest.jsonl](../../manifests/evaluation_manifest.jsonl). `provenance.json` binds the evaluation to both final checkpoints and this manifest. Checkpoint metadata and hashes were also verified locally against the supplied binaries; no training or metric reevaluation was performed during delivery preparation.

| SNR (dB) | Method | Cases | SI-SDR (dB) | SI-SDRi (dB) | STOI |
|---|---|---:|---:|---:|---:|
| all | mic1 | 120 | 0.0136 | 0.0000 | 0.7633 |
| all | dual | 120 | 4.6104 | 4.5968 | 0.7895 |
| all | duplicate | 120 | 5.1746 | 5.1610 | 0.7805 |
| -3 | mic1 | 40 | -2.9900 | 0.0000 | 0.6996 |
| -3 | dual | 40 | 1.6333 | 4.6233 | 0.7301 |
| -3 | duplicate | 40 | 2.2398 | 5.2298 | 0.7133 |
| 0 | mic1 | 40 | 0.0249 | 0.0000 | 0.7744 |
| 0 | dual | 40 | 4.5688 | 4.5439 | 0.7973 |
| 0 | duplicate | 40 | 5.1140 | 5.0891 | 0.7927 |
| 3 | mic1 | 40 | 3.0059 | 0.0000 | 0.8159 |
| 3 | dual | 40 | 7.6290 | 4.6231 | 0.8412 |
| 3 | duplicate | 40 | 8.1700 | 5.1642 | 0.8354 |

Duplicate exceeds Dual by 0.564244 dB SI-SDRi; Dual exceeds Duplicate by 0.009047 STOI. Both differences have the same direction in every SNR group. These are descriptive means from one training seed; no paired confidence intervals or significance claims are available.

## Success/failure candidates, not listening findings

Metric-ranked Dual candidates are `test_00067` (+17.0159 dB) and `test_00105` (+15.1013 dB); negative candidates are `test_00003` (−7.3731 dB) and `test_00096` (−8.2304 dB). These are SI-SDRi values, not perceptual judgments. Listening analysis remains outstanding. The four committed export groups are `test_00000`–`test_00003`, so only one of these ranked candidates is bundled. Full ranked candidate details are in `listening_candidates.json`.

## Scope and provenance

Runtime reported by the experiment owner: approximately 55 minutes per training arm and 5.6 minutes for evaluation on Colab T4. Original environment snapshot and train/valid manifests are not included in the supplied assets; the current dependency pins describe the supported setup, not a recovered historical `pip freeze`. Source catalog entries and licenses for each test case are retained in the manifest. No original V1 data, checkpoint, or metric export was rewritten.

The original `dual_history.jsonl` and `duplicate_history.jsonl` are also included as supplied training logs.
