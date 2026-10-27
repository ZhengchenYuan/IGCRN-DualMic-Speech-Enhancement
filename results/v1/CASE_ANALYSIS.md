# Quantitative Case Analysis

This analysis uses the unchanged [per-utterance CSV](per_utterance.csv) from the complete **120-case V1 test set**, evaluated with both independently trained final checkpoints. The two highest and two lowest Dual SI-SDRi cases are recorded in the original [listening candidate selection](listening_candidates.json). The [evaluation provenance](provenance.json) binds those results to the original checkpoint and manifest hashes.

## Metric-ranked positive and negative cases

| Test ID | Dual outcome | SNR (dB) | Dual SI-SDRi (dB) | Duplicate SI-SDRi (dB) | Dual − Duplicate (dB) | Evidence |
|---|---|---:|---:|---:|---:|---|
| test_00067 | Positive | 0 | +17.0159 | +17.2312 | -0.2154 | Quantitative only |
| test_00105 | Positive | -3 | +15.1013 | +16.6643 | -1.5630 | Quantitative only |
| test_00003 | Negative | -3 | -7.3731 | -1.0327 | -6.3404 | [Audio and metrics](../../examples/README.md) |
| test_00096 | Negative | -3 | -8.2304 | -0.8716 | -7.3588 | Quantitative only |

SI-SDRi is each method's SI-SDR minus unprocessed Mic1 SI-SDR for the same case and target. Positive and negative outcomes refer only to that improvement. Values are rounded to four decimal places for display; the CSV retains full precision. The comparison column is computed before rounding.

All four cases have higher Duplicate SI-SDRi than Dual SI-SDRi, including the two cases with large Dual gains. The positive cases therefore demonstrate improvement over Mic1, not superiority over the trained Duplicate control. Both negative cases show that average improvements do not ensure improvement on every utterance.

## Audio evidence and interpretation

Only `test_00003` among these four ranked cases has bundled audio. `test_00067`, `test_00105` and `test_00096` are **quantitative-only cases**. Their values remain valid entries in the completed evaluation record; no waveform reconstruction or case-specific listening claim is made.

The [available audio examples](../../examples/README.md) are instead the original exports for `test_00000`, `test_00001`, `test_00002` and `test_00003`, with four WAVs per case. Quantitative ranking and audio availability are separate selection criteria.

These descriptive examples do not establish a failure mechanism, statistical significance, or a perceptual quality ranking. One seed, a small synthetic dataset, two-second crops and limited first-order room simulation constrain interpretation. Paired confidence intervals and a documented human listening analysis would strengthen subsequent analysis.
