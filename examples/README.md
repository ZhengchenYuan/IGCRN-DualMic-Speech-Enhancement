# V1 Audio Examples

Four original held-out test cases are available for listening. Each provides unprocessed Mic1, the noise-free reverberant Mic1 target, and outputs from the independently trained Dual and Duplicate models. Open a WAV link to play or download the file.

## Available audio examples

| Test ID | Dual outcome | SNR (dB) | Mic1 | Target | Dual | Duplicate | Dual SI-SDRi (dB) | Duplicate SI-SDRi (dB) |
|---|---|---:|---|---|---|---|---:|---:|
| test_00000 | Positive | -3 | [WAV](test_00000/mic1.wav) | [WAV](test_00000/target.wav) | [WAV](test_00000/dual.wav) | [WAV](test_00000/duplicate.wav) | +3.3162 | +1.9778 |
| test_00001 | Positive | 0 | [WAV](test_00001/mic1.wav) | [WAV](test_00001/target.wav) | [WAV](test_00001/dual.wav) | [WAV](test_00001/duplicate.wav) | +7.2325 | +5.5319 |
| test_00002 | Negative | 3 | [WAV](test_00002/mic1.wav) | [WAV](test_00002/target.wav) | [WAV](test_00002/dual.wav) | [WAV](test_00002/duplicate.wav) | -0.4802 | +4.5110 |
| test_00003 | Negative | -3 | [WAV](test_00003/mic1.wav) | [WAV](test_00003/target.wav) | [WAV](test_00003/dual.wav) | [WAV](test_00003/duplicate.wav) | -7.3731 | -1.0327 |

Metrics come from the [complete 120-case evaluation CSV](../results/v1/per_utterance.csv). Positive / Negative denotes only the sign of Dual SI-SDRi relative to unprocessed Mic1; it is not a perceptual judgment or a claim that Dual outperforms Duplicate. These four cases are the original evaluator's first four manifest entries, exported using each entry's real test ID. They illustrate both metric improvements and degradations without selecting audio by a favorable result.

All 16 WAVs are mono, IEEE float32, 16 kHz, 32,000 samples (2 seconds). Original file contents and SHA256 are preserved. Decoded samples reproduce their corresponding CSV SI-SDR, SI-SDRi and STOI values within 1e-6 in the local verification. [Source attribution and file hashes](v1/manifest.json) and the [artifact hash inventory](../results/v1/artifact_sha256.json) identify every recording.

## Quantitative case analysis

The [separate quantitative analysis](../results/v1/CASE_ANALYSIS.md) reports the highest and lowest Dual SI-SDRi cases from the full test CSV. Among those four ranked cases, only `test_00003` has a bundled audio example. The other three are quantitative-only observations, with no listening or sound-quality claims.

## Interpretation and source attribution

The target includes Mic1's simulated room response; dereverberation is not required. Audio inspection should compare speech preservation and residual noise against this reverberant reference. No case-specific perceptual diagnosis is asserted here.

Speech derives from Mini LibriSpeech (CC BY 4.0), and environmental noise from ESC-10 (CC BY 3.0). Original per-recording attribution and license URLs are retained in the [export metadata](v1/manifest.json) and [evaluation manifest](../manifests/evaluation_manifest.jsonl). The repository's MIT license does not replace those audio licenses.

These listening exports contain only Mic1, not the original synchronized Mic2 input, so they do not alone reproduce Dual inference. Numbered WAVs and `manifest.json` at this directory's root are earlier synthetic chirp diagnostics, separate from the real V1 speech examples above.
