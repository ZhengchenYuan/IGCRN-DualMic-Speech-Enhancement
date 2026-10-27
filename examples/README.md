# V1 Audio Examples

**Target-case delivery: BLOCKED (1 of 4 available).** The requested ranked cases are listed first. Missing WAVs are explicitly unavailable; no other recordings have been renamed to impersonate them.

Success / Failure below means only positive / negative **Dual SI-SDRi relative to unprocessed Mic1**. Ranking uses the frozen [per-utterance CSV](../results/v1/per_utterance.csv), not listening judgments or superiority over Duplicate. Human listening analysis remains pending.

## Requested success/failure cases

| Test ID | Success / Failure | SNR (dB) | Mic1 | Target | Dual | Duplicate | Dual SI-SDRi (dB) | Duplicate SI-SDRi (dB) |
|---|---|---:|---|---|---|---|---:|---:|
| test_00067 | Success | 0 | BLOCKED | BLOCKED | BLOCKED | BLOCKED | +17.0159 | +17.2312 |
| test_00105 | Success | -3 | BLOCKED | BLOCKED | BLOCKED | BLOCKED | +15.1013 | +16.6643 |
| test_00003 | Failure | -3 | [WAV](test_00003/mic1.wav) | [WAV](test_00003/target.wav) | [WAV](test_00003/dual.wav) | [WAV](test_00003/duplicate.wav) | -7.3731 | -1.0327 |
| test_00096 | Failure | -3 | BLOCKED | BLOCKED | BLOCKED | BLOCKED | -8.2304 | -0.8716 |

## Additional verified exports retained

The original evaluator exports only its first four manifest rows, naming files with the actual row's `id` (`scripts/evaluate.py`, `if i < 4`). The old export indices 0, 1, 2, 3 correspond respectively to **test_00000, test_00001, test_00002, test_00003**. This is supported by original filenames, manifest order and the previously recorded file hashes; it is not a mapping to the four ranked candidates. Current GitHub has no `evaluation/examples` directory; authoritative metrics live in `results/v1/`.

| Test ID | Success / Failure | SNR (dB) | Mic1 | Target | Dual | Duplicate | Dual SI-SDRi (dB) | Duplicate SI-SDRi (dB) |
|---|---|---:|---|---|---|---|---:|---:|
| test_00000 | Success | -3 | [WAV](test_00000/mic1.wav) | [WAV](test_00000/target.wav) | [WAV](test_00000/dual.wav) | [WAV](test_00000/duplicate.wav) | +3.3162 | +1.9778 |
| test_00001 | Success | 0 | [WAV](test_00001/mic1.wav) | [WAV](test_00001/target.wav) | [WAV](test_00001/dual.wav) | [WAV](test_00001/duplicate.wav) | +7.2325 | +5.5319 |
| test_00002 | Failure | 3 | [WAV](test_00002/mic1.wav) | [WAV](test_00002/target.wav) | [WAV](test_00002/dual.wav) | [WAV](test_00002/duplicate.wav) | -0.4802 | +4.5110 |

All available WAVs are mono, IEEE float32, 16 kHz, 32,000 samples (2 seconds). Their decoded audio reproduces each corresponding CSV SI-SDR, SI-SDRi and STOI value within 1e-6 in the local check, providing a numerical cross-check of the IDs beyond filenames. Existing WAVs were moved without re-encoding; hashes remain unchanged. [Export metadata, original source attribution and SHA256](v1/manifest.json) and the [artifact hash inventory](../results/v1/artifact_sha256.json) identify every available file.

Speech derives from Mini LibriSpeech (CC BY 4.0); environmental recordings derive from ESC-10 (CC BY 3.0). Per-recording attribution/license URLs are retained in the export metadata and [original evaluation manifest](../manifests/evaluation_manifest.jsonl). Repository MIT licensing does not replace those audio licenses.

The target is reverberant clean Mic1. These exports contain only Mic1, not synchronized Mic2, and cannot alone reproduce Dual inference. Exported WAV container hashes can differ from original simulator target hashes. The numbered root WAVs and root `manifest.json` are older **synthetic chirp diagnostics**, not speech examples or model outputs.

## Restore the missing examples

See [exact missing-file list and Colab export commands](RECOVERY.md). No original audio or checkpoint is regenerated or overwritten. Once real exports are recovered, import their matching provenance/hashes and replace BLOCKED cells with verified links.
