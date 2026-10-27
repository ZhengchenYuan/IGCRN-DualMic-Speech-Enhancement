# Audio examples

Four real V1 held-out speech cases are provided below. Each WAV is mono, 16 kHz, 2 seconds, IEEE float32. Open a link to download/listen. These are the evaluator's first four cases, not cherry-picked listening successes.

| Case | SNR (dB) | Input Mic1 | Target | Dual | Duplicate |
|---|---:|---|---|---|---|
| test_00000 | -3 | [WAV](v1/test_00000/mic1.wav) | [WAV](v1/test_00000/target.wav) | [WAV](v1/test_00000/dual.wav) | [WAV](v1/test_00000/duplicate.wav) |
| test_00001 | 0 | [WAV](v1/test_00001/mic1.wav) | [WAV](v1/test_00001/target.wav) | [WAV](v1/test_00001/dual.wav) | [WAV](v1/test_00001/duplicate.wav) |
| test_00002 | 3 | [WAV](v1/test_00002/mic1.wav) | [WAV](v1/test_00002/target.wav) | [WAV](v1/test_00002/dual.wav) | [WAV](v1/test_00002/duplicate.wav) |
| test_00003 | -3 | [WAV](v1/test_00003/mic1.wav) | [WAV](v1/test_00003/target.wav) | [WAV](v1/test_00003/dual.wav) | [WAV](v1/test_00003/duplicate.wav) |

[Case metadata, original source attribution and SHA256](v1/manifest.json) accompany the exports. Speech derives from Mini LibriSpeech (CC BY 4.0); environmental recordings derive from ESC-10 (CC BY 3.0). Original per-recording attribution and source/license URLs are retained in that manifest and the [evaluation manifest](../manifests/evaluation_manifest.jsonl). Repository MIT licensing does not replace those audio licenses.

The target is noise-free reverberant Mic1, not dry speech. Exports contain only Mic1, not the original synchronized Mic2; they cannot alone reproduce Dual inference. WAV container hashes for exported targets need not equal original simulator target hashes. No audio quality or failure mechanism has been established by listening yet.

The four numbered WAV files at this directory's root and `manifest.json` are older **synthetic chirp diagnostics**, not speech examples or model outputs. `scripts.make_examples` regenerates only those diagnostics.
