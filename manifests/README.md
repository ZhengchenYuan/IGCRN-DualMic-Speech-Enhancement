The committed `evaluation_manifest.jsonl` is the original V1 test manifest: 120 held-out cases, 40 per SNR. Its byte-for-byte SHA256 is `08d960e741133f37b711606a954a51c15a944e904701d3f910a1bb2f98091f5e`. It retains original Colab `/content/` paths. The historical directory name `igcrn_experiment_v2` belongs to the completed V1 dataset and does not indicate a completed PRA/V2 experiment. See [checkpoint restoration](../docs/CHECKPOINTS.md). Train/valid manifests, full mixture WAVs and RIR arrays are not bundled.

## Automatic source catalog

Run `python -m scripts.prepare_data --mode smoke --output /content/igcrn_data_smoke --seed 2026` from the repository root. It creates `catalog.jsonl`, `preparation.json` and license/metadata `evidence/` outside Git. See [verified sources](../docs/DATA_SOURCES.md); licenses are checked, not supplied by the user as placeholder strings.

Required simulator fields:

| Field | Meaning |
|---|---|
| `path` | Absolute path to selected distributed FLAC/WAV, retained at original sample rate |
| `kind`, `split` | `speech` or `noise`; `train`, `valid`, `test` |
| `speaker_id` | Original LibriSpeech speaker (speech only) |
| `recording_id` | Original Freesound `src_file`, shared by all takes (noise only) |
| `license`, `source_url` | Verified distributed-dataset license and actual download URL |

Additional preparation fields include `source_id`, `original_source_id`, `selection_seed`, `sha256`, audio dimensions, `license_url`, `license_evidence_url`, speech `archive_member`, and noise `source_revision`, category, original attribution author/title/URL/literal license label. The hash describes the bytes actually obtained, not an unavailable original upstream upload. `preparation.json` stores publisher checksum/evidence hashes, exact selection counts/configuration, and catalog hash.

## Frozen simulation manifests

`scripts.simulate` consumes the catalog unchanged and creates train/valid/test JSONL with frozen audio paths/hashes, SNR, full source catalog entries, offsets, seed, geometry, RIR path and gain. Every new sample also records `retry_count`: the number of rejected silent/near-silent attempts before acceptance (0–19, within at most 20 total attempts). Retry draws use only that split’s source pools and the shared seeded NumPy generator; sample IDs, counts and SNR slots advance only on acceptance. Copy the generated `test.jsonl` to `evaluation_manifest.jsonl` before the final experiment. Publish all three frozen manifests and source catalog with the trained release, retaining attribution. Relocated paths may change manifest hashes: document relocation and keep source/mixture audio hashes unchanged. Never publish private source information.

Keep raw data and mixtures in `/content/`; save only the catalog, preparation record/evidence and simulation manifests to `/content/drive/MyDrive/IGCRN_NIO/manifests/<mode>-seed<seed>/`. Check regenerated manifests against saved ones after a Colab reset before checkpoint resumption.

Simulation output must be new or empty. Audio and all three manifests are staged in a hidden `.incomplete-*` sibling and published together by a single directory rename after full success. A failed/interrupted run does not publish a dataset. Existing non-empty directories are not overwritten; choose a fresh path for retrying an earlier partial run. Keep pre-existing frozen manifests unchanged for checkpoint compatibility.

Generated WAV hashes cover the entire file, not only decoded samples. Simulation writes IEEE float32 WAV through SciPy, omitting the optional wall-clock-dependent `PEAK` chunk previously added by libsndfile. This preserves float32 audio values but changes file/header hashes relative to old simulated WAVs. Do not regenerate datasets underneath saved checkpoints or rewrite their recorded hashes; use a fresh output and keep the exact environment/catalog/seed for reproducibility.
