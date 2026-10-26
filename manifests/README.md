The evaluation manifest is intentionally empty until actual data is simulated and frozen. It is not an evaluated test set.

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

`scripts.simulate` consumes the catalog unchanged and creates train/valid/test JSONL with frozen audio paths/hashes, SNR, full source catalog entries, offsets, seed, geometry, RIR path and gain. Copy the generated `test.jsonl` to `evaluation_manifest.jsonl` before the final experiment. Publish all three frozen manifests and source catalog with the trained release, retaining attribution. Relocated paths may change manifest hashes: document relocation and keep source/mixture audio hashes unchanged. Never publish private source information.

Keep raw data and mixtures in `/content/`; save only the catalog, preparation record/evidence and simulation manifests to `/content/drive/MyDrive/IGCRN_NIO/manifests/<mode>-seed<seed>/`. Check regenerated manifests against saved ones after a Colab reset before checkpoint resumption.
