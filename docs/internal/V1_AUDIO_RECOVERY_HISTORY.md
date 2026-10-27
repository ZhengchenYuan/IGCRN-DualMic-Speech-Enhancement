# Historical V1 audio-recovery notes

**Historical record; recovery work has been stopped by the experiment owner.** This document preserves earlier operational instructions, not a current submission requirement or a runnable next step. Public V1 presentation uses four original audio examples and a separate quantitative-only case analysis.

Colab diagnostics verified all six original source hashes. Of the six reconstructed target-case mixture/target WAVs, only `test_00105/target` matched the original manifest. A `test_00001` control target was bitwise identical to the real export. Recomputing scale/gain in the original arithmetic order produced exactly the same float32 candidates as saved-scalar replay, excluding that difference as the cause. No model inference or verified recovery export followed; no original hashes or thresholds were changed. The original numerical environment was not established, and no further reconstruction is planned.

## Archived instructions (superseded)

# Recover the missing ranked V1 cases

Status: **BLOCKED**. Locally available and committed exports are only `test_00000`–`test_00003`. The connected Drive `IGCRN_NIO/evaluation` folder likewise lists only those four groups. Drive searches for `test_00067` and `test_00105` returned no results; the `test_00096` search failed at transport level. This does not prove the recordings do not exist elsewhere. No original synchronized mixtures were available locally, so no new inference was run.

## Option A: existing real exports

If Colab/Drive already has the following files, download them unchanged (12 WAVs total), with the exporter provenance or source-to-ID mapping:

- `test_00067_{mic1,target,dual,duplicate}.wav`
- `test_00105_{mic1,target,dual,duplicate}.wav`
- `test_00096_{mic1,target,dual,duplicate}.wav`

Do not rename numbered files until their IDs can be verified. Existing `test_00003` is already present and needs no regeneration. Destination directories are `examples/test_00067/`, `examples/test_00105/`, `examples/test_00096/`, with filenames `mic1.wav`, `target.wav`, `dual.wav`, `duplicate.wav`. Import hashes and provenance alongside them; do not replace the original official metric files.

## Option B: export from the original test inputs

Only these **six original files**, plus both trusted final checkpoints, are needed. Restore to their original paths (the historical `igcrn_experiment_v2` directory is V1 data, not the PRA/V2 experiment):

```text
/content/igcrn_experiment_v2/test_00067_mix.wav
/content/igcrn_experiment_v2/test_00067_target.wav
/content/igcrn_experiment_v2/test_00105_mix.wav
/content/igcrn_experiment_v2/test_00105_target.wav
/content/igcrn_experiment_v2/test_00096_mix.wav
/content/igcrn_experiment_v2/test_00096_target.wav
```

Each mix must be the original synchronized **two-channel** file, not a mono export or artificially duplicated Mic1. Restore original bytes matching the committed manifest. Source FLACs, noise recordings and RIRs are not needed for this inference-only export. If these original mixtures were never backed up and the Colab runtime reset, stop: newly generated approximations must not be presented as the same V1 cases.

After mounting Drive, from the latest repository root with dependencies installed:

```bash
python -m scripts.export_examples \
  --manifest manifests/evaluation_manifest.jsonl \
  --metrics results/v1/per_utterance.csv \
  --provenance results/v1/provenance.json \
  --dual /content/drive/MyDrive/IGCRN_NIO/dual/last.pt \
  --duplicate /content/drive/MyDrive/IGCRN_NIO/duplicate/last.pt \
  --ids test_00067 test_00105 test_00096 \
  --output /content/v1_ranked_audio_export
```

The new tool selects explicit IDs, verifies the original manifest/checkpoint/input hashes, uses frozen eval-mode inference, and checks per-case SI-SDR/SI-SDRi/STOI against the existing CSV. It does not train or modify evaluation metrics. Numerical differences greater than 0.001 dB or 0.0001 STOI stop publication; investigate the numerical environment rather than changing the official CSV. Exports are staged and published only after every selected case passes; an existing output directory is refused. New WAVs have their own hashes and provenance; they need not match old libsndfile export container hashes.

Copy this small result only to a **new** Drive directory:

```bash
python - <<'PY'
from pathlib import Path
import shutil
src = Path('/content/v1_ranked_audio_export')
dst = Path('/content/drive/MyDrive/IGCRN_NIO/v1_ranked_audio_export')
if dst.exists():
    raise FileExistsError(f'Preserve existing export: {dst}')
shutil.copytree(src, dst)
PY
```

Download the entire `v1_ranked_audio_export` folder (including `export_provenance.json`) for a follow-up verified import. The current original four groups were **moved only**, with no new inference or re-encoding.

## Verification of the current reorganization

46 V1/export/delivery tests passed locally, including actual metric recomputation from all four existing WAV groups, byte hashes against the pre-existing original asset inventory, ID-based selection, rejection of duplicate/unknown IDs, numerical mismatch rejection, staged cleanup and refusal to overwrite an export. Exporter inference tests use explicitly synthetic inputs and an identity-model fixture; they do not constitute acceptance on the missing real cases. README relative links resolve. The official metric exports, evaluation manifest, checkpoints, model/loss and training code were not changed.
