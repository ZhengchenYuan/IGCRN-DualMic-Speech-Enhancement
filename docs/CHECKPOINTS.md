# V1 checkpoint access and evaluation

The formal comparison uses **last.pt from epoch 29 (zero-based), 30 completed epochs** for each independently trained arm. Validation-best checkpoints were not used. Both local supplied checkpoint binaries were SHA256-verified and inspected during delivery preparation.

| Arm | Original Drive file | SHA256 |
|---|---|---|
| Dual | [last.pt](https://drive.google.com/file/d/16THNVb6bH1LfV_mBuTOj5uDWipVkyEU3/view?usp=drivesdk) | `3f002a46900bc754196e0e395457e70bbea5dbcfc06f88924de683208ed4bd3a` |
| Duplicate | [last.pt](https://drive.google.com/file/d/1WgdwmHYXgXwTjA8d9O84_tLbFoMB1LYG/view?usp=drivesdk) | `1bde828ac8d03d6fce63e2db7a0513934e34ca8eee40236ffa65e0cd73f9d5f3` |

**Access checked 2026-10-27:** Google Drive permission metadata lists `anyone: reader` for both checkpoint files. Open the links above and use Drive's Download action; verify the downloaded binaries against the hashes in the table. No sharing permissions were changed by this repository cleanup. Checkpoints remain outside Git.

## Verified metadata

Both checkpoints match [configs/small.yaml](../configs/small.yaml): seed 2026, width 64, depth 6, FFT 512, hop 256, Adam learning rate 0.0002, batch 1, 30 epochs. Modes are respectively `dual` and `duplicate`. Each contains model/optimizer state, RNG state and epoch-level resume information. Both bind to these original manifests:

| Manifest | SHA256 |
|---|---|
| Train | `64088cffb3148e1d299f979ff2fbf8c528aec2f60356ef9239807b4a009641d0` |
| Validation | `cad5bc7cf0f78637ebd2ac6fc8718b5fbcedfbae8dfdead3cca0ec3d0393aad2` |
| Test | `08d960e741133f37b711606a954a51c15a944e904701d3f910a1bb2f98091f5e` |

The [committed test manifest](../manifests/evaluation_manifest.jsonl) is the original 120-row file. [Evaluation provenance](../results/v1/provenance.json) independently records its hash and both checkpoint hashes. The original full synchronized test waveform set was not archived. Train/valid manifest contents and RIR files are not bundled. The four original mono audio exports remain available for inspection and metric verification. Do not replace their hashes with those from a newly generated dataset.

## Restore and evaluate in Colab

After obtaining access, download the two files using Drive's Download action and place them at `/content/v1_checkpoints/dual/last.pt` and `/content/v1_checkpoints/duplicate/last.pt`. Alternatively, the owner can copy the existing mounted Drive files without changing them:

```bash
mkdir -p /content/v1_checkpoints/dual /content/v1_checkpoints/duplicate
cp /content/drive/MyDrive/IGCRN_NIO/dual/last.pt /content/v1_checkpoints/dual/last.pt
cp /content/drive/MyDrive/IGCRN_NIO/duplicate/last.pt /content/v1_checkpoints/duplicate/last.pt
sha256sum /content/v1_checkpoints/dual/last.pt /content/v1_checkpoints/duplicate/last.pt manifests/evaluation_manifest.jsonl
```

Compare all three hashes to the tables above before proceeding. Restore the original V1 test mixture and target WAVs to the exact paths recorded in the manifest, originally `/content/igcrn_experiment_v2/`. That historical directory name does **not** identify the later PRA V2 experiment. The original full test waveform set is not currently available. The following command documents the original evaluation interface for retained original inputs; it is not a claim that all 120 cases can currently be rerun. For a new controlled dataset, follow the main README and retain its new manifests and results separately. Audio exports in `examples/` contain only Mic1 and cannot replace synchronized inputs. A different data location requires a documented derived manifest; retain the original unchanged and record the derived manifest's separate hash.

From repository root, with original audio restored:

```bash
python -m scripts.evaluate \
  --manifest manifests/evaluation_manifest.jsonl \
  --dual /content/v1_checkpoints/dual/last.pt \
  --duplicate /content/v1_checkpoints/duplicate/last.pt \
  --output /content/drive/MyDrive/IGCRN_NIO/v1_review_evaluation
```

Use a fresh output directory to preserve original evaluation exports. The evaluator checks audio hashes and matched checkpoint configuration/budget. Only load trusted checkpoint files: they include Python RNG state and require pickle-based loading. Exact historical regeneration also requires the original train/valid manifests, source catalog, selected raw recordings, RIRs and environment snapshot; these are not replaced by the current public-data preparation presets.
