# V1 delivery audit — 2026-10-27

## Verified locally

- GitHub main matched local `091c431` before delivery edits; no pre-existing working-tree changes were overwritten.
- Real evaluation summary, 360 method/case metric rows, provenance and listening candidates imported unchanged.
- Original test manifest imported byte-for-byte: 120 unique cases, 40 at each SNR; recorded SHA256 preserved.
- Both local final checkpoint binaries match the published SHA256 values. Both are epoch 29 with identical configuration, batch size and train/validation manifest hashes.
- Four real audio groups imported unchanged: 16 mono IEEE float32 WAVs, 16 kHz, 32,000 samples, finite and non-silent; original recording attribution retained. No listening interpretation is claimed.
- 41 selected existing V1 and new artifact-consistency tests passed in 5.39 seconds. Command: `python -m pytest -q tests/test_core.py tests/test_prepare_data.py tests/test_simulation.py tests/test_submission.py`.
- All checked Markdown relative links resolve; notebook code parses; no dataset archive or checkpoint binary is included.
- No training, V2 execution, dataset download or report generation was performed. Full development tests containing optimizer training and V2 execution were deliberately not run in this delivery-only pass.

Test environment: local CPU, Python 3.10, PyTorch 2.3.1, NumPy 1.26.4, SciPy 1.13.1, SoundFile 0.13.1, using the existing temporary environment. This is not a validation of the pinned Colab environment or a fresh metric reevaluation.

## Remaining delivery actions

1. Owner must enable interviewer access to both checkpoint links. Drive metadata showed `shared=false` and owner-only permissions on 2026-10-27; no sharing permissions were changed.
2. Author will prepare the final report separately. The obsolete pending-results PDF was removed; no replacement PDF was generated. The historical `scripts.make_report` generator remains a development tool and must not be treated as the final report; CI no longer invokes it.
3. Human listening success/failure analysis remains outstanding; metric-ranked candidates are not perceptual diagnoses.
4. Exact original-checkpoint reevaluation requires original two-channel mixture/target WAVs at manifest paths. Exact historical training restoration additionally needs original train/valid manifests, selected source catalog/data, RIRs and environment snapshot. These are not bundled or proven accessible via a public archive. Do not upload the entire dataset to Git.

No manual download is needed for the requested summary/CSV/provenance/candidates, four exported audio groups or test manifest: all were available locally and imported. Missing original dataset assets are distinct from those supplied evaluation exports. Follow [checkpoint restoration](CHECKPOINTS.md) for verified paths and commands. V2 remains experimental and not formally trained.
