# V1 submission verification — 2026-10-27

The repository presents the completed V1 experiment: two independent 30-epoch training runs and evaluation on the same 120 held-out cases. V2 code is retained as experimental development and contributes no formal results.

## Verified submission evidence

- Original summary, 360 method/case CSV rows, evaluation provenance, candidate selection and both 30-epoch training histories are preserved.
- The original test manifest contains 120 unique cases, 40 per SNR. Its SHA256 and both final checkpoint SHA256 records are unchanged.
- Four original audio groups (`test_00000`–`test_00003`) provide 16 unmodified WAVs with source attribution and hashes. Their samples reproduce corresponding CSV SI-SDR, SI-SDRi and STOI values within 1e-6 locally.
- Quantitative Case Analysis is derived from the complete CSV. Only `test_00003` among its four ranked cases has bundled audio; the other three are quantitative-only observations.
- Drive metadata lists `anyone: reader` on both linked final checkpoints. No sharing permissions were changed during repository preparation.
- 47 relevant tests passed in 6.16 seconds: `python -m pytest -q tests/test_core.py tests/test_prepare_data.py tests/test_simulation.py tests/test_submission.py tests/test_export_examples.py`.
- Repository Markdown relative links resolve. All 29 protected result/manifest/audio artifacts are byte-identical to the previous main revision. Models, loss, scripts, configs and notebooks were not modified.

Tests ran on the existing local CPU environment (Python 3.10, PyTorch 2.3.1, NumPy 1.26.4, SciPy 1.13.1, SoundFile 0.13.1), not a reconstructed historical Colab environment. Tests include synthetic simulation fixtures and inference-only export fixtures, with no optimizer training or V2 execution. The submission workflow uses the same V1-focused tests without optimizer or PRA/V2 execution; no claim is made about its remote run before GitHub reports it.

## Reproducibility scope

The original full test waveform set was not archived. Original manifests and metric exports preserve the completed evaluation record; the four original audio exports support direct inspection and metric checks. New controlled experiments are supported by the public-data pipeline, configs and fixed seeds, but do not constitute bitwise restoration of all historical V1 data. Historical recovery work was stopped; its notes are archived separately in `docs/internal/V1_AUDIO_RECOVERY_HISTORY.md`.

The final report is prepared separately by the author. Case-specific perceptual findings and paired confidence intervals are not asserted. No training, V2 execution, reconstruction search or GitHub result alteration was performed during this presentation cleanup.
