# V1 submission verification — 2026-10-28

The repository presents the completed V1 experiment: two independent 30-epoch training runs and evaluation on the same 120 held-out cases. V2 code is retained as experimental development and contributes no formal results.

## Verified submission evidence

- Original summary, 360 method/case CSV rows, evaluation provenance, candidate selection and both 30-epoch training histories are preserved.
- The original test manifest contains 120 unique cases, 40 per SNR. Its SHA256 and both final checkpoint SHA256 records are unchanged.
- Four original audio groups (`test_00000`–`test_00003`) provide 16 unmodified WAVs with source attribution and hashes. Their samples reproduce corresponding CSV SI-SDR, SI-SDRi and STOI values within 1e-6 locally.
- Quantitative Case Analysis is derived from the complete CSV. Only `test_00003` among its four ranked cases has bundled audio; the other three are quantitative-only observations.
- Drive metadata lists `anyone: reader` on both linked final checkpoints. No sharing permissions were changed during repository preparation.
- 51 relevant tests passed in 6.30 seconds: `python -m pytest -q tests/test_core.py tests/test_prepare_data.py tests/test_simulation.py tests/test_submission.py tests/test_export_examples.py`.
- Repository Markdown relative links resolve. All 25 formal result/manifest/real-audio artifacts checked against the prior revision remain byte-identical; all four relocated synthetic WAVs also retain their original hashes. Model, loss, training configuration and training algorithm were not modified.

Tests ran on the existing local CPU environment (Python 3.10, PyTorch 2.3.1, NumPy 1.26.4, SciPy 1.13.1, SoundFile 0.13.1), not a reconstructed historical Colab environment. Tests include synthetic simulation fixtures and inference-only export fixtures, with no optimizer training or V2 execution. The submission workflow uses the same V1-focused tests without optimizer or PRA/V2 execution; no claim is made about its remote run before GitHub reports it.

## Reproducibility scope

The original full test waveform set was not archived. Original manifests and metric exports preserve the completed evaluation record; the four original audio exports support direct inspection and metric checks. New controlled experiments are supported by the public-data pipeline, configs and fixed seeds, but do not constitute bitwise restoration of all historical V1 data. Historical recovery work was stopped; its notes are archived separately in `docs/internal/V1_AUDIO_RECOVERY_HISTORY.md`.

The author-supplied final two-page Chinese report is committed at `report/report_zh.pdf` without modification; its SHA256 is recorded in `report/manifest.json`. The author’s single-listener, non-blind observations for all four available audio cases are documented in `examples/README.md`. These are case observations, not formal subjective scores or causal failure diagnoses. No paired confidence intervals are asserted. No training, V2 execution, reconstruction search or GitHub result alteration was performed during this presentation cleanup.

## Final delivery cleanup

- Root README links the final author-supplied PDF and the original experiment artifacts. The PDF's `53060b3` identifies the experiment snapshot; its contents were not regenerated or edited during import.
- Early validation claims are labeled historical, with links to the completed V1 record.
- Synthetic chirps are separated under `examples/diagnostics/`; the generator targets only that directory. The obsolete pending-results report generator was removed.
- Colab uses the same explicit V1-only test list as README/CI; the default checks do not launch optimizer training or V2 tests.
- Evaluation now refuses existing non-empty output directories before loading audio or models. Metric calculations and model inference are unchanged; tests verify existing result files are preserved.
- Final PDF page count/A4 dimensions and file hash, full-model parameter count, source/audio hashes, Markdown file links and notebook syntax were checked. No historical WAV reconstruction was attempted.
