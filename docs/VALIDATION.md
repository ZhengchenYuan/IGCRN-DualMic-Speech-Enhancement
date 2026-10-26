# Validation record

Local verification on 2026-10-26 (macOS CPU):

- Seven pytest cases passed: finite model gradients and output shapes, frequency-shared LSTM reshaping, STFT/ISTFT round trip, SI-SDR sanity/RIR determinism, state-dict round trip, simulated audio/hash/SNR/duplicated-input checks, and end-to-end independent training/resume/evaluation.
- Tiny synthetic fixed-batch overfit diagnostic passed its required 30% loss reduction in 60 optimizer steps. This is an optimizer check, not research training or evidence of speech enhancement quality.
- Epoch-boundary resume reproduced uninterrupted final model tensors bit-for-bit on the diagnostic fixture.
- Full default model: 1,342,540 trainable parameters. Reduced smoke model: 139,860.
- Chinese report: one page, 595.276 × 841.890 PDF points (A4); rendered with PyMuPDF and visually inspected.
- Four original synthetic WAV diagnostics generated with seed 2026; SHA256 recorded in `examples/manifest.json`.

The available local Python 3.10 environment used PyTorch 2.3.1, NumPy 1.26.4, SciPy 1.13.1, SoundFile 0.13.1, pystoi 0.4.1, PyYAML 6.0.1, pytest 8.4.2; PDF generation/rendering used PyMuPDF 1.28.0. These differ from the pinned Colab-oriented requirements. Dependency downloads were slow; the pinned environment and NVIDIA T4 execution were not locally verified. CI defines a separate pinned-environment CPU check; consult its actual run status rather than assuming it passed.

No public-corpus research training, held-out speech evaluation, T4 memory measurement, trained checkpoint publication or perceptual success/failure assessment has been performed. Those deliverables remain explicitly pending.
