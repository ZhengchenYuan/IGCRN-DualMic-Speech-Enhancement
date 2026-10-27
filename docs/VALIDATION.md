# Historical development validation record

This page preserves early development snapshots. Statements below that training, evaluation or publication had not occurred describe that historical stage, not the completed V1 experiment. The old one-page template was retired. For the current submission, see [V1 verification](SUBMISSION_STATUS.md), [formal results](../results/v1/README.md), [listening observations](../examples/README.md#non-blind-listening-observations) and the [final two-page report](../report/report_zh.pdf).

Local verification on 2026-10-26 (macOS CPU):

- Seven pytest cases passed: finite model gradients and output shapes, frequency-shared LSTM reshaping, STFT/ISTFT round trip, SI-SDR sanity/RIR determinism, state-dict round trip, simulated audio/hash/SNR/duplicated-input checks, and end-to-end independent training/resume/evaluation.
- Tiny synthetic fixed-batch overfit diagnostic passed its required 30% loss reduction in 60 optimizer steps. This is an optimizer check, not research training or evidence of speech enhancement quality.
- Epoch-boundary resume reproduced uninterrupted final model tensors bit-for-bit on the diagnostic fixture.
- Full default model: 1,342,540 trainable parameters. Reduced smoke model: 139,860.
- Chinese report: one page, 595.276 × 841.890 PDF points (A4); rendered with PyMuPDF and visually inspected.
- Four original synthetic WAV diagnostics generated with seed 2026; SHA256 originally recorded in `examples/manifest.json`; the unchanged WAVs and updated path index now reside in [synthetic diagnostics](../examples/diagnostics/README.md).

The available local Python 3.10 environment used PyTorch 2.3.1, NumPy 1.26.4, SciPy 1.13.1, SoundFile 0.13.1, pystoi 0.4.1, PyYAML 6.0.1, pytest 8.4.2; PDF generation/rendering used PyMuPDF 1.28.0. These differ from the pinned Colab-oriented requirements. Dependency downloads were slow; the pinned environment and NVIDIA T4 execution were not locally verified. CI defines a separate pinned-environment CPU check; consult its actual run status rather than assuming it passed.

No public-corpus research training, held-out speech evaluation, T4 memory measurement, trained checkpoint publication or perceptual success/failure assessment has been performed. Those deliverables remain explicitly pending.

## Public-data pipeline update (2026-10-26)

Full local CPU suite: **27 tests passed**.

- Preparation tests cover smoke and configurable experiment modes, deterministic byte-identical catalogs, disjoint speakers/original noise IDs, source metadata/hashes, checksum/cache corruption, license failures before audio download, unsafe tar paths/symlinks, invalid configuration and existing-simulator compatibility. Tests use generated audio and explicitly identified metadata fixtures, not fabricated public-source evidence.
- The source publisher page, Mini LibriSpeech checksum list, commit-pinned ESC metadata and per-clip license file were downloaded and checked. All 175 eligible original environmental-noise IDs had supported upstream attribution.
- Four actual default smoke-mode ESC WAVs downloaded and decoded successfully: original source IDs 35687, 141584, 195710, 65747. Each is mono, 44.1 kHz, five seconds; actual SHA256 receipts were generated outside Git. Original labels are CC0, CC0, CC0 and CC-BY respectively.
- The 332 MB speech archive transfer was stopped because this environment's download was unusually slow. A complete live public-corpus preparation/simulation was not claimed; fixture-backed extraction and simulator integration passed. Colab/T4 execution remains unverified.
- Notebook code cells compile, invoke subprocesses from the repository root, use `python -m pytest`, keep audio under `/content/`, save important artifacts under `/content/drive/MyDrive/IGCRN_NIO/`, and disable full training by default.
- The Chinese report's source description was updated, regenerated and visually checked; it remains exactly one A4 page. No full training or experimental score generation was launched.

## Silent-source retry fix (2026-10-26)

- Full local CPU suite: **38 tests passed**. No full training was launched.
- Actual silent and near-silent speech/noise fixture recordings recover through deterministic split-local selection. Accepted rows preserve hashes, source identities, offsets, SNR and retry counts across repeated runs.
- Exhaustion is bounded to 20 total attempts; errors identify both early and late split/sample failures. The twentieth attempt can succeed with `retry_count=19`.
- A 600-training-sample synthetic simulation generated all 600/120/120 rows and waveform/RIR artifacts with the original repeating SNR schedule. This is a simulator check, not a research experiment.
- Ordinary I/O failures and keyboard interruption after a completed training split publish no partial dataset or manifests. Full output is atomically published only after all splits succeed, and existing non-empty output directories are preserved.

## WAV timestamp reproducibility fix (2026-10-26)

- Investigated the Colab report of 36 passes and two hash mismatches. A local isolated experiment wrote the same float waveform twice, separated by 1.1 seconds. Decoded samples were exactly equal; full WAV hashes differed solely at the `PEAK` timestamp. SoundFile 0.13.1 / libsndfile 1.2.2 reproduced the issue.
- Added a delayed, directory-changing simulator regression before the fix: it failed on WAV bytes while source hashes and decoded waveforms matched. The earlier same-second tests could pass by timing luck; their full-file comparisons were correct and remain unchanged.
- Replaced only generated mixture/target serialization with public SciPy float32 WAV writing, which omits the timestamped optional chunk. No change to RNG draws, retries, filtering, precision, SNR scheduling, source files or full-file SHA256 hashing. Mono and stereo sample values were checked against the previous writer and matched exactly.
- Full local CPU suite: **39 tests passed**, including the delayed regression's source immutability, sample equality, complete WAV-byte/hash equality, exact manifest comparison apart from output paths, and absence of a PEAK chunk. No research training job was launched. Colab execution was not available locally; the environment versions above still apply.
