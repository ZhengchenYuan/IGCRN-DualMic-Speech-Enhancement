# IGCRN Dual-Microphone Speech Enhancement

A focused PyTorch reproduction of [Liu and Zhang, INTERSPEECH 2021](https://arxiv.org/abs/2107.11968), testing whether a second microphone helps under a controlled, independently trained ablation.

**V1: Completed Experiment.** Results, the fixed 120-case evaluation manifest and four real audio groups are committed. Checkpoint sharing and the separately prepared final report remain outstanding. **V2: Experimental / Not Trained**; it contributes no results below.

## Project Overview

The offline pipeline enhances synchronized 16 kHz audio and outputs one waveform. Dual uses Mic1 + Mic2; Duplicate uses Mic1 + Mic1 with the same architecture, independently initialized and trained. Both use identical training/validation manifests, configuration and training budget. The formal comparison uses each model's final checkpoint after 30 epochs.

## IGCRN Architecture

[Model and loss](igcrn/model.py): square-root Hann STFT → four real/imaginary features → six 64-channel frequency-preserving gated 5×1 convolutions → shared per-frequency, two-layer temporal Bi-LSTM → separate amplitude/phase decoders with skip connections → amplitude masking plus mapping → predicted unit phase → ISTFT.

The frequency axis remains at 257 bins. One temporal Bi-LSTM shares its weights across frequencies; bidirectionality makes the system offline. Input shape is `B × 2 × N`, spectrum `B × 2 × 257 × T`, recurrent input `(B × 257) × T × 64`, output `B × N`. The model has **1,342,540 parameters**. Loss combines cube-root magnitude and compressed complex real/imaginary MSE. [Implementation choices and paper deviations](docs/DEVIATIONS.md) are explicit.

## Dataset and Acoustic Simulation

V1 uses Mini LibriSpeech train-clean-5 (CC BY 4.0) and selected ESC-10 environmental recordings (CC BY 3.0). [Source URLs, attribution and license verification](docs/DATA_SOURCES.md) are recorded by the preparation pipeline. Speech speakers and original noise recordings are split before synthesis.

The room is 5 × 5 × 3 m, microphone spacing 2 cm, source radius 1.5 m. The simulator uses direct sound plus six first-order wall reflections, a fixed 0.65 pressure reflection coefficient and fractional delays. Speech/noise directions differ; test angles are interleaved with the training grid. SNR is defined from reverberant speech/noise energy at Mic1; noise scaling and peak normalization are common to both microphones. The target is **noise-free speech at Mic1 including its room response**, with no dereverberation or target realignment.

The [fixed evaluation manifest](manifests/evaluation_manifest.jsonl) contains real source identities, source/mixture hashes, offsets, angles, SNR, gain and RIR references. Its original Colab paths are retained. Full datasets and checkpoints are not stored in Git.

## Training Setup

| Setting | V1 |
|---|---|
| Train / validation / test | 600 / 120 / 120 mixtures |
| Sample rate / duration | 16 kHz / 2 seconds |
| SNR | −3, 0, +3 dB; 40 test cases each |
| Configuration / seed | [small.yaml](configs/small.yaml) / 2026 |
| Optimizer | Adam, learning rate 0.0002; gradient clipping 5 |
| Budget per arm | Batch 1, 30 epochs, 18,000 updates |
| Compute | Colab T4; approximately 55 minutes per arm |
| Formal checkpoint | `last.pt`, epoch 29 (zero-based) |

Runtime is the experiment owner's recorded estimate. Both supplied checkpoints were locally verified to have matching configuration, epoch, batch size and train/validation manifest hashes. Peak GPU memory and the historical environment snapshot are unavailable. Checkpoint saving/resuming is supported at completed epoch boundaries.

## Evaluation Results

Macro averages over the same 120 held-out cases, using the same reverberant target:

| Method | SI-SDR (dB) | SI-SDRi (dB) | STOI |
|---|---:|---:|---:|
| Unprocessed Mic1 | 0.0136 | 0.0000 | 0.7633 |
| Dual: Mic1 + Mic2 | 4.6104 | 4.5968 | 0.7895 |
| Duplicate: Mic1 + Mic1 | 5.1746 | 5.1610 | 0.7805 |

Both models improve average SI-SDR and STOI over Mic1. Duplicate exceeds Dual by **0.5642 dB SI-SDRi**; Dual exceeds Duplicate by **0.00905 STOI**. This direction holds in all three SNR groups. The second microphone therefore shows **no consistent advantage across metrics** in this experiment. These results do not claim the original paper's performance.

[Per-SNR results and analysis status](results/v1/README.md), [original summary](results/v1/summary.json), [per-utterance metrics](results/v1/per_utterance.csv), [provenance](results/v1/provenance.json) and [artifact hashes](results/v1/artifact_sha256.json) are available. SI-SDRi subtracts each case's unprocessed Mic1 SI-SDR; STOI uses `pystoi` with `extended=False`. No temporal alignment is applied. No paired confidence intervals or completed listening analysis are claimed.

## Reproduction Instructions

Use Python 3.10–3.12, from repository root:

```bash
git clone https://github.com/ZhengchenYuan/IGCRN-DualMic-Speech-Enhancement.git
cd IGCRN-DualMic-Speech-Enhancement
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
# Lightweight V1 checks: no optimizer training, download or V2 execution
python -m pytest -q tests/test_core.py tests/test_prepare_data.py tests/test_simulation.py tests/test_submission.py
```

These cover shapes/gradients, parameter count, waveform reconstruction, checkpoint roundtrip, source selection, silence retries, deterministic WAV hashes and delivered artifact consistency. A separate optimizer-based tiny-set diagnostic is available as `python -m scripts.overfit`; it is not a completed full experiment. The full development suite (`python -m pytest -q`) also includes short training fixtures and optional PRA/V2 tests; install `requirements-v2.txt` for that suite.

For Colab, mount Drive, clone the repository under `/content/`, change to its root, then run the same pip and `python -m pytest` commands. [The Colab notebook](notebooks/colab.ipynb) provides exact cells with repository-root subprocesses and training disabled by default. It prepares a new dataset; it does not restore the original V1 split automatically.

To generate a **new** V1-style reproduction dataset (not bitwise restoration of historical V1):

```bash
python -m scripts.prepare_data --mode experiment --output /content/igcrn_v1_reproduction_sources --seed 2026
python -m scripts.simulate --catalog /content/igcrn_v1_reproduction_sources/catalog.jsonl --output /content/igcrn_v1_reproduction --count 600 --seconds 2 --seed 2026
```

This creates 600/120/120 mixtures. The smaller `--mode smoke` plus simulation `--count 9` creates 9/3/3. Preparation downloads the approximately 332 MB Mini LibriSpeech archive, **not train-clean-100**; smoke mode still needs that archive once. Raw data and mixtures stay in `/content/`. Keep only manifests, configuration, environment records, important checkpoints and final results in Drive. Do not overwrite original V1 directories.

Explicit training commands for a new run (not required to inspect the submission):

```bash
python -m scripts.train --data /content/igcrn_v1_reproduction --mode dual --config configs/small.yaml --output /content/drive/MyDrive/IGCRN_NIO/v1_reproduction/dual
python -m scripts.train --data /content/igcrn_v1_reproduction --mode duplicate --config configs/small.yaml --output /content/drive/MyDrive/IGCRN_NIO/v1_reproduction/duplicate
```

Resume with the same arguments plus `--resume <arm-directory>/last.pt`. Compare equal completed epochs. For evaluation of the **original** checkpoints, follow the [restoration, SHA256 and evaluation commands](docs/CHECKPOINTS.md); original synchronized mixture files are required. Do not regenerate data underneath existing checkpoints. Record `pip freeze`, GPU details, git commit and commands for every new run. Numerical bitwise reproducibility across different device/software stacks is not guaranteed.

## Audio Examples

[**Listen to V1 success/failure examples and inspect per-case metrics →**](examples/README.md)

The ranked four-case selection is currently **BLOCKED**: only `test_00003` is available. Original `test_00000`–`test_00003` exports remain available with their true IDs and unchanged audio. See the listening page for missing-data recovery. Labels describe Dual SI-SDRi only; listening analysis is pending.

## Checkpoints and Audio Examples

[Checkpoint access and verified metadata](docs/CHECKPOINTS.md): real Drive links exist, but both are currently **restricted**, not public downloads. Owner sharing action is required before interviewer access.

[Four real test audio groups](examples/README.md) include Mic1, reverberant target, Dual and Duplicate outputs (16 WAVs). Source attribution and hashes accompany them. These mono exports do not include the original Mic2 input. Older synthetic chirps are explicitly labeled diagnostics. Ranked success/failure candidates are metric-only until listening is completed.

[Final report location](report/README.md): the final reproduction report is being prepared separately by the author; no final PDF is included.

## Limitations and Future Work

One seed, a small public-data subset, two-second crops and simplified first-order RIRs limit generalization. Missing paired confidence intervals and listening analysis limit interpretation. The current evidence does not establish why Dual has lower SI-SDRi, and spatial simulation limitations are hypotheses rather than demonstrated causes.

**V2 is experimental and not formally trained.** It explores high-order PRA room simulation while preserving model/loss and a fair independently trained control. [V2 development status](docs/v2/STAGE1_RESULTS.md), [protocol](docs/v2/README.md) and [disabled-by-default notebook](notebooks/v2_colab.ipynb) are separate from V1 results. Local acceptance does not establish Colab readiness or completed V2 training.

## References and AI Assistance Disclosure

- J. Liu and X. Zhang, *Inplace Gated Convolutional Recurrent Neural Network for Dual-channel Speech Enhancement*, INTERSPEECH 2021. [Paper](https://arxiv.org/abs/2107.11968).
- [Dataset licenses and original recording attribution](docs/DATA_SOURCES.md); [implementation deviations](docs/DEVIATIONS.md).

The implementation was written for this reproduction using PyTorch, NumPy/SciPy, SoundFile and pystoi; dataset audio retains its original licensing. AI assistance was used for implementation, tests, diagnostics and documentation. The author is responsible for understanding the implementation, validating experiments and writing the final report. No AI-generated listening assessment is presented.
