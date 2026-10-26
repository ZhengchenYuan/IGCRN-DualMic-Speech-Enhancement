# IGCRN Dual-Microphone Speech Enhancement

Focused, explainable PyTorch reproduction of [Liu & Zhang (INTERSPEECH 2021)](https://arxiv.org/abs/2107.11968). **No research training/evaluation has been completed. Results and trained checkpoints are pending.**

Repository: https://github.com/ZhengchenYuan/IGCRN-DualMic-Speech-Enhancement

## Setup

Python 3.10–3.12 recommended; Linux CUDA for experiments. From repository root:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
python -m scripts.overfit
python -c "from igcrn import IGCRN; m=IGCRN(); print(sum(p.numel() for p in m.parameters()))"
```
PyTorch 2.8 CUDA wheels support T4; Colab may require restarting its runtime after installation. No audio-device dependency is needed. Record `pip freeze`, `nvidia-smi`, commit hash and command lines alongside each run. Determinism is intended within the same device/software stack, not guaranteed bitwise across CPU/CUDA versions.

## Architecture and code map

`igcrn/model.py`: square-root Hann STFT → four real/imaginary channels → six 64-channel stride-one gated 5×1 convolutions → reshape B×F,T,C → shared two-layer temporal Bi-LSTM → projection → separate amplitude/phase skip decoders → mask × Mic1 amplitude + mapping → unit phase → ISTFT. See [all fidelity decisions](docs/DEVIATIONS.md). The network does not reduce frequency resolution; the LSTM has no separate weights for different bins. It is offline because it is bidirectional. Frequency convolutions do mix neighbors; preserving bins does not mean frequency independence throughout.

`igcrn/data.py`: synchronized audio and hash validation. `scripts/simulate.py`: RIR construction and mixtures. `scripts/train.py`: deterministic optimization, validation and resume. `scripts/evaluate.py`: SI-SDRi/STOI and listening candidates. `scripts/overfit.py`: fixed-batch diagnostic.

## Public data and simulation

Download and extract [LibriSpeech](https://www.openslr.org/12) train-clean-100, dev-clean, test-clean and [MUSAN](https://www.openslr.org/17) outside this repository. Both resource pages specify CC BY 4.0; retain corpus attribution/license files and cite original authors. Use only MUSAN `noise` recordings. Downloads are several GB; they are intentionally not automatic. Custom legally obtained sources can use the [catalog schema](manifests/README.md).

```bash
export IGCRN_DATA=/content/igcrn_data
mkdir -p "$IGCRN_DATA"
python -m scripts.catalog --librispeech /content/LibriSpeech --musan /content/musan --output "$IGCRN_DATA/catalog.jsonl"
python -m scripts.simulate --catalog "$IGCRN_DATA/catalog.jsonl" --output "$IGCRN_DATA/mixtures" --count 300 --seconds 2 --seed 2026
```
300 train / 60 validation / 60 test examples is a small development experiment, not a reproduction of the paper's dataset scale. Speech speakers and noise recording IDs are disjoint. Sorted sources and seed fix selection/offsets. Frozen WAVs, RIR arrays, hashes and JSONL manifests capture exact mixtures. 16 kHz synchronized channels use common source/noise signals convolved with their respective paths. Noise scaling is based on reverberant Mic1 energy; target is clean reverberant Mic1. No target time shifting or dereverberation.

## Fair independent ablation

Run both arms from scratch using the same seed, configuration and frozen manifests. The duplicated arm changes only the input channel selection; never initialize it from the dual checkpoint.
```bash
python -m scripts.train --data "$IGCRN_DATA/mixtures" --mode dual --output /content/drive/MyDrive/IGCRN/dual --config configs/small.yaml
python -m scripts.train --data "$IGCRN_DATA/mixtures" --mode duplicate --output /content/drive/MyDrive/IGCRN/duplicate --config configs/small.yaml
# Resume at an epoch boundary; use exactly the same config and batch size:
python -m scripts.train --data "$IGCRN_DATA/mixtures" --mode dual --output /content/drive/MyDrive/IGCRN/dual --config configs/small.yaml --resume /content/drive/MyDrive/IGCRN/dual/last.pt
```
Checkpoint includes model, optimizer, epoch, RNG states, loader generator, manifests, mode and config. Writes replace a temporary file atomically on local filesystems; Drive disconnection still requires checking the saved file. Resume is exact at completed epoch boundaries; interruption mid-epoch restarts that epoch. Load only trusted checkpoints (`weights_only=False` includes Python RNG state). `best.pt` uses validation loss; use equal-epoch `last.pt` for the primary fixed-budget comparison. Do not extend one arm alone. Seed 2026 is the primary run; repeat both arms with seeds 2027/2028 for uncertainty when resources permit.

Full configuration uses width 64, six layers, Adam 2e-4, 30 epochs, batch 1, 2 s clips. T4-compatible design starts at batch 1 in float32; actual peak VRAM is unmeasured. If out of memory, shorten clips by resimulating the entire shared dataset and restart both arms. `configs/smoke.yaml` is reduced-width debugging only. `python -m scripts.overfit --manifest "$IGCRN_DATA/mixtures/train.jsonl"` checks a real fixed mixture before longer runs.

## Evaluation and submission artifacts

```bash
python -m scripts.evaluate --manifest "$IGCRN_DATA/mixtures/test.jsonl" --dual /content/drive/MyDrive/IGCRN/dual/last.pt --duplicate /content/drive/MyDrive/IGCRN/duplicate/last.pt --output /content/drive/MyDrive/IGCRN/evaluation
```
SI-SDR removes the means, projects estimate onto the reverberant target, and uses no temporal realignment. SI-SDRi subtracts the same utterance's unprocessed Mic1 score; Mic1 SI-SDRi is zero by definition. STOI uses `pystoi`, 16 kHz, `extended=False`. Report macro means overall and at −3/0/+3 dB; include utterance count. Silent targets fail explicitly. Evaluator enforces identical config, training-manifest hashes, epochs and batch size. Metrics are not the paper's SDR/PESQ protocol.

| Method | SI-SDRi (dB) | STOI |
|---|---|---|
| Unprocessed Mic1 | 0 (definition) | pending |
| Mic1 + Mic2 | pending | pending |
| Independently trained Mic1 + Mic1 | pending | pending |

Repeat this table per SNR after evaluation. Publish `per_utterance.csv`, `summary.json`, `provenance.json` and the frozen evaluation manifest. The current [manifest](manifests/evaluation_manifest.jsonl) is intentionally empty. Export includes the first four target/baseline/model audio cases; metric-ranked best/worst candidates are recorded separately for listening. Listen to candidate cases before claiming success/failure. Consider close arrival angles, weak speech and room mismatch in error analysis.

[Four bundled audio diagnostics](examples/README.md) are original synthetic signals, not trained speech examples. Generate with `python -m scripts.make_examples`. Real 3–5 speech cases remain a post-training deliverable with source attribution. [Checkpoint access instructions](docs/CHECKPOINTS.md); [one-page Chinese report](report/report_zh.pdf) and its editable generator `scripts/make_report.py`.

## Colab

Open [notebooks/colab.ipynb](notebooks/colab.ipynb), select GPU/T4, and run the cells. Mount Google Drive for checkpoints; keep raw datasets/mixtures in `/content` (copy frozen manifests and hashes to Drive). The notebook fails if CUDA is missing. Raw data and temporary outputs stay outside Git. No credentials are required in code.

## Validation

Seven local CPU tests and the tiny overfit diagnostic passed. The full model has 1,342,540 parameters. See [validation record and tested environment](docs/VALIDATION.md); the pinned environment and T4 runtime remain unverified.

## Limitations and AI disclosure

This is a focused architectural reproduction with public replacement data and a simplified RIR model, not a verified reproduction of published scores. T4 performance remains to be measured. AI assistance was used to draft implementation, tests, documentation and report; a human should review the math, run experiments and listen to outputs before submission. Pending result fields must be filled only from saved evaluation outputs.
