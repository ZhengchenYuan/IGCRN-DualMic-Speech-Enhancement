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
python -m pytest -q
python -m scripts.overfit
python -c "from igcrn import IGCRN; m=IGCRN(); print(sum(p.numel() for p in m.parameters()))"
```
PyTorch 2.8 CUDA wheels support T4; Colab may require restarting its runtime after installation. No audio-device dependency is needed. Record `pip freeze`, `nvidia-smi`, commit hash and command lines alongside each run. Determinism is intended within the same device/software stack, not guaranteed bitwise across CPU/CUDA versions.

## Architecture and code map

`igcrn/model.py`: square-root Hann STFT → four real/imaginary channels → six 64-channel stride-one gated 5×1 convolutions → reshape B×F,T,C → shared two-layer temporal Bi-LSTM → projection → separate amplitude/phase skip decoders → mask × Mic1 amplitude + mapping → unit phase → ISTFT. See [all fidelity decisions](docs/DEVIATIONS.md). The network does not reduce frequency resolution; the LSTM has no separate weights for different bins. It is offline because it is bidirectional. Frequency convolutions do mix neighbors; preserving bins does not mean frequency independence throughout.

`igcrn/data.py`: synchronized audio and hash validation. `scripts/simulate.py`: RIR construction and mixtures. `scripts/train.py`: deterministic optimization, validation and resume. `scripts/evaluate.py`: SI-SDRi/STOI and listening candidates. `scripts/overfit.py`: fixed-batch diagnostic.

## Public data and simulation

The default pipeline uses [Mini LibriSpeech train-clean-5](https://www.openslr.org/31/) (CC BY 4.0) and selected environmental recordings from [ESC-10](https://github.com/karolpiczak/ESC-50/blob/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6/LICENSE) (CC BY 3.0). It **does not download train-clean-100 or the full ESC-50/MUSAN archives**. One speech archive is about 332 MB; smoke mode still needs that archive once, but extracts only selected utterances and downloads four noise WAVs (about 1.8 MB). Reuse its verified archive for a larger selection.

| Preparation mode | Speakers train/valid/test | Utterances per speaker | Noise originals train/valid/test |
|---|---|---|---|
| `smoke` (default) | 1 / 1 / 1 | up to 2 | 2 / 1 / 1 |
| `experiment` | 12 / 3 / 3 | up to 20 | 40 / 10 / 10 |

These are new speaker-disjoint splits within Mini LibriSpeech, not official LibriSpeech benchmarks. Noise uses only ESC-10-tagged rain, sea waves, crackling fire, helicopter, chainsaw and clock tick recordings. One clip per original Freesound recording is retained; grouping uses `src_file`, not clip/take names. Classes may occur in all splits; recordings cannot. The small subset is suitable for debugging/development, not strong generalization claims.

From the repository root (Colab cells below):
```bash
python -m scripts.prepare_data --mode smoke --output /content/igcrn_data_smoke --seed 2026
python -m scripts.simulate --catalog /content/igcrn_data_smoke/catalog.jsonl --output /content/igcrn_data_smoke/mixtures --count 9 --seconds 2 --seed 2026
# Larger configurable preparation; reuses the same publisher-checksummed archive:
python -m scripts.prepare_data --mode experiment --output /content/igcrn_data_experiment --speech-archive /content/igcrn_data_smoke/downloads/train-clean-5.tar.gz --speakers 12 3 3 --utterances-per-speaker 20 --noise-recordings 40 10 10 --seed 2026
python -m scripts.simulate --catalog /content/igcrn_data_experiment/catalog.jsonl --output /content/igcrn_data_experiment/mixtures --count 300 --seconds 2 --seed 2026
```
The simulator produces 9/3/3 smoke mixtures or 300/60/60 development mixtures. Preparation itself creates no mixtures and starts no training. `--speech-archive` also supports a manually downloaded archive, but requires the publisher checksum; all other sources are fetched or reused from verified caches. Use a new output directory for a different selection configuration. `python -m scripts.prepare_data --help` lists overrides.

Source IDs, authors/attribution, literal upstream original-license labels, dataset licenses, evidence URLs, original byte hashes and deterministic selection seed are recorded in `catalog.jsonl`; `preparation.json` records source/archive checksums, evidence hashes, selection parameters and actual counts. ESC metadata/audio URLs pin an immutable commit. Source recordings are left at original sample rates; the existing simulator resamples them to 16 kHz. Raw recordings/archives/mixtures stay under `/content/`. Only catalog, preparation/evidence files and mixture manifests are copied to Drive. See [verified sources, attribution and failure policy](docs/DATA_SOURCES.md) and [manifest fields](manifests/README.md).

If a required license statement, per-clip attribution or checksum cannot be verified, preparation exits with an error and does not publish a new catalog. Stop and confirm the identified publisher terms manually; do not fill in guessed licenses. Cached files are rehashed, partial downloads are not accepted, and selected speech paths are checked before extraction.

Silent or near-silent source crops are retried using the same seeded RNG and that split's source pools, with **20 total attempts per sample**. Both reverberant Mic1 speech/noise energies must remain finite and at least `1e-12`; no artificial noise is substituted. Each accepted manifest row records `retry_count` (0 on first-attempt success, at most 19). Failed attempts do not consume sample indices or SNR slots: `--count 600` still produces 600/120/120 entries with the repeating −3/0/+3 dB schedule. Exhaustion identifies the split, zero-based sample index, attempts and last source energies/paths.

Simulation publishes the entire dataset only after all three splits succeed, using a hidden sibling staging directory and one atomic directory rename. **Use a new or empty output directory**; existing non-empty outputs are refused to protect frozen data. For example, after an earlier interrupted attempt, use `--output /content/igcrn_data_experiment/mixtures_600` rather than reusing its partial directory. Errors and normal interrupts clean staging files; a force-killed process may leave an explicitly named `.incomplete-*` sibling, never published manifests at the requested output path. Remove such leftovers only after confirming the generating process has stopped. Keep old datasets/manifests for existing checkpoints: the new retry metadata changes manifest hashes, and rejected attempts alter later RNG draws when silence is encountered.

Generated mixtures and targets are IEEE float32 WAVs written with SciPy, without libsndfile's optional `PEAK` chunk and its wall-clock timestamp. Full-file SHA256 comparisons remain strict: the same catalog/seed and numerical environment reproduce bytes even across different write times and output directories. This serialization fix preserves sample precision but changes WAV headers/hashes relative to older generated datasets; keep old frozen data/manifests with their checkpoints. See [the timestamp investigation](docs/REPRODUCIBILITY.md).

Fixed hashes and seed reproduce selection independently of directory ordering. Frozen WAVs, RIR arrays and simulator manifests capture exact mixtures. Noise scaling uses reverberant Mic1 energy; target is clean reverberant Mic1. No target time shifting or dereverberation. Retain attribution if redistributing real speech/noise examples. The legacy `scripts.catalog` remains available for manually supplied full corpora but is not used by the default pipeline.

## Fair independent ablation

For later experiments, set `IGCRN_DATA=/content/igcrn_data_experiment`. Do not launch full training during data preparation. Run both arms from scratch using the same seed, configuration and frozen manifests. The duplicated arm changes only the input channel selection; never initialize it from the dual checkpoint.
```bash
python -m scripts.train --data "$IGCRN_DATA/mixtures" --mode dual --output /content/drive/MyDrive/IGCRN_NIO/dual --config configs/small.yaml
python -m scripts.train --data "$IGCRN_DATA/mixtures" --mode duplicate --output /content/drive/MyDrive/IGCRN_NIO/duplicate --config configs/small.yaml
# Resume at an epoch boundary; use exactly the same config and batch size:
python -m scripts.train --data "$IGCRN_DATA/mixtures" --mode dual --output /content/drive/MyDrive/IGCRN_NIO/dual --config configs/small.yaml --resume /content/drive/MyDrive/IGCRN_NIO/dual/last.pt
```
Checkpoint includes model, optimizer, epoch, RNG states, loader generator, manifests, mode and config. Writes replace a temporary file atomically on local filesystems; Drive disconnection still requires checking the saved file. Resume is exact at completed epoch boundaries; interruption mid-epoch restarts that epoch. Load only trusted checkpoints (`weights_only=False` includes Python RNG state). `best.pt` uses validation loss; use equal-epoch `last.pt` for the primary fixed-budget comparison. Do not extend one arm alone. Seed 2026 is the primary run; repeat both arms with seeds 2027/2028 for uncertainty when resources permit.

Full configuration uses width 64, six layers, Adam 2e-4, 30 epochs, batch 1, 2 s clips. T4-compatible design starts at batch 1 in float32; actual peak VRAM is unmeasured. If out of memory, shorten clips by resimulating the entire shared dataset and restart both arms. `configs/smoke.yaml` is reduced-width debugging only. `python -m scripts.overfit --manifest "$IGCRN_DATA/mixtures/train.jsonl"` checks a real fixed mixture before longer runs.

## Evaluation and submission artifacts

```bash
python -m scripts.evaluate --manifest "$IGCRN_DATA/mixtures/test.jsonl" --dual /content/drive/MyDrive/IGCRN_NIO/dual/last.pt --duplicate /content/drive/MyDrive/IGCRN_NIO/duplicate/last.pt --output /content/drive/MyDrive/IGCRN_NIO/evaluation
```
SI-SDR removes the means, projects estimate onto the reverberant target, and uses no temporal realignment. SI-SDRi subtracts the same utterance's unprocessed Mic1 score; Mic1 SI-SDRi is zero by definition. STOI uses `pystoi`, 16 kHz, `extended=False`. Report macro means overall and at −3/0/+3 dB; include utterance count. Silent targets fail explicitly. Evaluator enforces identical config, training-manifest hashes, epochs and batch size. Metrics are not the paper's SDR/PESQ protocol.

| Method | SI-SDRi (dB) | STOI |
|---|---|---|
| Unprocessed Mic1 | 0 (definition) | pending |
| Mic1 + Mic2 | pending | pending |
| Independently trained Mic1 + Mic1 | pending | pending |

Repeat this table per SNR after evaluation. Publish `per_utterance.csv`, `summary.json`, `provenance.json` and the frozen evaluation manifest. The current [manifest](manifests/evaluation_manifest.jsonl) is intentionally empty. Export includes the first four target/baseline/model audio cases; metric-ranked best/worst candidates are recorded separately for listening. Listen to candidate cases before claiming success/failure. Consider close arrival angles, weak speech and room mismatch in error analysis.

[Four bundled audio diagnostics](examples/README.md) are original synthetic signals, not trained speech examples. Generate with `python -m scripts.make_examples`. Real 3–5 speech cases remain a post-training deliverable with source attribution. [Checkpoint access instructions](docs/CHECKPOINTS.md); [one-page Chinese report](report/report_zh.pdf) and its editable generator `scripts/make_report.py`.

## Colab: exact preparation commands

Open [notebooks/colab.ipynb](notebooks/colab.ipynb) and select GPU/T4. Its default run stops after preparation; `RUN_FULL_TRAINING = False` gates later training/evaluation. Every subprocess uses the repository root, and tests run through `python -m pytest`, preventing `No module named 'scripts'` caused by a different working directory.

Alternatively, run these cells in order. Setup cell:
```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import os, subprocess, sys
repo = Path('/content/IGCRN-DualMic-Speech-Enhancement')
if not repo.exists():
    subprocess.run(['git', 'clone', 'https://github.com/ZhengchenYuan/IGCRN-DualMic-Speech-Enhancement.git', str(repo)], cwd='/content', check=True)
os.chdir(repo)
def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=repo, check=True)
run('-m', 'pip', 'install', '-r', 'requirements.txt')
run('-m', 'pip', 'install', '-e', '.')
run('-m', 'pytest', '-q')
```
Preparation and simulation cell (no training):
```python
mode = 'smoke'  # 'experiment' uses the larger preset
seed = 2026
data = Path('/content') / f'igcrn_data_{mode}'
run('-m', 'scripts.prepare_data', '--mode', mode, '--seed', seed, '--output', data)
run('-m', 'scripts.simulate', '--catalog', data / 'catalog.jsonl', '--output', data / 'mixtures',
    '--count', 9 if mode == 'smoke' else 300, '--seconds', 2, '--seed', seed)
```
Save only reproducibility artifacts to Drive:
```python
import shutil
save = Path('/content/drive/MyDrive/IGCRN_NIO/manifests') / f'{mode}-seed{seed}'
save.mkdir(parents=True, exist_ok=True)
artifacts = [data / 'catalog.jsonl', data / 'preparation.json']
artifacts += [data / 'mixtures' / f'{split}.jsonl' for split in ('train', 'valid', 'test')]
for source in artifacts:
    target = save / source.name
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise RuntimeError(f'Existing saved manifest differs: {target}')
    shutil.copy2(source, target)
shutil.copytree(data / 'evidence', save / 'evidence', dirs_exist_ok=True)
for name, command in [('environment.txt', [sys.executable, '-m', 'pip', 'freeze']),
                      ('commit.txt', ['git', 'rev-parse', 'HEAD'])]:
    with (save / name).open('w') as stream:
        subprocess.run(command, cwd=repo, stdout=stream, check=True)
```
After a runtime reset, regenerate the same source selection and mixtures and compare hashes with saved manifests before resuming. Archive and selected raw-audio download receipts remain local; important source hashes are also in the saved catalog/preparation record. Checkpoints and final evaluation files belong under `/content/drive/MyDrive/IGCRN_NIO/`. Never copy raw corpora or temporary training files there. No credentials are required in code.

## Validation

39 local CPU tests and the original tiny overfit diagnostic passed. Preparation tests use explicit synthetic fixtures; four real selected noise downloads were also checked. The full public speech-archive preparation was not completed locally because of slow transfer. The full model has 1,342,540 parameters. See [validation record and tested environment](docs/VALIDATION.md); the pinned environment and T4 runtime remain unverified.

## Limitations and AI disclosure

This is a focused architectural reproduction with public replacement data and a simplified RIR model, not a verified reproduction of published scores. T4 performance remains to be measured. AI assistance was used to draft implementation, tests, documentation and report; a human should review the math, run experiments and listen to outputs before submission. Pending result fields must be filled only from saved evaluation outputs.
