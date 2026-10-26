# Stage-one acceptance record — 2026-10-27

Status: implementation and local acceptance complete; **NOT READY to start the full overnight experiment until actual Colab V1 inputs are restored and their gates pass**. No production V2 generation/training was launched. Tiny reduced-model synthetic optimizer tests were used only to verify paired scheduling/resume.

## V1 evidence

All 25 supplied V1 asset files were hashed before/after inspection and remained unchanged (see `v1_assets_integrity.json`; paths there are relative asset names). The supplied frozen Dual checkpoint hash is `3f002a46900bc754196e0e395457e70bbea5dbcfc06f88924de683208ed4bd3a`, epoch 29. The test-manifest hash is `08d960e741133f37b711606a954a51c15a944e904701d3f910a1bb2f98091f5e`.

The 120 manifest source-angle pairs, interpreted with the documented V1 geometry, predict speech/noise TDOA separation quantiles of approximately 0.0357 / 0.3623 / 0.8370 / 1.2151 / 1.8569 samples (min/Q1/median/Q3/max). Six cases are below 0.1 sample and 21 below 0.25 sample. These are **geometric predictions**, not measured isolated-source or mixture delays; they cannot establish the cause of V1's metric tradeoff.

The manifest points to Colab `/content/igcrn_experiment_v2/...` synchronized mixture WAVs. Those files are absent locally; the supplied evaluation directory has mono exports, not the original Mic2. Consequently actual waveform correlation, mixture TDOA/IPD/ILD and frozen-model normal-vs-duplicate-vs-zero Mic2 sensitivity are **UNVERIFIED (0/120 audio cases)**. The diagnostic did not substitute mono exports or synthesize missing Mic2. The local assets also do not supply the complete original train/valid manifests/raw sources or a verifiable V1 training-code commit. V1 identity protection for a production registry cannot yet be certified locally. See `v1_diagnostic.json`.

## Actual PRA physical acceptance

Official pyroomacoustics 0.10.0 macOS universal2 wheel was downloaded and checked against the publisher SHA256 `bbf2576cfe82ace938d04489168d3c438ae748207d9e02eb8850d8f291018856`, then installed offline into a temporary environment. Initial automatic install permission reviews timed out; the public metadata/wheel download and offline route succeeded. No dataset download was needed.

Legacy, PRA direct and PRA first-order probes ran at 0°, 22.5°, 45°, 90°. PRA direct-field phase-slope TDOA error was at most about **0.000087 sample**; 4 kHz IPD error at most about **0.0481°**, below the declared 0.05-sample / 3° tolerances. Full multipath phase is not judged against a single free-field delay. See `acoustics_acceptance.json`.

| Requested RT60 | Actual ISM order | Measured decay across four paths | Probe CPU time |
|---|---:|---:|---:|
| 0.15 s | 20 | about 0.1307–0.1309 s | about 0.011 s |
| 0.25 s | 33 | about 0.2540–0.2545 s | about 0.030 s |
| 0.35 s | 46 | about 0.3854–0.3870 s | about 0.083 s |

Times are local probe measurements, not Colab guarantees. No acoustic parameters were silently changed to make requested and measured RT60 coincide.

## Small actual-PRA generation acceptance

Two independent generations, each 3 train / 3 valid / 3 test samples, covered all three RT60 requests and SNR values. Inputs were explicitly original Gaussian **test fixtures, not human speech, V1 audio, or a research dataset**. Both runs independently built RIR caches and produced identical complete WAV and RIR hashes. Maximum measured SNR error: about **6.1e-9 dB**. Maximum error against independently reconstructed reverberant Mic1 target: about **2.98e-8**. See `small_generation_acceptance.json`.

Source selection, geometry creation and geometry assignment use distinct seeded streams. Shuffled geometry cycles prevent the RT60 cycle from becoming confounded with the SNR cycle. Tests assert all RT60×SNR cells are present and margins balanced in the approved production schedule. No model-performance condition is used for acceptance.

## Tests, environment and remaining gates

Full local suite: **61 passed**, including actual PRA tests (no skipped PRA acceptance), RIR cache/recomputation, real-PRA target/SNR/determinism, split locks/checkpoint-manifest binding, silent retry exhaustion, source protection, paired optimizer resume matching uninterrupted model tensors bit-for-bit, first/late pair interruption, budget non-reset, evaluation timeout and provenance guards. Notebook code cells compile, and the production CLI defaults to PLAN_ONLY.

Local test environment: Python 3.10, PyTorch 2.3.1, NumPy 1.26.4, SciPy 1.13.1, SoundFile 0.13.1, pyroomacoustics 0.10.0. This differs from pinned Colab-oriented requirements. Colab/T4 throughput, resident-model VRAM, mounted-Drive transfer semantics and a full 3000/300/300 real-source run are **not locally verified**.

Production execution additionally requires physical acceptance PASS, actual V1 dual-channel diagnostic VERIFIED, all three original V1 manifests bound to the frozen checkpoint, restored legal source files, a production dataset integrity check, and explicit `RUN_OVERNIGHT=True` / `--execute`. A fixture or missing-audio report cannot satisfy those gates. Stop here until the user approves the next stage.
