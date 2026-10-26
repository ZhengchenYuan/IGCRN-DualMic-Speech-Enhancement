The evaluation manifest is intentionally empty until legally obtained data is simulated. It is not an evaluated test set.

`scripts.simulate` generates train/valid/test JSONL with frozen paths, hashes, SNR, source identities, licenses, offsets, geometry, RIR path and gains. Copy the resulting `test.jsonl` here before the final experiment (replace absolute paths when relocating and retain audio hashes). Publish all three manifests and the source catalog with the trained release. Never publish private file paths or private source metadata.

Input catalog JSONL, one source per row:
```json
{"path":"/content/data/LibriSpeech/train-clean-100/19/198/19-198-0000.flac","kind":"speech","split":"train","speaker_id":"19","license":"CC BY 4.0","source_url":"https://www.openslr.org/12"}
{"path":"/content/data/musan/noise/free-sound/noise-free-sound-0000.wav","kind":"noise","split":"train","recording_id":"noise-free-sound-0000","license":"VERIFY INDIVIDUAL SOURCE LICENSE","source_url":"https://www.openslr.org/17"}
```
The second row is a schema illustration, not clearance to use that recording. Supply at least one speech and noise recording in each split; use disjoint speakers and noise recordings. The simulator rejects identity leakage and duplicate source hashes. Curate enough speakers/recordings for meaningful results.
