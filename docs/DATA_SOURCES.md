# Public-data sources and license verification

Verified against publisher pages and upstream attribution on 2026-10-26. The pipeline uses these sources only; no guessed individual license versions or scraped private recordings are introduced.

## Speech: Mini LibriSpeech

- Publisher: https://www.openslr.org/31/
- Archive: https://www.openslr.org/resources/31/train-clean-5.tar.gz (about 332 MB, five-hour regression-testing subset).
- Publisher checksums: https://www.openslr.org/resources/31/md5sum.txt
- Published archive MD5: `5df7d4e78065366204ca6845bb08f490`.
- Dataset license: CC BY 4.0, explicitly stated on SLR31. Terms: https://creativecommons.org/licenses/by/4.0/
- Attribution: Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur, *LibriSpeech: an ASR corpus based on public domain audio books*, ICASSP 2015. Underlying LibriSpeech: https://www.openslr.org/12/

The script saves the publisher page and checksum list as evidence, verifies the license statement and expected MD5 before extraction, and records its own archive SHA256. MD5 is publisher-provided integrity evidence, not a modern authenticity guarantee; HTTPS and SHA256 receipts provide additional provenance. Catalog IDs retain original LibriSpeech speaker/chapter/utterance IDs through the archive member and utterance filename. Only selected regular FLAC members are extracted. There is no default train-clean-100, dev-clean or test-clean download.

## Environmental noise: ESC-10 subset of ESC-50

- Publisher repository: https://github.com/karolpiczak/ESC-50
- Pinned revision: `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`.
- License and individual attribution evidence: https://raw.githubusercontent.com/karolpiczak/ESC-50/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6/LICENSE
- Metadata: https://raw.githubusercontent.com/karolpiczak/ESC-50/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6/meta/esc50.csv
- Dataset license for **ESC-10-tagged clips only**: CC BY 3.0 Unported. Terms: https://creativecommons.org/licenses/by/3.0/
- Attribution: Karol J. Piczak, *ESC: Dataset for Environmental Sound Classification*, ACM Multimedia 2015, DOI https://doi.org/10.1145/2733373.2806390, plus each original recording author as listed in the upstream LICENSE.

The complete ESC-50 dataset is CC BY-NC 3.0; that is **not** the license asserted for all clips by this project. We filter `esc10=True` and environmental categories `rain`, `sea_waves`, `crackling_fire`, `helicopter`, `chainsaw`, `clock_tick`. The LICENSE explicitly grants CC BY 3.0 for ESC-10 and lists each derived clip's original Freesound URL, title, author and license label. Only clips with an explicit upstream `CC0` or `CC-BY` original label are accepted. An absent or unsupported attribution stops preparation; do not infer permission from the category or the repository's top-level license alone.

Original `CC-BY` labels have no version in this evidence. `original_license_label` retains that literal label without assigning a version. The catalog's `license=CC-BY-3.0` refers to the distributed ESC-10 subset, not an invented version of the original Freesound license. Likewise, the pipeline records Freesound URLs from upstream LICENSE; it does not claim to have independently checked currently live Freesound pages or fabricate original-file hashes. `sha256` is the hash of the actual distributed ESC WAV or Mini LibriSpeech FLAC bytes downloaded/extracted, not the original untrimmed Freesound upload.

One derived WAV is selected per original `src_file` (lexicographically first eligible take). Splits group by original `src_file`, preventing take-level leakage. Files are fetched individually from commit-pinned `audio/<filename>` URLs, not the roughly 600 MB ESC-50 ZIP. Original sample rate is retained; the simulator converts to 16 kHz before RIR convolution.

## Determinism, storage and manual-confirmation policy

Selection ranks source identities by SHA256 of `seed:namespace:identity`, independent of directory listing order or NumPy version. Speakers/noise originals are assigned disjointly in train/valid/test order. Changing budgets can change split assignments: freeze catalog/config/hash for a comparison and use a new output directory for any change. Smoke mode is a pipeline check, not a statistically meaningful benchmark.

`catalog.jsonl` records selected original IDs, distributed-file hashes, source/evidence URLs, licenses and attribution. `preparation.json` records configuration, counts, archive/evidence hashes and catalog hash. Cached downloads are checked against SHA256 receipts; speech is additionally checked against the publisher MD5. Partial files are never accepted as complete. Cached evidence allows reproducible reruns of the checked snapshot; it is not a promise to re-check later legal changes.

Raw archives/recordings, RIRs and mixtures default to `/content/igcrn_data_<mode>/`; raw output paths under Google Drive are rejected. Colab copies only important manifests, evidence, configs/environment information, checkpoints and final results into `/content/drive/MyDrive/IGCRN_NIO/`. Local tests use temporary directories and no external downloads.

If evidence is unavailable, missing a required statement, contradictory, or a selected source lacks supported attribution, the command stops. Restore network access for unavailable evidence. For licensing ambiguity, manually confirm the relevant publisher terms and original attribution before changing the allowed-source policy; do not insert a plausible license string to bypass the check. No partial catalog is published on a failed first run.
