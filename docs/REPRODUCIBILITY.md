# Investigation: identical audio, different WAV hashes

## Reproduction and root cause

Colab reported two failures after commit `9b451f9` in the speech-silence recovery parametrizations. The mismatch was in complete-file mixture/target SHA256 hashes, despite matching source selection and other metadata.

An isolated local reproduction using SoundFile 0.13.1 and libsndfile 1.2.2 wrote the same waveform twice with a real 1.1-second delay. Decoded float32 samples were exactly equal. Parsing RIFF chunks showed matching `fmt `, `fact` and audio `data` payloads; only the timestamp inside `PEAK` differed. The full-file SHA256 therefore correctly differed. A delayed simulator regression reproduced the same failure before the fix, with the first changed byte at offset 60 in that particular WAV layout (the test does not rely on a fixed offset).

The upstream libsndfile implementation writes `time(NULL)` into the optional PEAK chunk of FLOAT WAV files:

- Implementation: https://github.com/libsndfile/libsndfile/blob/1.2.2/src/wavlike.c#L1158-L1167
- FLOAT WAV default PEAK-chunk behavior: https://libsndfile.github.io/libsndfile/command.html#sfc_set_add_peak_chunk

Fast tests could write both outputs within one second and pass. Crossing a second boundary could make any such comparison fail; the two speech parametrizations are not evidence that the speech math or silence retries were nondeterministic. Seeding Python/NumPy cannot control libsndfile's C wall clock.

## Checks performed

1. **Randomness:** source selection, offsets and angles use the one `np.random.default_rng(args.seed)` created for each simulation. No unseeded simulation RNG was found. Temporary-directory names are intentionally random but are never embedded in generated WAVs or manifest paths.
2. **Sources:** simulation opens catalog audio only for reading. The delayed regression snapshots and rechecks every source-file hash after both runs. Input files are neither overwritten nor recreated between simulations. Test fixtures create their source WAVs once; those original bytes are reused.
3. **Signal processing:** decoded mono/stereo output samples matched exactly in the failing reproduction. No FFT, resampling, floating-point math or retry-order change was needed. Bitwise numerical equivalence across different NumPy/SciPy versions, hardware or builds is not claimed.
4. **Writing and hashing:** hashing the full file was correct; the old writer added time-dependent container metadata. The existing strict hash assertions are retained.
5. **Paths:** identical inputs were written to different output/staging directories. Paths were not responsible for the data/header difference; the timestamp was. With the fix, every complete WAV byte matches across the different directories and times.

## Fix and regression

Mixtures and targets now use `scipy.io.wavfile.write` with explicit float32 arrays. This public writer emits IEEE float32 WAV without the optional timestamped PEAK chunk. It retains the previous FLOAT sample precision and 16 kHz sample rate. A separate mono/stereo precision check confirmed that decoded samples match the previous SoundFile writer exactly. There is no integer quantization, timestamp patching, rounding tolerance, sample-only hashing, or private SoundFile API dependency.

The new regression deliberately waits 1.1 seconds between two complete simulations using one catalog/seed, with silent-speech retries and different output directories. It asserts source immutability, expected row counts, exact decoded samples, complete WAV bytes, full-file hashes, FLOAT subtype, absence of a PEAK chunk using RIFF chunk parsing, and complete manifest equivalence apart from generated-output paths. Python clock monkeypatches would not affect C `time(NULL)`, so a real delay is necessary to expose the old failure reliably.

Old generated files remain valid audio, but regenerated headers and full-file hashes change with the new serialization. Do not edit saved manifest hashes or regenerate data beneath existing checkpoints. Keep old frozen datasets for old checkpoints; create a new output directory for new experiments and retain the code/environment versions with the manifests.
