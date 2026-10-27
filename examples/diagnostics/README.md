# Synthetic diagnostics

These four original chirp/noise signals test signal construction only. They are not speech recordings, trained-model outputs or evidence for V1 enhancement results. [The manifest](manifest.json) records the original WAV hashes; moving the files here did not change their contents.

Real held-out speech examples are indexed in [V1 Audio Examples](../README.md). `python -m scripts.make_examples` regenerates only this diagnostic directory and its manifest; generated containers can differ from the original retained WAVs, so do not use regeneration to replace frozen artifacts.
